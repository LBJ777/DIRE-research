"""
compute_trajectory_consistency.py 
Features:
1. Hybrid Detection: Computes both Spatial Consistency (Method A) and Gaussianity Stats (Method B).
2. Metrics Output: [Mean_Sim, Min_Sim, Std_Sim, Avg_Abs_Skew, Avg_Abs_Kurt] (5 dims).
3. Optimized for purely PyTorch execution.
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.distributed as dist
from mpi4py import MPI

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# === 核心配置 ===
DEFAULT_GRID_SIZE = 4 
# =============

def count_image_files(directory):
    """递归统计目录下所有图片文件的数量"""
    count = 0
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                count += 1
    return count

def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs

def compute_hybrid_metrics(latents, grid_size=DEFAULT_GRID_SIZE):
    """
    计算混合特征指标 (Hybrid Metrics).
    Returns: [B, 5] -> (Con_Mean, Con_Min, Con_Std, Gau_Skew, Gau_Kurt)
    """
    B, C, H, W = latents.shape
    
    # --- 1. 预处理：切分成 Patches ---
    target_h = (H // grid_size) * grid_size
    target_w = (W // grid_size) * grid_size
    
    if H != target_h or W != target_w:
        latents = F.interpolate(latents, size=(target_h, target_w), mode="bilinear")
        H, W = target_h, target_w

    patch_h, patch_w = H // grid_size, W // grid_size

    # [B, C, grid_h, patch_h, grid_w, patch_w] -> [B, N_patches, Flattened_Dim]
    patches = latents.view(B, C, grid_size, patch_h, grid_size, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches_flat = patches.reshape(B, grid_size * grid_size, -1) 

    # --- 2. Part A: 一致性检测 (Consistency Check) ---
    patches_norm = F.normalize(patches_flat, p=2, dim=2)
    sim_matrix = torch.bmm(patches_norm, patches_norm.transpose(1, 2)) # [B, N, N]
    
    N = grid_size * grid_size
    indices = torch.triu_indices(N, N, offset=1, device=latents.device)
    
    consistency_stats = []
    for b in range(B):
        mat = sim_matrix[b]
        valid_scores = mat[indices[0], indices[1]]
        
        c_mean = valid_scores.mean()
        c_min = valid_scores.min()  # 捕捉几何崩塌
        c_std = valid_scores.std()  # 捕捉噪声不均
        consistency_stats.append(torch.stack([c_mean, c_min, c_std]))
    
    consistency_stats = torch.stack(consistency_stats) # [B, 3]

    # --- 3. Part B: 高斯性检测 (Gaussianity Check) ---
    p_mean = patches_flat.mean(dim=2, keepdim=True)
    p_std = patches_flat.std(dim=2, keepdim=True) + 1e-6 # 防止除零
    
    # 标准化数据 (z-score)
    z_scores = (patches_flat - p_mean) / p_std
    
    # 计算三阶矩和四阶矩 [B, N]
    skewness = torch.mean(z_scores ** 3, dim=2)
    kurtosis = torch.mean(z_scores ** 4, dim=2) - 3.0 # Excess Kurtosis
    
    avg_abs_skew = torch.mean(torch.abs(skewness), dim=1) # [B]
    avg_abs_kurt = torch.mean(torch.abs(kurtosis), dim=1) # [B]
    
    gaussianity_stats = torch.stack([avg_abs_skew, avg_abs_kurt], dim=1) # [B, 2]

    # --- 4. 合并输出 ---
    return torch.cat([consistency_stats, gaussianity_stats], dim=1)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.configure(dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.log(str(args))

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    
    if args.num_samples > 0:
        total_samples = args.num_samples
    else:
        logger.log(f"Counting files in {args.images_dir} to determine dataset size...")
        total_samples = count_image_files(args.images_dir)
        logger.log(f"Detected {total_samples} images in total.")

    logger.log(f"Computing Hybrid Metrics with Grid {DEFAULT_GRID_SIZE}x{DEFAULT_GRID_SIZE}...")
    have_finished_images = 0

    while have_finished_images < total_samples:
        # 动态调整最后一个 batch 的大小，防止超出 total_samples
        remaining = total_samples - have_finished_images
        world_size = MPI.COMM_WORLD.size
        
        # 如果剩余不足一个完整 batch 或刚好的逻辑
        if remaining < args.batch_size * world_size:
             batch_size = args.batch_size 
        else:
             batch_size = args.batch_size
            
        try:
            imgs, out_dicts, paths = next(data)
        except StopIteration:
            break

        imgs = imgs[:batch_size].to(dist_util.dev())
        paths = paths[:batch_size]
        
        actual_batch_size = imgs.shape[0]
        if actual_batch_size == 0: break
        
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(low=0, high=NUM_CLASSES, size=(actual_batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
            
        imgs = reshape_image(imgs, args.image_size)

        reverse_fn_progressive = diffusion.ddim_reverse_sample_loop_progressive
        trajectory_generator = reverse_fn_progressive(
            model,
            (actual_batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        batch_trajectory_metrics = []
        for i, sample in enumerate(trajectory_generator):
            latent = sample['sample']
            metrics = compute_hybrid_metrics(latent, grid_size=DEFAULT_GRID_SIZE)
            batch_trajectory_metrics.append(metrics)

        # [Time, B, 5] -> [B, Time, 5]
        batch_trajectory_metrics = torch.stack(batch_trajectory_metrics, dim=0).permute(1, 0, 2)
        batch_scores_np = batch_trajectory_metrics.cpu().numpy()
        
        for idx in range(len(batch_scores_np)):
            if args.has_subfolder:
                relative_path = os.path.join(paths[idx].split("/")[-2])
                save_dir = os.path.join(args.output_dir, relative_path)
            else:
                save_dir = args.output_dir
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(paths[idx]).split(".")[0] + ".npy"
            np.save(os.path.join(save_dir, filename), batch_scores_np[idx])

        # 更新已处理计数
        have_finished_images += batch_size * dist.get_world_size()
        logger.log(f"have processed {min(have_finished_images, total_samples)} / {total_samples} images")

    dist.barrier()
    logger.log("finish computing hybrid metrics!")

def create_argparser():
    defaults = dict(
        images_dir="",
        output_dir="",
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=True,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()