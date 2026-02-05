"""
compress_dataset.py
功能: 对数据集进行 JPEG 压缩处理 (Robustness Test)
说明:
1. 读取原始 PNG/JPG 图片。
2. 使用指定的 Quality (如 50) 进行 JPEG 压缩。
3. 保存到新的文件夹，用于后续的鲁棒性测试。
"""

import os
import glob
from PIL import Image
from tqdm import tqdm # 如果没有安装 tqdm，可以注释掉或用简单的 print 代替

# ================= 配置区域 =================
# 原始数据路径 (请修改为您实际的路径)
SRC_ROOT = "../data11/LSUN" 
# 目标保存路径 (会自动创建)
DST_ROOT = "../data11/LSUN_JPEG50" 

# 压缩质量 (越低越模糊，50 是一个较强的攻击，70 是常见网络质量)
JPEG_QUALITY = 50 

# 需要处理的子文件夹
SUB_DIRS = ["0_real", "1_fake"] # 或者是 ["1_fake", "0_real"] 取决于您的目录结构
# ===========================================

def compress_images(src_dir, dst_dir, quality):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"Created directory: {dst_dir}")

    # 支持常见的图片格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(src_dir, ext)))
    
    # 排序以保证顺序
    files.sort()
    
    print(f"Processing {len(files)} images from {src_dir} ...")
    
    for file_path in tqdm(files, desc=f"Compressing (Q={quality})"):
        try:
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            
            # 打开图片
            img = Image.open(file_path).convert("RGB")
            
            # 构建新路径 (改为 .jpg 后缀)
            # 注意: 如果您的特征提取代码写死了读取 .png，您可能需要修改特征提取代码
            # 或者这里可以玩个花招: 先存成 jpg bytes 再存回 png (模拟压缩伪影但保持后缀)
            # 这里我们采用标准的 .jpg 保存
            new_filename = f"{name}.jpg"
            save_path = os.path.join(dst_dir, new_filename)
            
            # 保存为 JPEG
            img.save(save_path, "JPEG", quality=quality)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    print(f"=== JPEG Compression Tool (Quality={JPEG_QUALITY}) ===")
    
    for sub in SUB_DIRS:
        src = os.path.join(SRC_ROOT, sub)
        dst = os.path.join(DST_ROOT, sub)
        
        if os.path.exists(src):
            compress_images(src, dst, JPEG_QUALITY)
        else:
            print(f"[Warning] Source directory not found: {src}")
            # 尝试自动寻找可能的目录名 (兼容 1_fake/0_real 这种命名)
            possible_dirs = glob.glob(os.path.join(SRC_ROOT, "*"))
            print(f"Available directories in {SRC_ROOT}: {[os.path.basename(d) for d in possible_dirs]}")

    print("\nDone! Next steps:")
    print("1. Please run your 'Feature Extraction' script on this new dataset folder.")
    print(f"   (Source path for extraction: {DST_ROOT})")
    print("2. Run the classifier training script again to check robustness.")

if __name__ == "__main__":
    main()
