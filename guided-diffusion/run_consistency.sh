#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="../models/256x256_diffusion_uncond.pt" # 您的 SD 模型路径

# 输出目录
OUTPUT_DIR="../results/LSUN/fake"
# 输入图片目录
IMAGES_DIR="../data/LSUN/1_fake"

SAMPLE_FLAGS="--batch_size 32 --timestep_respacing ddim100 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python -u compute_trajectory_consistency.py \
  --model_path $MODEL_PATH $MODEL_FLAGS $SAMPLE_FLAGS \
  --images_dir $IMAGES_DIR --output_dir $OUTPUT_DIR \
  --num_samples 1000