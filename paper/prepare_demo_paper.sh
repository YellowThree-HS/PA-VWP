#!/usr/bin/env bash

set -e

# 把 paper/demo_paper 里的文件整理成 inference_rgbd_real.py 需要的目录结构

PROJ_ROOT="/home/hs_25/projs/PA-VWP"
SRC_DIR="$PROJ_ROOT/paper/demo_paper"
DST_DIR="$PROJ_ROOT/real_images"

echo "源目录: $SRC_DIR"
echo "目标目录: $DST_DIR"

# 创建目标目录及子目录
mkdir -p "$DST_DIR/box_masks"

# 拷贝 / 覆盖 RGB 与深度
cp "$SRC_DIR/capture_rgb.png" "$DST_DIR/capture_rgb.png"
cp "$SRC_DIR/capture_depth.npy" "$DST_DIR/capture_depth.npy"

echo "已复制 RGB 和 depth 到 $DST_DIR"

# 将 mask_X.png 重命名为 box_XX.png 复制到 box_masks
idx=1
for mask_path in "$SRC_DIR"/mask_*.png; do
  [ -e "$mask_path" ] || continue
  # 生成两位数编号
  printf -v box_id "%02d" "$idx"
  dst_mask="$DST_DIR/box_masks/box_${box_id}.png"
  cp "$mask_path" "$dst_mask"
  echo "复制: $(basename "$mask_path") -> $(basename "$dst_mask")"
  idx=$((idx + 1))
done

echo "整理完成，现在可以在项目根目录运行:"
echo "  python inference_rgbd_real.py --checkpoint_dir /DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100 --image_dir /home/hs_25/projs/PA-VWP/real_images --device cuda --output_dir paper_demo_results"

