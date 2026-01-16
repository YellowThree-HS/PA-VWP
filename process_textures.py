"""
纹理图片预处理脚本
- 统一重命名为 cardboard_XX.jpg
- 统一调整尺寸为 512x512
- 均匀化光照（高斯除法 + CLAHE）
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# 默认配置
DEFAULT_INPUT_DIR = Path("assets/cardboard_textures")
DEFAULT_OUTPUT_DIR = Path("assets/cardboard_textures_processed")
TARGET_SIZE = (512, 512)


def normalize_lighting(img, blur_size=101):
    """高斯除法去除低频光照渐变"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    blurred = cv2.GaussianBlur(l_channel, (blur_size, blur_size), 0)
    mean_val = np.mean(l_channel)
    normalized = (l_channel / (blurred + 1e-6)) * mean_val
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    lab[:, :, 0] = normalized
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe(img, clip_limit=2.0, grid_size=8):
    """CLAHE 增强局部对比度"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fix_lighting(img):
    """完整光照处理：高斯除法 + CLAHE"""
    result = normalize_lighting(img)
    result = apply_clahe(result)
    return result


def process_textures(input_dir, output_dir):
    """处理纹理：调整尺寸 + 均匀光照"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        print(f"目录中没有找到图片: {input_dir}")
        return

    print(f"找到 {len(image_files)} 张图片")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    for idx, img_path in enumerate(sorted(image_files), start=1):
        try:
            # 用 PIL 读取并裁剪/缩放
            pil_img = Image.open(img_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            original_size = pil_img.size

            # 裁剪为正方形（取中心区域）
            width, height = pil_img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            pil_img = pil_img.crop((left, top, left + min_dim, top + min_dim))

            # 调整尺寸
            pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)

            # 转换为 OpenCV 格式进行光照处理
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 光照均匀化
            cv_img = fix_lighting(cv_img)

            # 保存
            new_name = f"cardboard_{idx:02d}.jpg"
            output_path = output_dir / new_name
            cv2.imwrite(str(output_path), cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"[{idx:02d}] {img_path.name} {original_size} -> {TARGET_SIZE}")

        except Exception as e:
            print(f"[错误] {img_path.name}: {e}")

    print("-" * 40)
    print(f"完成！共处理 {len(image_files)} 张纹理")


def main():
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        input_dir = sys.argv[1]
        output_dir = str(Path(input_dir).parent / (Path(input_dir).name + "_processed"))
    else:
        input_dir = DEFAULT_INPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR

    process_textures(input_dir, output_dir)


if __name__ == "__main__":
    main()
