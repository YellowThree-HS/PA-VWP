"""
纹理图光照分析脚本
分析图片的光照均匀性，找出问题所在
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def analyze_image(image_path):
    """分析单张图片的光照情况"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None

    # 转换到不同色彩空间
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]  # 亮度通道

    h, w = gray.shape

    # 1. 整体亮度统计
    mean_brightness = np.mean(l_channel)
    std_brightness = np.std(l_channel)

    # 2. 分区域分析 (3x3 网格)
    region_means = []
    grid_size = 3
    rh, rw = h // grid_size, w // grid_size

    print(f"\n{'='*50}")
    print(f"图片: {image_path.name}")
    print(f"尺寸: {w} x {h}")
    print(f"{'='*50}")

    print(f"\n[整体亮度统计]")
    print(f"  平均亮度: {mean_brightness:.1f} / 255")
    print(f"  亮度标准差: {std_brightness:.1f}")

    print(f"\n[分区域亮度 (3x3网格)]")
    grid_values = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            region = l_channel[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
            region_mean = np.mean(region)
            grid_values[i, j] = region_mean
            region_means.append(region_mean)

    # 打印网格
    print("  " + "-" * 25)
    for i in range(grid_size):
        row = " | ".join([f"{grid_values[i,j]:5.1f}" for j in range(grid_size)])
        print(f"  | {row} |")
    print("  " + "-" * 25)

    # 3. 光照不均匀度分析
    region_std = np.std(region_means)
    region_range = max(region_means) - min(region_means)

    print(f"\n[光照均匀性分析]")
    print(f"  区域间标准差: {region_std:.1f}")
    print(f"  区域间极差: {region_range:.1f}")

    # 4. 判断问题类型
    print(f"\n[问题诊断]")

    issues = []

    # 检查整体过暗/过亮
    if mean_brightness < 80:
        issues.append("整体偏暗")
    elif mean_brightness > 180:
        issues.append("整体过亮")

    # 检查光照不均匀
    if region_range > 40:
        issues.append(f"光照不均匀 (极差={region_range:.0f})")

        # 判断不均匀的模式
        top_mean = np.mean(grid_values[0, :])
        bottom_mean = np.mean(grid_values[2, :])
        left_mean = np.mean(grid_values[:, 0])
        right_mean = np.mean(grid_values[:, 2])
        center_mean = grid_values[1, 1]
        corner_mean = np.mean([grid_values[0,0], grid_values[0,2],
                               grid_values[2,0], grid_values[2,2]])

        if top_mean - bottom_mean > 30:
            issues.append("  -> 上亮下暗")
        elif bottom_mean - top_mean > 30:
            issues.append("  -> 下亮上暗")

        if left_mean - right_mean > 30:
            issues.append("  -> 左亮右暗")
        elif right_mean - left_mean > 30:
            issues.append("  -> 右亮左暗")

        if center_mean - corner_mean > 25:
            issues.append("  -> 中心亮边缘暗 (暗角)")
        elif corner_mean - center_mean > 25:
            issues.append("  -> 边缘亮中心暗")

    # 检查对比度
    if std_brightness < 30:
        issues.append("对比度过低")
    elif std_brightness > 80:
        issues.append("对比度过高")

    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  光照基本均匀，无明显问题")

    return {
        'mean': mean_brightness,
        'std': std_brightness,
        'region_std': region_std,
        'region_range': region_range,
        'issues': issues
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_lighting.py <图片路径或目录>")
        print("示例: python analyze_lighting.py texture.png")
        print("      python analyze_lighting.py ./textures/")
        return

    path = Path(sys.argv[1])

    if path.is_file():
        analyze_image(path)
    elif path.is_dir():
        # 支持的图片格式
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        images = [f for f in path.iterdir()
                  if f.suffix.lower() in extensions]

        if not images:
            print(f"目录中没有找到图片: {path}")
            return

        print(f"找到 {len(images)} 张图片，开始分析...\n")

        all_results = []
        for img_path in sorted(images)[:10]:  # 最多分析10张
            result = analyze_image(img_path)
            if result:
                all_results.append((img_path.name, result))

        # 汇总统计
        if all_results:
            print(f"\n{'='*50}")
            print("汇总统计")
            print(f"{'='*50}")

            problem_images = [r for r in all_results if r[1]['issues']]
            print(f"有问题的图片: {len(problem_images)} / {len(all_results)}")

            avg_range = np.mean([r[1]['region_range'] for r in all_results])
            print(f"平均区域极差: {avg_range:.1f}")
    else:
        print(f"路径不存在: {path}")


if __name__ == "__main__":
    main()
