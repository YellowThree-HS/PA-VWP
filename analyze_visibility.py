"""分析可见性统计数据，帮助确定合理的阈值"""
import json
import glob
import numpy as np

def load_all_stats():
    """加载所有视角的统计数据"""
    all_data = []
    files = glob.glob("dataset/**/visibility_stats.json", recursive=True)
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            for item in data:
                item['source_file'] = f
            all_data.extend(data)
    return all_data

def analyze(data):
    """分析数据分布"""
    ratios = [d['visibility_ratio'] for d in data]
    pixels = [d['actual_pixels'] for d in data]

    print("=" * 60)
    print("可见性统计分析")
    print("=" * 60)
    print(f"总样本数: {len(data)}")
    print()

    # 基本统计
    print("【visibility_ratio 分布】")
    print(f"  最小值: {min(ratios):.4f}")
    print(f"  最大值: {max(ratios):.4f}")
    print(f"  平均值: {np.mean(ratios):.4f}")
    print(f"  中位数: {np.median(ratios):.4f}")
    print(f"  标准差: {np.std(ratios):.4f}")
    print()

    print("【actual_pixels 分布】")
    print(f"  最小值: {min(pixels)}")
    print(f"  最大值: {max(pixels)}")
    print(f"  平均值: {np.mean(pixels):.1f}")
    print(f"  中位数: {np.median(pixels):.1f}")
    print()

    # 分段统计 visibility_ratio
    print("【visibility_ratio 分段统计】")
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    for i in range(len(bins) - 1):
        count = sum(1 for r in ratios if bins[i] <= r < bins[i+1])
        pct = count / len(ratios) * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:4d} ({pct:5.1f}%)")
    print()

    # 分段统计 actual_pixels
    print("【actual_pixels 分段统计】")
    pixel_bins = [0, 10, 20, 40, 100, 200, 500, 1000, 2000, 5000, 100000]
    for i in range(len(pixel_bins) - 1):
        count = sum(1 for p in pixels if pixel_bins[i] <= p < pixel_bins[i+1])
        pct = count / len(pixels) * 100
        print(f"  [{pixel_bins[i]:5d}, {pixel_bins[i+1]:5d}): {count:4d} ({pct:5.1f}%)")
    print()

    # 问题分析：ratio > 1 的情况（理论上不应该超过1）
    over_one = [d for d in data if d['visibility_ratio'] > 1.0]
    print(f"【异常：visibility_ratio > 1.0 的样本】")
    print(f"  数量: {len(over_one)} ({len(over_one)/len(data)*100:.1f}%)")
    if over_one:
        avg_ratio = np.mean([d['visibility_ratio'] for d in over_one])
        print(f"  平均 ratio: {avg_ratio:.2f}")
    print()

    # 低可见度样本分析
    print("【低可见度样本 (ratio < 0.2)】")
    low_vis = [d for d in data if d['visibility_ratio'] < 0.2]
    print(f"  数量: {len(low_vis)} ({len(low_vis)/len(data)*100:.1f}%)")
    if low_vis:
        low_pixels = [d['actual_pixels'] for d in low_vis]
        print(f"  像素范围: {min(low_pixels)} - {max(low_pixels)}")
        print(f"  像素平均: {np.mean(low_pixels):.1f}")
    print()

    # 极低像素样本
    print("【极低像素样本 (pixels < 50)】")
    tiny = [d for d in data if d['actual_pixels'] < 50]
    print(f"  数量: {len(tiny)} ({len(tiny)/len(data)*100:.1f}%)")
    if tiny:
        tiny_ratios = [d['visibility_ratio'] for d in tiny]
        print(f"  ratio 范围: {min(tiny_ratios):.4f} - {max(tiny_ratios):.4f}")
    print()

    # 建议阈值分析
    print("=" * 60)
    print("【阈值建议分析】")
    print("=" * 60)

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for t in thresholds:
        visible = sum(1 for r in ratios if r >= t)
        print(f"  ratio >= {t:.2f}: 保留 {visible:4d} 个 ({visible/len(data)*100:.1f}%)")
    print()

    pixel_thresholds = [0, 10, 20, 30, 40, 50, 100]
    for pt in pixel_thresholds:
        visible = sum(1 for p in pixels if p >= pt)
        print(f"  pixels >= {pt:3d}: 保留 {visible:4d} 个 ({visible/len(data)*100:.1f}%)")
    print()

    # 组合阈值
    print("【组合阈值效果】")
    combos = [
        (0.1, 0), (0.1, 20), (0.1, 40),
        (0.15, 0), (0.15, 20), (0.15, 40),
        (0.2, 0), (0.2, 20), (0.2, 40),
    ]
    for ratio_t, pixel_t in combos:
        visible = sum(1 for d in data if d['visibility_ratio'] >= ratio_t and d['actual_pixels'] >= pixel_t)
        print(f"  ratio >= {ratio_t:.2f} AND pixels >= {pixel_t:2d}: 保留 {visible:4d} 个 ({visible/len(data)*100:.1f}%)")

if __name__ == "__main__":
    data = load_all_stats()
    if data:
        analyze(data)
    else:
        print("未找到 visibility_stats.json 文件")
