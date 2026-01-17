"""
修复数据集相机视角上下颠倒问题
垂直翻转所有图像和numpy数组，并修复bbox坐标
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path

DATASET_DIR = r"C:\projs\BoxWorld-MVP\dataset"
IMAGE_HEIGHT = 480  # 图像高度 (640x480)


def flip_image_vertically(img):
    """垂直翻转图像"""
    return cv2.flip(img, 0)


def flip_npy_vertically(arr):
    """垂直翻转numpy数组"""
    return np.flip(arr, axis=0)


def fix_bbox_y(bbox):
    """翻转bbox的y坐标"""
    y1 = bbox["y1"]
    y2 = bbox["y2"]
    bbox["y1"] = IMAGE_HEIGHT - 1 - y2
    bbox["y2"] = IMAGE_HEIGHT - 1 - y1
    return bbox


def process_visible_boxes(filepath):
    """处理 visible_boxes.json 中的 bbox"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for item in data:
        if "bbox" in item:
            item["bbox"] = fix_bbox_y(item["bbox"])
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    return False


def process_view_folder(view_dir):
    """处理单个视角文件夹"""
    fixed_count = 0

    # 1. 处理根目录 PNG 文件
    root_png_files = ["rgb.png", "depth.png", "visible_mask.png", "summary.png"]
    for filename in root_png_files:
        filepath = os.path.join(view_dir, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                cv2.imwrite(filepath, flip_image_vertically(img))
                fixed_count += 1

    # 2. 处理根目录 NPY 文件
    root_npy_files = ["depth.npy", "mask.npy"]
    for filename in root_npy_files:
        filepath = os.path.join(view_dir, filename)
        if os.path.exists(filepath):
            arr = np.load(filepath)
            np.save(filepath, flip_npy_vertically(arr))
            fixed_count += 1

    # 3. 处理 visible_boxes.json
    vb_path = os.path.join(view_dir, "visible_boxes.json")
    if os.path.exists(vb_path):
        if process_visible_boxes(vb_path):
            fixed_count += 1

    # 4. 处理 removals 目录
    removals_dir = os.path.join(view_dir, "removals")
    if os.path.exists(removals_dir):
        for entry in os.listdir(removals_dir):
            item_path = os.path.join(removals_dir, entry)
            if not os.path.isdir(item_path):
                continue

            # 处理 PNG 文件
            png_files = ["affected_mask.png", "removed_mask.png", "annotated.png"]
            for filename in png_files:
                filepath = os.path.join(item_path, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        cv2.imwrite(filepath, flip_image_vertically(img))
                        fixed_count += 1

            # result.json 只包含世界坐标，不需要处理

    return fixed_count


def main():
    dataset_path = Path(DATASET_DIR)

    # 收集所有视角文件夹
    view_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name.startswith("round_"):
            view_folders.append(item)

    view_folders.sort()

    print(f"Dataset: {DATASET_DIR}")
    print(f"Image height: {IMAGE_HEIGHT}")
    print(f"Found {len(view_folders)} view folders")
    print()

    total_fixed = 0

    for view_folder in view_folders:
        view_name = view_folder.name
        fixed = process_view_folder(view_folder)
        total_fixed += fixed
        print(f"  {view_name}: {fixed} files fixed")

    print()
    print(f"Total: {total_fixed} files fixed")
    print("Done!")


if __name__ == "__main__":
    main()
