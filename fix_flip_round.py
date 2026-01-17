"""修复相机视角上下颠倒问题 - 处理整个视角文件夹"""
import cv2
import numpy as np
import json
import os

TARGET_DIR = r"C:\projs\BoxWorld-MVP\dataset\round_001_33_east"
IMAGE_HEIGHT = 480


def flip_image_vertically(img):
    return cv2.flip(img, 0)


def flip_npy_vertically(arr):
    return np.flip(arr, axis=0)


def fix_bbox_y(bbox):
    """翻转bbox的y坐标"""
    y1 = bbox["y1"]
    y2 = bbox["y2"]
    bbox["y1"] = IMAGE_HEIGHT - 1 - y2
    bbox["y2"] = IMAGE_HEIGHT - 1 - y1
    return bbox


def process_visible_boxes(filepath):
    """处理 visible_boxes.json"""
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
        print(f"  Fixed: {os.path.basename(filepath)}")


def process_summary_json(filepath):
    """处理 summary.json（不含bbox，无需修改）"""
    pass


def main():
    print(f"Processing: {TARGET_DIR}")
    print(f"Image height: {IMAGE_HEIGHT}")
    print()

    # 需要处理的 PNG 文件（根目录）
    root_png_files = ["rgb.png", "depth.png", "visible_mask.png", "summary.png"]

    # 需要处理的 NPY 文件（根目录）
    root_npy_files = ["depth.npy", "mask.npy"]

    # 1. 处理根目录的 PNG 文件
    print("Root PNG files:")
    for filename in root_png_files:
        filepath = os.path.join(TARGET_DIR, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                flipped = flip_image_vertically(img)
                cv2.imwrite(filepath, flipped)
                print(f"  Flipped: {filename}")
            else:
                print(f"  Failed to read: {filename}")

    # 2. 处理根目录的 NPY 文件
    print("\nRoot NPY files:")
    for filename in root_npy_files:
        filepath = os.path.join(TARGET_DIR, filename)
        if os.path.exists(filepath):
            arr = np.load(filepath)
            flipped = flip_npy_vertically(arr)
            np.save(filepath, flipped)
            print(f"  Flipped: {filename}")

    # 3. 处理 visible_boxes.json
    vb_path = os.path.join(TARGET_DIR, "visible_boxes.json")
    if os.path.exists(vb_path):
        print("\nJSON files:")
        process_visible_boxes(vb_path)

    # 4. 处理 removals 目录（排除已处理的 1）
    removals_dir = os.path.join(TARGET_DIR, "removals")
    if os.path.exists(removals_dir):
        print("\nRemovals:")
        for entry in os.listdir(removals_dir):
            if entry == "1":
                print(f"  Skipped: {entry} (already processed)")
                continue

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

            # 处理 result.json（不含bbox，无需修改）
            # result.json 只有 pos_before/pos_after，是世界坐标不需要翻转

    print("\nDone!")
