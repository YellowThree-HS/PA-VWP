"""修复相机视角上下颠倒问题 - 垂直翻转所有数据"""
import cv2
import numpy as np
import json
import os

TARGET_DIR = r"C:\projs\BoxWorld-MVP\dataset\round_001_33_east\removals\1"
IMAGE_HEIGHT = 480  # 假设图像高度为480


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


def process_json_file(filepath):
    """处理JSON文件中的bbox"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False

    # 处理 visible_boxes.json
    if isinstance(data, list):
        for item in data:
            if "bbox" in item:
                item["bbox"] = fix_bbox_y(item["bbox"])
                modified = True

    # 处理 removals 下的 result.json
    if "affected_boxes" in data:
        # result.json 没有 bbox，不需要处理
        pass

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  Fixed: {os.path.basename(filepath)}")


def main():
    print(f"Processing: {TARGET_DIR}")
    print(f"Image height: {IMAGE_HEIGHT}")
    print()

    # 1. 处理 PNG 文件
    png_files = ["affected_mask.png", "removed_mask.png", "annotated.png"]
    for filename in png_files:
        filepath = os.path.join(TARGET_DIR, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                flipped = flip_image_vertically(img)
                cv2.imwrite(filepath, flipped)
                print(f"Flipped: {filename}")
            else:
                print(f"Failed to read: {filename}")

    # 2. 处理 JSON 文件中的 bbox
    json_files = ["result.json"]
    for filename in json_files:
        filepath = os.path.join(TARGET_DIR, filename)
        if os.path.exists(filepath):
            process_json_file(filepath)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
