"""
比较 old_all_dataset 和 all_dataset 的 val 目录是否完全一致
"""
import os
from pathlib import Path
from collections import Counter


def get_all_files(directory):
    """获取目录下所有文件的相对路径"""
    files = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return files
    
    for root, dirs, filenames in os.walk(directory):
        root_path = Path(root)
        for filename in filenames:
            rel_path = root_path.relative_to(dir_path)
            files.append(str(rel_path / filename))
    
    return sorted(files)


def compare_directories(dir1, dir2):
    """比较两个目录的内容是否完全一致"""
    print(f"正在比较:")
    print(f"  目录1: {dir1}")
    print(f"  目录2: {dir2}\n")
    
    # 检查目录是否存在
    path1 = Path(dir1)
    path2 = Path(dir2)
    
    if not path1.exists():
        print(f"错误: 目录1不存在: {dir1}")
        return False
    
    if not path2.exists():
        print(f"错误: 目录2不存在: {dir2}")
        return False
    
    # 获取所有文件列表
    print("正在收集文件列表...")
    files1 = get_all_files(dir1)
    files2 = get_all_files(dir2)
    
    print(f"目录1中的文件数: {len(files1)}")
    print(f"目录2中的文件数: {len(files2)}\n")
    
    # 比较文件列表
    set1 = set(files1)
    set2 = set(files2)
    
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    common = set1 & set2
    
    print("=" * 60)
    print("文件列表比较结果:")
    print("=" * 60)
    
    if only_in_1:
        print(f"\n仅在目录1中存在的文件 ({len(only_in_1)} 个):")
        for f in sorted(list(only_in_1))[:20]:  # 只显示前20个
            print(f"  - {f}")
        if len(only_in_1) > 20:
            print(f"  ... 还有 {len(only_in_1) - 20} 个文件")
    else:
        print("\n✓ 目录1中没有独有文件")
    
    if only_in_2:
        print(f"\n仅在目录2中存在的文件 ({len(only_in_2)} 个):")
        for f in sorted(list(only_in_2))[:20]:  # 只显示前20个
            print(f"  - {f}")
        if len(only_in_2) > 20:
            print(f"  ... 还有 {len(only_in_2) - 20} 个文件")
    else:
        print("\n✓ 目录2中没有独有文件")
    
    print(f"\n共同文件数: {len(common)}")
    
    # 比较文件内容
    print("\n" + "=" * 60)
    print("文件内容比较:")
    print("=" * 60)
    
    import hashlib
    
    def get_file_hash(filepath):
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            return None
    
    different_files = []
    same_files = 0
    
    print(f"\n正在比较 {len(common)} 个共同文件的内容...")
    for i, rel_file in enumerate(sorted(common)):
        if (i + 1) % 100 == 0:
            print(f"  已比较 {i + 1}/{len(common)} 个文件...")
        
        file1 = path1 / rel_file
        file2 = path2 / rel_file
        
        hash1 = get_file_hash(file1)
        hash2 = get_file_hash(file2)
        
        if hash1 is None or hash2 is None:
            different_files.append((rel_file, "无法读取文件"))
        elif hash1 != hash2:
            different_files.append((rel_file, "内容不同"))
        else:
            same_files += 1
    
    print(f"\n✓ 内容相同的文件: {same_files}")
    if different_files:
        print(f"\n✗ 内容不同的文件 ({len(different_files)} 个):")
        for f, reason in different_files[:20]:  # 只显示前20个
            print(f"  - {f} ({reason})")
        if len(different_files) > 20:
            print(f"  ... 还有 {len(different_files) - 20} 个文件不同")
    else:
        print("\n✓ 所有共同文件的内容都相同")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    
    is_identical = (len(only_in_1) == 0 and len(only_in_2) == 0 and len(different_files) == 0)
    
    if is_identical:
        print("\n✓✓✓ 两个目录完全一致！")
    else:
        print("\n✗✗✗ 两个目录不一致！")
        print(f"   - 目录1独有文件: {len(only_in_1)} 个")
        print(f"   - 目录2独有文件: {len(only_in_2)} 个")
        print(f"   - 内容不同的文件: {len(different_files)} 个")
    
    return is_identical


if __name__ == "__main__":
    base_dir = "/DATA/disk0/hs_25/pa"
    dir1 = os.path.join(base_dir, "old_all_dataset", "val")
    dir2 = os.path.join(base_dir, "all_dataset", "val")
    
    compare_directories(dir1, dir2)
