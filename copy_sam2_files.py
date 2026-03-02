import os
import shutil
from pathlib import Path

# 定义源和目标路径
source_dir = Path(r"G:\TJ\GroundingDINO\sam2")
target_dir = Path(r"G:\TJ\GroundingDINO")

# 需要复制的文件和文件夹
items_to_copy = [
    "sam2",  # sam2 Python 包文件夹
    "setup.py",
    "pyproject.toml",
]

copied_count = 0
skipped_count = 0

for item in items_to_copy:
    source_path = source_dir / item
    target_path = target_dir / item

    if not source_path.exists():
        print(f"⚠️  源文件不存在: {source_path}")
        skipped_count += 1
        continue

    # 如果目标已存在，先删除
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    # 复制文件或文件夹
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
        print(f"✓ 已复制文件夹: {item}")
    else:
        shutil.copy2(source_path, target_path)
        print(f"✓ 已复制文件: {item}")

    copied_count += 1

print(f"\n复制完成! 共复制 {copied_count} 项，跳过 {skipped_count} 项")
print(f"sam2 文件已从 {source_dir} 复制到 {target_dir}")
