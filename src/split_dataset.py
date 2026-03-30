import os
import shutil
import random

# ===== 配置 =====
SOURCE_DIR = "data_all"
TARGET_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 123

random.seed(SEED)

# ===== 创建目录 =====
for split in ["train", "val", "test"]:
    split_path = os.path.join(TARGET_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# ===== 遍历每个类别 =====
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    print(f"{class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 创建子目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TARGET_DIR, split, class_name), exist_ok=True)

    # ===== 拷贝文件 =====
    def copy_files(file_list, split):
        for file_name in file_list:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(TARGET_DIR, split, class_name, file_name)
            shutil.copy2(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

print("Dataset split completed!")