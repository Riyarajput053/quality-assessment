import os
import shutil
import random

dataset_dir = "dataset"
train_dir = "train"
test_dir = "test"
val_dir = "val"

split_ratio_train = 0.7  # 70% Train
split_ratio_val = 0.15   # 15% Validation
split_ratio_test = 0.15  # 15% Test

# Function to split data into train, validation, and test sets
def split_data(class_name):
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * split_ratio_train)
    val_split = int(len(images) * (split_ratio_train + split_ratio_val))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    for folder in [train_dir, test_dir, val_dir]:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

# Apply split
split_data("fine_grain")
split_data("damaged_grain")

print("✅ Dataset split completed! (No augmentation applied)")
