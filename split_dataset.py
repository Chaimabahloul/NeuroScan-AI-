import os
import shutil
import random

# Dossiers source
RAW_DIR = "data/raw"  # contient COVID-19, Tuberculosis, Normal
BASE_DIR = "dataset"  # dossier où train/val/test seront créés

CLASS_NAMES = ["COVID-19", "Tuberculosis", "Normal"]

# Proportions
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Créer dossiers train/val/test
for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Répartir les images
for cls in CLASS_NAMES:
    cls_dir = os.path.join(RAW_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        for img in split_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(BASE_DIR, split_name, cls, img)
            shutil.copy(src, dst)

    print(f"{cls} => Train: {n_train}, Val: {n_val}, Test: {n_test}")
