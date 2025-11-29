import os
import shutil
import random

# Dossier contenant les images brutes classées par catégories (COVID-19, Tuberculosis, Normal).
RAW_DIR = "data/raw"

# Dossier cible où seront générés les sous-dossiers train/val/test.
BASE_DIR = "dataset"

# Liste des classes présentes dans les données.
CLASS_NAMES = ["COVID-19", "Tuberculosis", "Normal"]

# Proportions utilisées pour la répartition des données.
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Création des sous-dossiers train, val, et test pour chaque classe.
for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Répartition des images pour chaque classe.
for cls in CLASS_NAMES:
    cls_dir = os.path.join(RAW_DIR, cls)

    # Récupération de toutes les images avec extensions valides.
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Mélange aléatoire pour garantir une distribution équilibrée.
    random.shuffle(images)

    # Calcul du nombre total d'images et du nombre alloué à chaque ensemble.
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # garantit que la somme reste correcte

    # Définition des partitions pour cette classe.
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    # Copie physique des images vers les dossiers correspondants.
    for split_name, split_images in splits.items():
        for img in split_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(BASE_DIR, split_name, cls, img)
            shutil.copy(src, dst)

    # Affichage du nombre final d'images par ensemble pour cette classe.
    print(f"{cls} => Train: {n_train}, Val: {n_val}, Test: {n_test}")
