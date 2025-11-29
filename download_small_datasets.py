# utils/download_small_datasets.py
import os
import urllib.request
from zipfile import ZipFile

DATA_DIR = "../data/raw"

# Créer dossier si n'existe pas
os.makedirs(DATA_DIR, exist_ok=True)

datasets = {
    "COVID-19": "https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip",
    #normal et tuberculose j'ai installer manuelement
}


for name, url in datasets.items():
    zip_path = os.path.join(DATA_DIR, f"{name}.zip")
    if not os.path.exists(zip_path):
        print(f"Téléchargement {name}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"{name} téléchargé.")

    # Dézipper
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_DIR, name))
        print(f" {name} extrait.")
