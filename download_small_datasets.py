# utils/download_small_datasets.py

import os
import urllib.request
from zipfile import ZipFile

DATA_DIR = "../data/raw"

# Création du répertoire destiné à contenir les datasets téléchargés.
# 'exist_ok=True' évite les erreurs si le dossier existe déjà.
os.makedirs(DATA_DIR, exist_ok=True)

# Dictionnaire contenant les noms des datasets et leurs liens de téléchargement.
# Les datasets "normal" et "tuberculose" ont été ajoutés manuellement, donc seuls ceux à télécharger sont listés ici.
datasets = {
    "COVID-19": "https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip",
}

# Parcours de tous les datasets définis dans le dictionnaire.
for name, url in datasets.items():

    # Chemin complet du fichier ZIP qui sera téléchargé.
    zip_path = os.path.join(DATA_DIR, f"{name}.zip")

    # Téléchargement du dataset uniquement si le ZIP n'existe pas encore.
    if not os.path.exists(zip_path):
        print(f"Téléchargement {name}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"{name} téléchargé.")

    # Extraction du fichier ZIP vers un dossier portant le nom du dataset.
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_DIR, name))
        print(f"{name} extrait.")
