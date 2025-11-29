# utils/data_loader_small.py

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dossier contenant les images brutes classées par dossier (un dossier par classe)
DATA_DIR = "../data/raw"

def create_generators(batch_size=16, target_size=(224, 224)):
    # Création d'un générateur d'images avec normalisation et augmentation légère
    # rescale : normalise les pixels entre 0 et 1
    # validation_split : réserve une partie du dataset pour la validation
    # horizontal_flip : applique un retournement horizontal pour augmenter les données
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True
    )

    # Générateur pour l'ensemble d'entraînement
    # subset='training' garantit que seules les images dédiées à l'entraînement sont chargées
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Générateur pour l'ensemble de validation
    # subset='validation' garantit que les données utilisées pour valider le modèle sont distinctes
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Les deux générateurs (train et validation) sont renvoyés à l’appelant
    return train_gen, val_gen
