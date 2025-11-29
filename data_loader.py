# utils/data_loader_small.py
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "../data/raw"

def create_generators(batch_size=16, target_size=(224, 224)):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen
