import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ----------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE
# Définition des paramètres de base : taille des images, batch size,
# nombre d'époques et chemins des répertoires nécessaires au projet.
# ----------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_PATH = "models/lung_model.h5"

# ----------------------------------------------------------------------
# DATASET ET AUGMENTATION
# Préparation des générateurs pour le training, validation et test.
# L'augmentation d'images permet d'améliorer la robustesse du modèle.
# ----------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Validation et test n’utilisent pas d’augmentation pour garantir l’objectivité.
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des données depuis les dossiers organisés train/val/test.
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Le set de test est chargé sans mélange afin de permettre une évaluation correcte.
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Nombre automatique de classes détectées à partir des sous-dossiers.
num_classes = len(train_gen.class_indices)

print("\nClasses détectées :", train_gen.class_indices)

# ----------------------------------------------------------------------
# CONSTRUCTION DU MODÈLE CNN
# Architecture simple et efficace incluant BatchNormalization pour
# stabiliser l'apprentissage et Dropout pour réduire l'overfitting.
# ----------------------------------------------------------------------

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),  # permet de limiter l’overfitting
    Dense(num_classes, activation="softmax")  # sortie multiclasses
])

# Compilation du modèle avec Adam et ajout de précision/rappel comme métriques.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ----------------------------------------------------------------------
# CALLBACKS
# ModelCheckpoint : sauvegarde automatique du meilleur modèle.
# EarlyStopping : arrête l’entraînement si la validation stagne.
# ----------------------------------------------------------------------

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1
)

early_stop = EarlyStopping(
    patience=6,             # nombre d'époques sans amélioration avant arrêt
    monitor="val_loss",     # critère utilisé pour décider l’arrêt
    restore_best_weights=True
)

# ----------------------------------------------------------------------
# ENTRAINEMENT DU MODÈLE
# Les callbacks permettent un entraînement plus efficace et contrôlé.
# ----------------------------------------------------------------------

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ----------------------------------------------------------------------
# ÉVALUATION SUR LE TEST SET
# Le test final fournit les métriques globales de performance.
# ----------------------------------------------------------------------

print("\nÉvaluation sur TEST SET :\n")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)

print(f"Accuracy : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall : {test_rec:.4f}")

print("\nModèle sauvegardé dans :", MODEL_PATH)
