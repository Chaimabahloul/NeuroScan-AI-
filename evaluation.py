import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report


# ================================================================
#  CONFIG
# ================================================================
MODEL_PATH = "models/lung_model.h5"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["COVID-19", "Tuberculosis", "Normal"]


# ================================================================
#  1. Charger le modèle
# ================================================================
print("Chargement du modèle...")
model = load_model(MODEL_PATH)
print("Modèle chargé avec succès !")


# ================================================================
#  2. Charger les images de test
# ================================================================
print("Préparation du dataset de test...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # Important pour garder l’ordre
)

print("Dataset de test prêt !")


# ================================================================
#  3. Prédictions
# ================================================================
print(" Prédiction en cours...")
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_generator.classes
print(" Prédictions terminées !")


# ================================================================
# 4. Matrice de confusion
# ================================================================
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# ================================================================
#  5. Rapport de classification
# ================================================================
print("\n Classification Report :\n")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=CLASS_NAMES
))


# ================================================================
# (OPTIONNEL) 6. Exemple de sauvegarde
# ================================================================
plt.savefig("confusion_matrix.png")
print("\n Image sauvegardée : confusion_matrix.png")
