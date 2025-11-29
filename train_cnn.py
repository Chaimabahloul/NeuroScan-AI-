import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --------------------------
# CONFIG
# --------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_PATH = "models/lung_model.h5"

# --------------------------
# DATASET + AUGMENTATION
# --------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_gen.class_indices)

print("\nClasses détectées :", train_gen.class_indices)

# --------------------------
# MODEL CNN
# --------------------------

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
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# --------------------------
# CALLBACKS
# --------------------------

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1
)

early_stop = EarlyStopping(
    patience=6,
    monitor="val_loss",
    restore_best_weights=True
)

# --------------------------
# TRAINING
# --------------------------

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# --------------------------
# TEST EVALUATION
# --------------------------

print("\nÉvaluation sur TEST SET :\n")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)

print(f"Accuracy : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall : {test_rec:.4f}")

print("\nModèle sauvegardé dans :", MODEL_PATH)
