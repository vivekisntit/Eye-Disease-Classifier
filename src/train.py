import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from model import build_model

# ------------------ CONSTANTS ------------------
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 27
DATASET_PATH = "dataset/Eye_Diseases"
SPLIT_FILE = "dataset/split.json"

# ------------------ LOAD SPLIT ------------------
with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)

class_names = sorted(os.listdir(DATASET_PATH))
num_classes = len(class_names)

print("Classes:", class_names)

# ------------------ DATASET BUILDER ------------------
def build_dataset_from_split(split):
    image_paths = []
    labels = []

    for class_name, img_name in split:
        image_paths.append(
            os.path.join(DATASET_PATH, class_name, img_name)
        )
        labels.append(class_names.index(class_name))

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img, label

    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# ------------------ BUILD DATASETS ------------------
train_ds = build_dataset_from_split(split_data["train"])
val_ds   = build_dataset_from_split(split_data["val"])
test_ds  = build_dataset_from_split(split_data["test"])

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ------------------ MODEL ------------------
model = build_model(num_classes)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=4,
#     restore_best_weights=True
# )

# ------------------ TRAIN ------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    # callbacks=[early_stopping]
)

# ------------------ SAVE MODEL ------------------
os.makedirs("models", exist_ok=True)
model.save("models/eye_disease_model.keras")

print("Model trained and saved successfully.")


# This file:
# Loads dataset

# Splits into train/val/test
# Trains model
# Saves model