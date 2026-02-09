import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
DATASET_PATH = "dataset/Eye_Diseases"
SPLIT_FILE = "dataset/split.json"

model = tf.keras.models.load_model("models/eye_disease_model.keras")

with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)

class_names = sorted(os.listdir(DATASET_PATH))

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

test_ds = build_dataset_from_split(split_data["test"])
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for _, y in test_ds], axis=0)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
# results
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# this is for final results