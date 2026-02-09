import os
import json
import random

DATASET_PATH = "dataset/Eye_Diseases"
SPLIT_FILE = "dataset/split.json"

train, val, test = [], [], []

for class_name in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    n = len(images)
    train += [(class_name, img) for img in images[:int(0.8*n)]]
    val   += [(class_name, img) for img in images[int(0.8*n):int(0.9*n)]]
    test  += [(class_name, img) for img in images[int(0.9*n):]]

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

with open(SPLIT_FILE, "w") as f:
    json.dump({
        "train": train,
        "val": val,
        "test": test
    }, f)

print("Dataset split saved.")

# this is for splitting data