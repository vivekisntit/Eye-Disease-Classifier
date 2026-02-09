import tensorflow as tf
from preprocessing import resize_and_rescale, data_augmentation

def build_model(num_classes):
    model = tf.keras.Sequential([
        resize_and_rescale,
        data_augmentation,

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Defines the architecture of the model, including preprocessing and augmentation layers.