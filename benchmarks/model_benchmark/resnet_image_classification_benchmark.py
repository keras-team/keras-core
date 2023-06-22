import tensorflow as tf
import tensorflow_datasets as tfds

import keras_core
from keras_core.applications.efficientnet_v2 import EfficientNetV2B0

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
CHANNELS = 3

train_dataset, val_dataset = tfds.load(
    "cats_vs_dogs", split=["train[:90%]", "train[90%:]"], as_supervised=True
)

resizing = keras_core.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)


def preprocess_inputs(image, label):
    image = tf.cast(image, "float32")
    return resizing(image), label


train_dataset = (
    train_dataset.map(preprocess_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    val_dataset.map(preprocess_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

model = EfficientNetV2B0(include_top=False, weights="imagenet")
classification_layer = keras_core.layers.Dense(2, activation="softmax")
classifier = keras_core.models.Sequential(
    [
        keras_core.Input([IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS]),
        model,
        keras_core.layers.GlobalAveragePooling2D(),
        keras_core.layers.Dense(2, activation="softmax"),
    ]
)

classifier.compile(
    optimizer=keras_core.optimizers.Adam(5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
classifier.fit(
    train_dataset.take(10), epochs=1, validation_data=val_dataset.take(2)
)
