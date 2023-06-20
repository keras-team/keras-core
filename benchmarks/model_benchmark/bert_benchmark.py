"""Benchmark BERT model on GLUE/MRPC task.

To run the script, use this command:
```
python3 glue.py --epochs 5 \
                --batch_size 16 \
                --learning_rate 0.001 \
                --mixed_precision_policy mixed_float16
```

"""

import inspect
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
import keras_core as keras

import keras_nlp
import numpy as np


flags.DEFINE_float("learning_rate", 0.005, "The initial learning rate.")
flags.DEFINE_string(
    "mixed_precision_policy",
    "mixed_float16",
    "The global precision policy to use, e.g., 'mixed_float16' or 'float32'.",
)
flags.DEFINE_integer("epochs", 2, "The number of epochs.")
flags.DEFINE_integer("batch_size", 8, "Batch Size.")


FLAGS = flags.FLAGS


class BenchmarkMetricsCallback:
    def __init__(self, start_batch=1, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        # Store the throughput of each epoch.
        self.state = {"throughput": []}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["epoch_begin_time"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            epoch_end_time = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                epoch_end_time - self.state["epoch_begin_time"]
            )
            self.state["throughput"].append(throughput)


def load_data():
    """Load data.

    Load GLUE/MRPC dataset, and convert the dictionary format to
    (features, label), where `features` is a tuple of all input sentences.
    """
    feature_names = ("sentence1", "sentence2")

    def split_features(x):
        # GLUE comes with dictonary data, we convert it to a uniform format
        # (features, label), where features is a tuple consisting of all
        # features. This format is necessary for using KerasNLP preprocessors.
        features = tuple([x[name] for name in feature_names])
        label = x["label"]
        return (features, label)

    train_ds, test_ds, validation_ds = tfds.load(
        "glue/mrpc",
        split=["train", "test", "validation"],
    )

    train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        split_features, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds, test_ds, validation_ds


def main(_):
    keras.mixed_precision.set_global_policy(FLAGS.mixed_precision_policy)

    logging.info(
        "Benchmarking configs...\n"
        "=========================\n"
        f"MODEL: {FLAGS.model}\n"
        f"PRESET: {FLAGS.preset}\n"
        f"TASK: glue/{FLAGS.task}\n"
        f"BATCH_SIZE: {FLAGS.batch_size}\n"
        f"EPOCHS: {FLAGS.epochs}\n"
        "=========================\n"
    )

    # Load datasets.
    train_ds, test_ds, validation_ds = load_data()
    train_ds = train_ds.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.batch(FLAGS.batch_size).prefetch(
        tf.data.AUTOTUNE
    )

    # Load the model.
    model = keras_nlp.models.BertClassifier.from_preset(
        "bert_small_en_uncased", num_classes=2
    )
    # Set loss and metrics.
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # Configure optimizer.
    lr = keras.optimizers.schedules.PolynomialDecay(
        FLAGS.learning_rate,
        decay_steps=train_ds.cardinality() * FLAGS.epochs,
        end_learning_rate=0.0,
    )
    optimizer = keras.optimizers.AdamW(lr, weight_decay=0.01)
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    benchmark_metrics_callback = BenchmarkMetricsCallback(
        start_batch=1,
        stop_batch=train_ds.cardinality(),
    )

    # Start training.
    logging.info("Starting Training...")

    st = time.time()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FLAGS.epochs,
        callbacks=[benchmark_metrics_callback],
    )

    wall_time = time.time() - st
    validation_accuracy = history.history["val_sparse_categorical_accuracy"][-1]
    examples_per_second = np.mean(
        np.array(benchmark_metrics_callback.state["throughput"])
    )

    logging.info("Training Finished!")
    logging.info(f"Wall Time: {wall_time:.4f} seconds.")
    logging.info(f"Validation Accuracy: {validation_accuracy:.4f}")
    logging.info(f"examples_per_second: {examples_per_second:.4f}")


if __name__ == "__main__":
    app.run(main)
