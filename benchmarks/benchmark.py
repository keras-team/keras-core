import keras_core

import numpy as np
import tensorflow as tf
import time


class BenchmarkMetricsCallback:
    def __init__(self, end_batch, begin_batch=2):
        self.end_batch = end_batch
        self.begin_batch = begin_batch

        self.state = {}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.begin_batch:
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.end_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.end_batch - self.begin_batch) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.begin_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.end_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.end_batch - self.begin_batch) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput


class KerasCoreBenchmarkMetricsCallback(keras_core.callbacks.Callback):
    def __init__(self, end_batch, begin_batch=2):
        self._callback = BenchmarkMetricsCallback(end_batch, begin_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


class TFKerasBenchmarkMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, end_batch, begin_batch=2):
        self._callback = BenchmarkMetricsCallback(end_batch, begin_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


class LayerBenchmark:
    def __init__(self, layer_name, init_args, input_shape, num_iterations=100):
        self.layer_name = layer_name
        self.input_shape = input_shape
        _keras_core_layer_class = getattr(keras_core.layers, layer_name)
        _tf_keras_layer_class = getattr(tf.keras.layers, layer_name)

        self._keras_core_layer = _keras_core_layer_class(**init_args)
        self._tf_keras_layer = _tf_keras_layer_class(**init_args)

        self._keras_core_model = keras_core.Sequential([self._keras_core_layer])
        self._tf_keras_model = tf.keras.Sequential([self._tf_keras_layer])

        self.input_shape = input_shape
        self.num_iterations = num_iterations

    def benchmark_predict(self, num_samples, batch_size, num_iterations=None):
        data_shape = [num_samples] + list(self.input_shape)
        data = np.random.normal(size=data_shape)
        num_iterations = num_iterations or num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(num_iterations)
        tf_keras_callback = TFKerasBenchmarkMetricsCallback(num_iterations)

        self._keras_core_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[callback],
        )

        self._tf_keras_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[tf_keras_callback],
        )

        print(
            f"Keras Core throughput: {callback._callback.state['throughput']} samples/sec."
        )
        print(
            f"TF Keras throughput: {tf_keras_callback._callback.state['throughput']} samples/sec."
        )
        
    def benchmark_train(self, num_samples, batch_size, num_iterations=None):
        data_shape = [num_samples] + list(self.input_shape)
        data = np.random.normal(size=data_shape)
        label = self._keras_core_layer(data)
        
        num_iterations = num_iterations or num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(num_iterations)
        tf_keras_callback = TFKerasBenchmarkMetricsCallback(num_iterations)

        self._keras_core_model.compile(loss="mse", optimizer="sgd")
        self._keras_core_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[callback],
        )

        self._tf_keras_model.compile(loss="mse", optimizer="sgd")
        self._tf_keras_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[tf_keras_callback],
        )

        print(
            f"Keras Core throughput: {callback._callback.state['throughput']} samples/sec."
        )
        print(
            f"TF Keras throughput: {tf_keras_callback._callback.state['throughput']} samples/sec."
        )


if __name__ == "__main__":
    layer_name = "BatchNormalization"
    init_args = {}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[32, 32, 3])

    benchmark.benchmark_predict(
        num_samples=4000,
        batch_size=20,
        num_iterations=199,
    )

    benchmark.benchmark_train(
        num_samples=4000,
        batch_size=20,
        num_iterations=199,
    )
