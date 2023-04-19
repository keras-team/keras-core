import contextlib
import warnings

import tensorflow as tf
from tensorflow.python.eager import context as tf_context

from keras_core import callbacks as callbacks_module
from keras_core import optimizers as optimizers_module
from keras_core.trainers import trainer as base_trainer
from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.epoch_iterator import EpochIterator


class Trainer(base_trainer.Trainer):
    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg():
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        # Compute gradients
        # TODO: move value conversion to the optimizer
        trainable_weights = [v.value for v in self.trainable_weights]
        gradients = tape.gradient(loss, trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        raise NotImplementedError

    def predict_step(self, data):
        raise NotImplementedError

    def make_train_function(self, force=False):
        # TODO: support tf.distribute and steps_per_execution.
        if self.train_function is not None and not force:
            return self.train_function

        def one_step_on_data(data):
            """Runs a single training step."""
            return self.train_step(data)

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(
                one_step_on_data, jit_compile=True, reduce_retracing=True
            )

        def one_step_on_iterator(iterator):
            data = next(iterator)
            return one_step_on_data(data)

        if not self.run_eagerly:
            train_function = tf.function(
                one_step_on_iterator, reduce_retracing=True
            )
        else:
            train_function = one_step_on_iterator
        self.train_function = train_function

    def make_test_function(self, force=False):
        raise NotImplementedError

    def make_predict_function(self, force=False):
        raise NotImplementedError

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )
        # TODO: respect compiled trainable state
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter_utils.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator.enumerate_epoch():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(step, logs)
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs
            epoch_logs = self._process_logs(self.get_metrics_result())

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = EpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        epochs=1,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(self._process_logs(val_logs))

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        raise NotImplementedError

    def _process_logs(self, logs):
        result = {}
        for key, value in logs.items():
            try:
                value = float(value)
            except:
                pass
            result[key] = value
        return result


class TFEpochIterator(EpochIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steps_seen = 0

    def enumerate_epoch(self):
        if self.steps_per_epoch:
            if not self._current_iterator:
                self._current_iterator = iter(
                    self.data_adapter.get_tf_dataset()
                )
            for step in range(self.steps_per_epoch):
                yield step, self._current_iterator
        else:
            iterator = iter(self.data_adapter.get_tf_dataset())
            if self.num_batches:
                for step in range(self.num_batches):
                    yield step, iterator
            else:
                step = -1
                while True:
                    step += 1
                    self._steps_seen = step + 1
                    yield step, iterator
        self.data_adapter.on_epoch_end()

    def tf_sync(self):
        tf_context.async_wait()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
            self.tf_sync()
        except (StopIteration, tf.errors.OutOfRangeError):
            if self._num_batches is None:
                self._num_batches = self._steps_seen
            warnings.warn(
                "Your input ran out of data; interrupting training. "
                "Make sure that your dataset or generator can generate "
                "at least `steps_per_epoch * epochs` batches. "
                "You may need to use the `.repeat()` "
                "function when building your dataset.",
                stacklevel=2,
            )
            self._current_iterator = None
            self.data_adapter.on_epoch_end()
