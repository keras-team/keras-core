import warnings

import numpy as np
import tensorflow as tf
import torch

from keras_core import callbacks as callbacks_module
from keras_core import optimizers as optimizers_module
from keras_core.trainers import trainer as base_trainer
from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.epoch_iterator import EpochIterator


class TorchTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def train_step(self, data):
        data = data[0]
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Compute prediction error
        if self._call_has_training_arg():
            y_pred = self(x, training=True)
        else:
            y_pred = self(x)

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients for
        # the weights from the previous train step.
        self.zero_grad()

        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
        )
        self._loss_tracker.update_state(loss)

        # Compute gradients
        if self.trainable_weights:
            # Backpropagation
            trainable_weights = [v for v in self.trainable_weights]

            # Call torch.Tensor.backward() on the loss to compute gradients for
            # the weights.
            loss.backward()

            gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                self.optimizer.apply_gradients(
                    zip(gradients, trainable_weights)
                )
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        with torch.no_grad():
            data = data[0]
            x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(
                data
            )
            if self._call_has_training_arg():
                y_pred = self(x, training=False)
            else:
                y_pred = self(x)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
            self._loss_tracker.update_state(loss)
            return self.compute_metrics(
                x, y, y_pred, sample_weight=sample_weight
            )

    def predict_step(self, data):
        with torch.no_grad():
            data = data[0]
            x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
            if self._call_has_training_arg():
                y_pred = self(x, training=False)
            else:
                y_pred = self(x)
            return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            return self.train_step(data)

        self.train_function = one_step_on_data

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            return self.test_step(data)

        self.test_function = one_step_on_data

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)

        self.predict_function = one_step_on_data

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
            # TODO: Support torch tensors for validation data.
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
        epoch_iterator = EpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
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

        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            # Switch the torch Module to training mode. Inform torch layers to
            # do training behavior in case the user did not use `self.training`
            # when implementing a custom layer with torch layers.
            self.train()
            for step, data in epoch_iterator.enumerate_epoch(return_type="np"):
                # Callbacks
                callbacks.on_train_batch_begin(step)

                logs = self.train_function(data)

                # Callbacks
                callbacks.on_train_batch_end(step, self._pythonify_logs(logs))
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs
            epoch_logs = self.get_metrics_result()

            # Switch the torch Module back to testing mode.
            self.eval()

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = EpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
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
                epoch_logs.update(val_logs)

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
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = EpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_test_function()
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        for step, data in epoch_iterator.enumerate_epoch(return_type="np"):
            callbacks.on_test_batch_begin(step)
            logs = self.test_function(data)
            callbacks.on_test_batch_end(step, self._pythonify_logs(logs))
        logs = self.get_metrics_result()
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = EpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tf.nest.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tf.__internal__.nest.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = None
        for step, data in epoch_iterator.enumerate_epoch(return_type="np"):
            callbacks.on_predict_batch_begin(step)
            batch_outputs = self.predict_function(data)
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
        callbacks.on_predict_end()
        return tf.__internal__.nest.map_structure_up_to(
            batch_outputs, np.concatenate, outputs
        )

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell the
                model to "pay more attention" to samples from an
                under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must
                be one-hot encoded, or an explicit final dimension of 1
                must be included for sparse class labels.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        self._assert_compile_called("train_on_batch")
        self.make_train_function()
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        data = (x, y, sample_weight)
        logs = self.train_function([data])
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        """Test the model on a single batch of samples.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        self._assert_compile_called("test_on_batch")
        self.make_test_function()

        data = (x, y, sample_weight)

        logs = self.test_function([data])
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        self.make_predict_function()
        batch_outputs = self.predict_function((x,))
        batch_outputs = tf.nest.map_structure(
            lambda x: np.array(x), batch_outputs
        )
        return batch_outputs
