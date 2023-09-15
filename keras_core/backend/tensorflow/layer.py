import tensorflow as tf

from keras_core.utils import tf_utils


class TFLayer(tf.__internal__.tracking.AutoTrackable):
    def __init__(self, *args, **kwargs):
        # Export-related attributes
        self._saved_model_inputs_spec = None
        self._saved_model_arg_spec = None

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _set_save_spec(self, inputs, args=None, kwargs=None):
        """Defines the save spec so that serialization can trace layer calls.

        The TensorSpecs of the call function `inputs`, `args`, and `kwargs` are
        saved into a tuple of `([inputs] + args, kwargs)`.

        Args:
          inputs: possibly nested inputs passed into the call function.
          args: a list of positional arguments passed into call.
          kwargs: a dictionary of keyword arguments passed into call.
        """
        if self._saved_model_inputs_spec is not None:
            return  # Already set.

        inputs_spec = tf.nest.map_structure(tf_utils.get_tensor_spec, inputs)
        args_spec = tf.nest.map_structure(tf_utils.get_tensor_spec, args or [])
        kwargs_spec = {}
        # Filter out non-tensor arguments from kwargs.
        for key, kwarg in kwargs.items():
            flat_kwarg = tf.nest.flatten(kwarg)
            flat_specs = [tf_utils.get_tensor_spec(x) for x in flat_kwarg]
            if any(s is None for s in flat_specs):
                continue
            kwargs_spec[key] = tf.nest.pack_sequence_as(kwarg, flat_specs)

        self._saved_model_inputs_spec = inputs_spec
        self._saved_model_arg_spec = (
            [inputs_spec] + list(args_spec),
            kwargs_spec,
        )

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            # SavedModel needs to ignore the execution functions.
            train_function = getattr(self, "train_function", None)
            test_function = getattr(self, "test_function", None)
            predict_function = getattr(self, "predict_function", None)
            self.train_function = None
            self.test_function = None
            self.predict_function = None

        children = super()._trackable_children(save_type, **kwargs)

        if save_type == "savedmodel":
            self.train_function = train_function
            self.test_function = test_function
            self.predict_function = predict_function

        return children

    @property
    def _default_save_signature(self):
        """For SavedModel support: returns the default serving signature."""

        from keras_core.models.functional import Functional
        from keras_core.models.model import Model
        from keras_core.models.sequential import Sequential

        if not isinstance(self, Model):
            return None

        inputs = None
        if (
            isinstance(self, Sequential)
            and getattr(self, "_functional", None) is not None
        ):
            inputs = self._functional.input
        elif isinstance(self, Functional):
            inputs = self.input

        if inputs is not None:
            input_signature = [
                tf.nest.map_structure(
                    lambda x: tf.TensorSpec(x.shape, self.compute_dtype),
                    inputs,
                )
            ]
        else:
            shapes_dict = self._build_shapes_dict
            if len(shapes_dict) == 1:
                input_shape = tuple(shapes_dict.values())[0]
                input_signature = [
                    tf.TensorSpec(input_shape, self.compute_dtype)
                ]
            else:
                input_signature = [
                    tf.nest.map_structure(
                        lambda x: tf.TensorSpec(x.shape, self.compute_dtype),
                        shapes_dict,
                    )
                ]

        @tf.function(input_signature=input_signature)
        def serving_default(inputs):
            return self(inputs)

        return serving_default
