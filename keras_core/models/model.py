from keras_core.api_export import keras_core_export
from keras_core.backend import Trainer
from keras_core.layers.layer import Layer


@keras_core_export(["keras_core.Model", "keras_core.models.Model"])
class Model(Layer, Trainer):
    """

    Combination of a Layer and Trainer. Adds:

    - layer surfacing
    - saving
    - export
    - summary

    Limitations:

    - call must have a single inputs argument
    - no masking support
    """

    def __new__(cls, *args, **kwargs):
        # Signature detection
        if functional_init_arguments(args, kwargs) and cls == Model:
            # Functional model
            from keras_core.models import functional

            return functional.Functional(*args, **kwargs, skip_init=True)
        else:
            return super(Model, cls).__new__(cls, *args, **kwargs)

    def call(self, inputs, training=False):
        raise NotImplementedError

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of horizontal graph traversal (bottom-up).

        Args:
            name: String, name of layer.
            index: Integer, index of layer.

        Returns:
            A layer instance.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a layer name or a layer index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError(
                    f"Was asked to retrieve layer at index {index}"
                    f" but model only has {len(self.layers)}"
                    " layers."
                )
            else:
                return self.layers[index]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(
                f"No such layer: {name}. Existing layers are: "
                f"{list(layer.name for layer in self.layers)}."
            )
        raise ValueError(
            "Provide either a layer name or layer index at `get_layer`."
        )

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        """Prints a string summary of the network.

        Args:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided, becomes
                `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
            print_fn: Print function to use. By default, prints to `stdout`.
                If `stdout` doesn't work in your environment, change to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested: Whether to expand the nested models.
                Defaults to `False`.
            show_trainable: Whether to show if a layer is trainable.
                Defaults to `False`.
            layer_range: a list or tuple of 2 strings,
                which is the starting layer name and ending layer name
                (both inclusive) indicating the range of layers to be printed
                in summary. It also accepts regex patterns instead of exact
                name. In such case, start predicate will be the first element
                it matches to `layer_range[0]` and the end predicate will be
                the last element it matches to `layer_range[1]`.
                By default `None` which considers all layers of model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        if not self.built:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling `build()` or by calling "
                "the model on a batch of data."
            )
        layer_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )

    def save(self, filepath):
        raise NotImplementedError

    def export(self, filepath):
        raise NotImplementedError


def functional_init_arguments(args, kwargs):
    return (
        len(args) == 2
        or len(args) == 1
        and "outputs" in kwargs
        or "inputs" in kwargs
        and "outputs" in kwargs
    )
