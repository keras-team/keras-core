import inspect
import textwrap

from tensorflow import nest

from keras_core import backend
from keras_core.backend.keras_tensor import any_symbolic_tensors
from keras_core.operations.node import Node
from keras_core.saving import serialization_lib
from keras_core.utils import python_utils
from keras_core.utils.naming import auto_name


class Operation:
    def __init__(self, name=None):
        if name is None:
            name = auto_name(self.__class__.__name__)
        if not isinstance(name, str):
            raise ValueError(
                "Argument `name` should be a string. "
                f"Received instead: name={name} (of type {type(name)})"
            )
        self.name = name
        self._inbound_nodes = []
        self._outbound_nodes = []

    def __call__(self, *args, **kwargs):
        if any_symbolic_tensors(args, kwargs):
            return self.symbolic_call(*args, **kwargs)
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args, **kwargs):
        # Perform shape/dtype inference.
        outputs = self.compute_output_spec(*args, **kwargs)
        # Record a new node in the operations graph.
        # The Node wires itself to inbound and outbound ops.  The
        # Node constructor updates this op's self._inbound_nodes,
        # sets _keras_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the ops that produced the inputs to this
        # call.
        Node(
            operation=self, call_args=args, call_kwargs=kwargs, outputs=outputs
        )
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def compute_output_spec(self, *args, **kwargs):
        try:
            return backend.compute_output_spec(self.call, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Could not automatically infer the output shape / dtype of "
                "this operation. "
                "Please implement the `compute_output_spec` method "
                f"on your object ({self.__class__.__name__}). "
                f"Error encountered: {e}"
            )

    def __new__(cls, *args, **kwargs):
        """We override __new__ to saving serializable constructor arguments.

        These arguments are used to auto-generate an object serialization
        config, which enables user-created subclasses to be serializable
        out of the box in most cases without forcing the user
        to manually implement `get_config()`.
        """
        # Generate a config to be returned by default by `get_config()`.
        arg_names = inspect.getfullargspec(cls.__init__).args
        kwargs.update(dict(zip(arg_names[1 : len(args) + 1], args)))
        instance = super(Operation, cls).__new__(cls)
        # For safety, we only rely on auto-configs for a small set of
        # serializable types.
        supported_types = (str, int, float, bool, type(None))
        try:
            flat_arg_values = nest.flatten(kwargs)
            auto_config = True
            for value in flat_arg_values:
                if not isinstance(value, supported_types):
                    auto_config = False
                    break
        except TypeError:
            auto_config = False
        try:
            if auto_config:
                instance._auto_config = serialization_lib.SerializableDict(
                    **kwargs
                )
            else:
                instance._auto_config = None
        except RecursionError:
            # Setting an instance attribute in __new__ has the potential
            # to trigger an infinite recursion if a subclass overrides
            # setattr in an unsafe way.
            pass
        return instance

    @python_utils.default
    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-insstantiate it.

        Returns:
            Python dictionary.
        """
        config = {
            "name": self.name,
        }

        if not python_utils.is_default(self.get_config):
            # In this case the subclass implements get_config()
            return config

        # In this case the subclass doesn't implement get_config():
        # Let's see if we can autogenerate it.
        if getattr(self, "_auto_config", None) is not None:
            xtra_args = set(config.keys())
            config.update(self._auto_config.config)
            # Remove args non explicitly supported
            argspec = inspect.getfullargspec(self.__init__)
            if argspec.varkw != "kwargs":
                for key in xtra_args - xtra_args.intersection(argspec.args[1:]):
                    config.pop(key, None)
            return config
        else:
            raise NotImplementedError(
                textwrap.dedent(
                    f"""
        Object {self.__class__.__name__} was created by passing
        non-serializable argument values in `__init__()`,
        and therefore the object must override `get_config()` in
        order to be serializable. Please implement `get_config()`.

        Example:

        class CustomLayer(keras.layers.Layer):
            def __init__(self, arg1, arg2, **kwargs):
                super().__init__(**kwargs)
                self.arg1 = arg1
                self.arg2 = arg2

            def get_config(self):
                config = super().get_config()
                config.update({
                    "arg1": self.arg1,
                    "arg2": self.arg2,
                })
                return config"""
                )
            )

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        Args:
            config: A Python dictionary, typically the
                output of get_config.

        Returns:
            A layer instance.
        """
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using "
                f"config={config}.\n\nException encountered: {e}"
            )

    def __repr__(self):
        return f"<Operation name={self.name}>"

    @property
    def input(self):
        """Retrieves the input tensor(s) of a symbolic operation.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Input tensor or list of input tensors.
        """
        return self._get_node_attribute_at_index(0, "input_tensors", "input")

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Output tensor or list of output tensors.
        """
        return self._get_node_attribute_at_index(0, "output_tensors", "output")

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the properties:
        - output
        - input

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The operation's attribute `attr` at the node of index `node_index`.
        """
        if not self._inbound_nodes:
            raise ValueError(
                f"The layer {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the operation has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values
