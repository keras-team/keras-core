import inspect

from keras_core.api_export import keras_core_export
from keras_core.initializers.constant_initializers import Constant
from keras_core.initializers.constant_initializers import Ones
from keras_core.initializers.constant_initializers import Zeros
from keras_core.initializers.initializer import Initializer
from keras_core.initializers.random_initializers import GlorotNormal
from keras_core.initializers.random_initializers import GlorotUniform
from keras_core.initializers.random_initializers import HeNormal
from keras_core.initializers.random_initializers import HeUniform
from keras_core.initializers.random_initializers import LecunNormal
from keras_core.initializers.random_initializers import LecunUniform
from keras_core.initializers.random_initializers import RandomNormal
from keras_core.initializers.random_initializers import RandomUniform
from keras_core.initializers.random_initializers import VarianceScaling
from keras_core.saving import serialization_lib
from keras_core.utils.naming import to_snake_case

ALL_OBJECTS = {
    Initializer,
    Constant,
    Ones,
    Zeros,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    RandomNormal,
    RandomUniform,
    VarianceScaling,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)


@keras_core_export("keras_core.initializers.serialize")
def serialize(initializer):
    return serialization_lib.serialize_keras_object(initializer)


@keras_core_export("keras_core.initializers.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras initializer object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_core_export("keras_core.initializers.get")
def get(identifier):
    """Retrieve a Keras initializer object via an identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> keras_core.initializers.deserialize(identifier)
    <...keras_core.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> keras_core.initializers.deserialize(cfg)
    <...keras_core.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    Args:
        identifier: String or dict that contains the initializer name or
            configurations.

    Returns:
        Initializer instance base on the input identifier.
    """

    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config)
    elif callable(identifier):
        if inspect.isclass(identifier):
            identifier = identifier()
        return identifier
    else:
        raise ValueError(
            f"Could not interpret initializer object identifier: {identifier}"
        )
