import json
import warnings
import os

import numpy as np

from absl import logging
from keras_core.utils import io_utils
from keras_core import optimizers
from keras_core import backend
from keras_core import losses
from keras_core.saving import object_registration
from keras_core.saving.legacy import saving_utils
from keras_core.saving.legacy.saved_model import json_utils


try:
    import h5py
except ImportError:
    h5py = None


HDF5_OBJECT_HEADER_LIMIT = 64512


def save_model_to_hdf5(model, filepath, overwrite=True, include_optimizer=True):
    if h5py is None:
        raise ImportError(
            "`save_model()` using h5 format requires h5py. Could not "
            "import h5py."
        )

    # Ensures that all models saved in HDF5 format follow the old serialization
    model.use_legacy_config = True

    if not isinstance(filepath, h5py.File):
        # If file exists and should not be overwritten.
        if not overwrite and os.path.isfile(filepath):
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        # Try creating dir if not exist
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.path.makedirs(dirpath)

        f = h5py.File(filepath, mode="w")
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False
    try:
        model_metadata = saving_utils.model_metadata(model, include_optimizer)
        for k, v in model_metadata.items():
            if isinstance(v, (dict, list, tuple)):
                f.attrs[k] = json.dumps(
                    v, default=json_utils.get_json_type
                ).encode("utf8")
            else:
                f.attrs[k] = v

        model_weights_group = f.create_group("model_weights")
        save_weights_to_hdf5_group(model_weights_group, model)

        # TODO(b/128683857): Add integration tests between tf.keras and external
        # Keras, to avoid breaking TF.js users.
        if (
            include_optimizer
            and hasattr(model, "optimizer")
        ):
            save_optimizer_weights_to_hdf5_group(f, model.optimizer)

        f.flush()
    finally:
        if opened_new_file:
            f.close()

        # Remove legacy serialization attribute after H5 saving complete
        delattr(model, "use_legacy_config")


def load_model_from_hdf5(filepath, custom_objects=None, compile=True):
    """Loads a model saved via `save_model_to_hdf5`.

    Args:
        filepath: One of the following:
            - String, path to the saved model
            - `h5py.File` object from which to load the model
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    Returns:
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    Raises:
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError(
            "`load_model()` using h5 format requires h5py. Could not "
            "import h5py."
        )

    if not custom_objects:
        custom_objects = {}

    gco = object_registration.GLOBAL_CUSTOM_OBJECTS
    custom_objects = {**custom_objects, **gco}

    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode="r")
    else:
        f = filepath

    model = None
    try:
        # instantiate model
        model_config = f.attrs.get("model_config")
        if model_config is None:
            raise ValueError(
                f"No model config found in the file at {filepath}."
            )
        if hasattr(model_config, "decode"):
            model_config = model_config.decode("utf-8")
        model_config = json_utils.decode(model_config)
        model = saving_utils.model_from_config(
            model_config, custom_objects=custom_objects
        )

        # set weights
        load_weights_from_hdf5_group(f["model_weights"], model)

        if compile:
            # instantiate optimizer
            training_config = f.attrs.get("training_config")
            if hasattr(training_config, "decode"):
                training_config = training_config.decode("utf-8")
            if training_config is None:
                logging.warning(
                    "No training configuration found in the save file, so "
                    "the model was *not* compiled. Compile it manually."
                )
                return model
            training_config = json_utils.decode(training_config)

            # Compile model.
            model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config, custom_objects
                )
            )
            saving_utils.try_build_compiled_arguments(model)

            # Set optimizer weights.
            if "optimizer_weights" in f:
                try:
                    if isinstance(model.optimizer, optimizers.Optimizer):
                        model.optimizer.build(model._trainable_variables)
                    else:
                        model.optimizer._create_all_weights(
                            model._trainable_variables
                        )
                except (NotImplementedError, AttributeError):
                    logging.warning(
                        "Error when creating the weights of optimizer {}, "
                        "making it impossible to restore the saved optimizer "
                        "state. As a result, your model is starting with "
                        "a freshly initialized optimizer."
                    )

                optimizer_weight_values = (
                    load_optimizer_weights_from_hdf5_group(f)
                )
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning(
                        "Error in loading the saved optimizer "
                        "state. As a result, your model is "
                        "starting with a freshly initialized "
                        "optimizer."
                    )
    finally:
        if opened_new_file:
            f.close()
    return model


def save_weights_to_hdf5_group(f, model):
    """Saves the weights of a list of layers to a HDF5 group.

    Args:
        f: HDF5 group.
        model: Model instance.
    """
    from keras_core import __version__ as keras_version

    save_attributes_to_hdf5_group(
        f, "layer_names", [layer.name.encode("utf8") for layer in model.layers]
    )
    f.attrs["backend"] = backend.backend().encode("utf8")
    f.attrs["keras_version"] = str(keras_version).encode("utf8")

    # Sort model layers by layer name to ensure that group names are strictly
    # growing to avoid prefix issues.
    for layer in sorted(model.layers, key=lambda x: x.name):
        g = f.create_group(layer.name)
        weights = _legacy_weights(layer)
        save_subset_weights_to_hdf5_group(g, weights)
    weights = list(v for v in model._trainable_variables + model._non_trainable_variables if v in model.weights)
    g = f.create_group("top_level_model_weights")
    save_subset_weights_to_hdf5_group(g, weights)


def save_subset_weights_to_hdf5_group(f, weights):
    """Save top-level weights of a model to a HDF5 group.

    Args:
        f: HDF5 group.
        weights: List of weight variables.
    """
    weight_values = [backend.convert_to_numpy(w) for w in weights]
    weight_names = [w.name.encode("utf8") for w in weights]
    save_attributes_to_hdf5_group(f, "weight_names", weight_names)
    for name, val in zip(weight_names, weight_values):
        param_dset = f.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val


def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
    """Saves optimizer weights of a optimizer to a HDF5 group.

    Args:
        hdf5_group: HDF5 group.
        optimizer: optimizer instance.
    """
    if isinstance(optimizer, optimizers.Optimizer):
        symbolic_weights = optimizer.variables
    else:
        symbolic_weights = getattr(optimizer, "weights")
    if symbolic_weights:
        weights_group = hdf5_group.create_group("optimizer_weights")
        weight_names = [str(w.name).encode("utf8") for w in symbolic_weights]
        save_attributes_to_hdf5_group(
            weights_group, "weight_names", weight_names
        )
        weight_values = [backend.convert_to_numpy(w) for w in symbolic_weights]
        for name, val in zip(weight_names, weight_values):
            param_dset = weights_group.create_dataset(
                name, val.shape, dtype=val.dtype
            )
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val


def save_attributes_to_hdf5_group(group, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not
    able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
        data: Attributes data to store.

    Raises:
      RuntimeError: If any single attribute is too large to be saved.
    """
    # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
    # because in that case even chunking the array would not make the saving
    # possible.
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    # Expecting this to never be true.
    if bad_attributes:
        raise RuntimeError(
            "The following attributes cannot be saved to HDF5 file because "
            f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} "
            f"bytes: {bad_attributes}"
        )

    data_npy = np.asarray(data)

    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)

    # This will never loop forever thanks to the test above.
    while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs["%s%d" % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data


def load_weights_from_hdf5_group(f, model):
    """Implements topological (order-based) weight loading.

    Args:
        f: A pointer to a HDF5 group.
        model: Model instance.

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None

    filtered_layers = []
    for layer in model.layers:
        weights = _legacy_weights(layer)
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(f, "layer_names")
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError(
            "Layer count mismatch when loading weights from file. "
            f"Model expected {len(filtered_layers)} layers, found "
            f"{len(layer_names)} saved layers."
        )

    for k, name in enumerate(layer_names):
        g = f[name]
        layer = filtered_layers[k]
        symbolic_weights = _legacy_weights(layer)
        weight_values = load_subset_weights_from_hdf5_group(g)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(
                f"Weight count mismatch for layer #{k} (named {layer.name} in "
                f"the current model, {name} in the save file). "
                f"Layer expects {len(symbolic_weights)} weight(s). Received "
                f"{len(weight_values)} saved weight(s)"
            )
        for ref_v, val in zip(symbolic_weights, weight_values):
            ref_v.assign(val)

    if "top_level_model_weights" in f:
        symbolic_weights = list(
            # model.weights
            v
            for v in model._trainable_variables + model._non_trainable_variables
            if v in model.weights
        )
        weight_values = load_subset_weights_from_hdf5_group(
            f["top_level_model_weights"]
        )
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(
                "Weight count mismatch for top-level weights when loading "
                "weights from file. "
                f"Model expects {len(symbolic_weights)} top-level weight(s). "
                f"Received {len(weight_values)} saved top-level weight(s)"
            )
        for ref_v, val in zip(symbolic_weights, weight_values):
            ref_v.assign(val)


def load_weights_from_hdf5_group_by_name(f, model, skip_mismatch=False):
    """Implements name-based weight loading (instead of topological loading).

    Layers that have no matching name are skipped.

    Args:
        f: A pointer to a HDF5 group.
        model: Model instance.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file and skip_match=False.
    """
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, "layer_names")

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    for k, name in enumerate(layer_names):
        g = f[name]
        weight_values = load_subset_weights_from_hdf5_group(g)
        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn(
                        f"Skipping loading of weights for layer #{k} (named "
                        f"{layer.name}) due to mismatch in number of weights. "
                        f"Layer expects {len(symbolic_weights)} weight(s). "
                        f"Received {len(weight_values)} saved weight(s)",
                        stacklevel=2,
                    )
                    continue
                raise ValueError(
                    f"Weight count mismatch for layer #{k} "
                    f"(named {layer.name}). "
                    f"Layer expects {len(symbolic_weights)} weight(s). "
                    f"Received {len(weight_values)} saved weight(s)"
                )
            # Set values.
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(
                            f"Skipping loading weights for layer #{k} (named "
                            f"{layer.name}) due to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}",
                            stacklevel=2,
                        )
                        continue
                    raise ValueError(
                        f"Shape mismatch in layer #{k} (named {layer.name}) "
                        f"for weight {symbolic_weights[i].name}. "
                        f"Weight expects shape {expected_shape}. "
                        "Received saved weight "
                        f"with shape {received_shape}"
                    )
                else:
                    symbolic_weights[i].assign(weight_values[i])

    if "top_level_model_weights" in f:
        symbolic_weights = (
            model.trainable_weights + model.non_trainable_weights
        )
        weight_values = load_subset_weights_from_hdf5_group(
            f["top_level_model_weights"]
        )

        if len(weight_values) != len(symbolic_weights):
            if skip_mismatch:
                warnings.warn(
                    "Skipping loading top-level weights for model due to "
                    "mismatch in number of weights. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "Weight count mismatch for top-level weights of model. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)"
                )
        else:
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(
                            "Skipping loading top-level weight for model due "
                            "to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}",
                            stacklevel=2,
                        )
                    else:
                        raise ValueError(
                            "Shape mismatch in model for top-level weight "
                            f"{symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}"
                        )
                else:
                    symbolic_weights[i].assign(weight_values[i])


def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.

    Args:
        f: A pointer to a HDF5 group.

    Returns:
        List of NumPy arrays of the weight values.

    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")
    return [np.asarray(f[weight_name]) for weight_name in weight_names]


def load_optimizer_weights_from_hdf5_group(hdf5_group):
    """Load optimizer weights from a HDF5 group.

    Args:
        hdf5_group: A pointer to a HDF5 group.

    Returns:
        data: List of optimizer weight names.
    """
    weights_group = hdf5_group["optimizer_weights"]
    optimizer_weight_names = load_attributes_from_hdf5_group(
        weights_group, "weight_names"
    )
    return [
        weights_group[weight_name] for weight_name in optimizer_weight_names
    ]


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    Returns:
        data: Attributes data.
    """
    if name in group.attrs:
        data = [
            n.decode("utf8") if hasattr(n, "decode") else n
            for n in group.attrs[name]
        ]
    else:
        data = []
        chunk_id = 0
        while f"{name}{chunk_id}" in group.attrs:
            data.extend(
                [
                    n.decode("utf8") if hasattr(n, "decode") else n
                    for n in group.attrs[f"{name}{chunk_id}"]
                ]
            )
            chunk_id += 1
    return data


def _legacy_weights(layer):
    """Legacy weight order converter.

    For legacy reason, the layer.weights was in the order of
    [self.trainable_weights + self.non_trainable_weights], and this order was
    used for preserving the weights in h5 format. The new order of layer.weights
    are the same as layer.get_weights() which is more intuitive for user. To
    keep supporting the existing saved h5 file, this method should be used to
    save/load weights.

    Args:
        layer: a `Model` or `Layer` instance.

    Returns:
        A list of variables with the legacy weight order.
    """
    return layer.trainable_weights + layer.non_trainable_weights
