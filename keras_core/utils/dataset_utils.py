import multiprocessing
import os

from keras_core.api_export import keras_core_export
from keras_core.utils.module_utils import tensorflow as tf
from keras_core.utils import io_utils
from keras_core.utils import file_utils
from keras_core.backend.config import backend


@keras_core_export("keras_core.utils.split_dataset")
def split_dataset(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
):
    """Splits a dataset into a left half and a right half (e.g. train / test).

    Args:
        dataset: A `tf.data.Dataset` object, or a list/tuple of arrays with the
            same length.
        left_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the left dataset.
            If integer, it signifies the number of samples to pack
            in the left dataset. If `None`, it defaults to the complement
            to `right_size`.
        right_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the right dataset.
            If integer, it signifies the number of samples to pack
            in the right dataset. If `None`, it defaults to the complement
            to `left_size`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.

    Returns:
        A tuple of two `tf.data.Dataset` objects: the left and right splits.

    Example:

    >>> data = np.random.random(size=(1000, 4))
    >>> left_ds, right_ds = split_dataset(data, left_size=0.8)
    >>> int(left_ds.cardinality())
    800
    >>> int(right_ds.cardinality())
    200
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.split_dataset(
        dataset,
        left_size=left_size,
        right_size=right_size,
        shuffle=shuffle,
        seed=seed,
    )


@keras_core_export(
    [
        "keras_core.utils.timeseries_dataset_from_array",
        "keras_core.preprocessing.timeseries_dataset_from_array",
    ]
)
def timeseries_dataset_from_array(
    data,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
):
    """Creates a dataset of sliding windows over a timeseries provided as array.

    This function takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    length of the sequences/windows, spacing between two sequence/windows, etc.,
    to produce batches of timeseries inputs and targets.

    Args:
        data: Numpy array or eager tensor
            containing consecutive data points (timesteps).
            Axis 0 is expected to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            `targets[i]` should be the target
            corresponding to the window that starts at index `i`
            (see example 2 below).
            Pass `None` if you don't have target data (in this case the dataset
            will only yield the input data).
        sequence_length: Length of the output sequences
            (in number of timesteps).
        sequence_stride: Period between successive output sequences.
            For stride `s`, output samples would
            start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i], data[i + r], ... data[i + sequence_length]`
            are used for creating a sample sequence.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one). If `None`, the data will not be batched
            (the dataset will yield individual samples).
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        seed: Optional int; random seed for shuffling.
        start_index: Optional int; data points earlier (exclusive)
            than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Optional int; data points later (exclusive) than `end_index`
            will not be used in the output sequences.
            This is useful to reserve part of the data for test or validation.

    Returns:

    A `tf.data.Dataset` instance. If `targets` was passed, the dataset yields
    tuple `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields
    only `batch_of_sequences`.

    Example 1:

    Consider indices `[0, 1, ... 98]`.
    With `sequence_length=10,  sampling_rate=2, sequence_stride=3`,
    `shuffle=False`, the dataset will yield batches of sequences
    composed of the following indices:

    ```
    First sequence:  [0  2  4  6  8 10 12 14 16 18]
    Second sequence: [3  5  7  9 11 13 15 17 19 21]
    Third sequence:  [6  8 10 12 14 16 18 20 22 24]
    ...
    Last sequence:   [78 80 82 84 86 88 90 92 94 96]
    ```

    In this case the last 2 data points are discarded since no full sequence
    can be generated to include them (the next sequence would have started
    at index 81, and thus its last step would have gone over 98).

    Example 2: Temporal regression.

    Consider an array `data` of scalar values, of shape `(steps,)`.
    To generate a dataset that uses the past 10
    timesteps to predict the next timestep, you would use:

    ```python
    input_data = data[:-10]
    targets = data[10:]
    dataset = timeseries_dataset_from_array(
        input_data, targets, sequence_length=10)
    for batch in dataset:
      inputs, targets = batch
      assert np.array_equal(inputs[0], data[:10])  # First sequence: steps [0-9]
      # Corresponding target: step 10
      assert np.array_equal(targets[0], data[10])
      break
    ```

    Example 3: Temporal regression for many-to-many architectures.

    Consider two arrays of scalar values `X` and `Y`,
    both of shape `(100,)`. The resulting dataset should consist samples with
    20 timestamps each. The samples should not overlap.
    To generate a dataset that uses the current timestamp
    to predict the corresponding target timestep, you would use:

    ```python
    X = np.arange(100)
    Y = X*2

    sample_length = 20
    input_dataset = timeseries_dataset_from_array(
        X, None, sequence_length=sample_length, sequence_stride=sample_length)
    target_dataset = timeseries_dataset_from_array(
        Y, None, sequence_length=sample_length, sequence_stride=sample_length)

    for batch in zip(input_dataset, target_dataset):
        inputs, targets = batch
        assert np.array_equal(inputs[0], X[:sample_length])

        # second sample equals output timestamps 20-40
        assert np.array_equal(targets[1], Y[sample_length:2*sample_length])
        break
    ```
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=sequence_stride,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        start_index=start_index,
        end_index=end_index,
    )


@keras_core_export(
    [
        "keras_core.utils.text_dataset_from_directory",
        "keras_core.preprocessing.text_dataset_from_directory",
    ]
)
def text_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from text files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    ```

    Then calling `text_dataset_from_directory(main_directory,
    labels='inferred')` will return a `tf.data.Dataset` that yields batches of
    texts from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.txt` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing text files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            text files found in the directory. Labels should be sorted according
            to the alphanumeric order of the text file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `"labels"` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        max_length: Maximum size of a text string. Texts longer than this will
            be truncated to `max_length`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"` or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
        `(batch_size,)`, containing the contents of a batch of text files.
    - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        has shape `(batch_size,)` and `labels` follows the format described
        below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.text_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
    )


@keras_core_export("keras_core.utils.audio_dataset_from_directory")
def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from audio files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav
    ```

    Then calling `audio_dataset_from_directory(main_directory,
    labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of audio files from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.wav` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain subdirectories,
            each containing audio files for a class. Otherwise, the directory
            structure is ignored.
        labels: Either "inferred" (labels are generated from the directory
            structure), `None` (no labels), or a list/tuple of integer labels
            of the same size as the number of audio files found in
            the directory. Labels should be sorted according to the
            alphanumeric order of the audio file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers (e.g. for
              `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are encoded as a categorical
              vector (e.g. for `categorical_crossentropy` loss)
            - `"binary"` means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0
              or 1 (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if "labels" is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Default: 32. If `None`,
            the data will not be batched
            (the dataset will yield individual samples).
        sampling_rate: Audio sampling rate (in samples per second).
        output_sequence_length: Maximum length of an audio sequence. Audio files
            longer than this will be truncated to `output_sequence_length`.
            If set to `None`, then all sequences in the same batch will
            be padded to the
            length of the longest sequence in the batch.
        ragged: Whether to return a Ragged dataset (where each sequence has its
            own length). Defaults to `False`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1, fraction of data to
            reserve for validation.
        subset: Subset of the data to return. One of `"training"`,
            `"validation"` or `"both"`. Only used if `validation_split` is set.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
      `(batch_size,)`, containing the contents of a batch of audio files.
    - Otherwise, it yields a tuple `(audio, labels)`, where `audio`
      has shape `(batch_size, sequence_length, num_channels)` and `labels`
      follows the format described
      below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.audio_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
    )

def get_training_or_validation_split(samples, labels, validation_split, subset):
    """Potentially restict samples & labels to a training or validation split.

    Args:
      samples: List of elements.
      labels: List of corresponding labels.
      validation_split: Float, fraction of data to reserve for validation.
      subset: Subset of the data to return.
        Either "training", "validation", or None. If None, we return all of the
        data.

    Returns:
      tuple (samples, labels), potentially restricted to the specified subset.
    """
    if not validation_split:
        return samples, labels

    num_val_samples = int(validation_split * len(samples))
    if subset == "training":
        print(f"Using {len(samples) - num_val_samples} files for training.")
        samples = samples[:-num_val_samples]
        labels = labels[:-num_val_samples]
    elif subset == "validation":
        print(f"Using {num_val_samples} files for validation.")
        samples = samples[-num_val_samples:]
        labels = labels[-num_val_samples:]
    else:
        raise ValueError(
            '`subset` must be either "training" '
            f'or "validation", received: {subset}'
        )
    return samples, labels

def check_validation_split_arg(validation_split, subset, shuffle, seed):
    """Raise errors in case of invalid argument values.

    Args:
      validation_split: float between 0 and 1, fraction of data to reserve for
        validation.
      subset: One of "training", "validation" or "both". Only used if
        `validation_split` is set.
      shuffle: Whether to shuffle the data. Either True or False.
      seed: random seed for shuffling and transformations.
    """
    if validation_split and not 0 < validation_split < 1:
        raise ValueError(
            "`validation_split` must be between 0 and 1, "
            f"received: {validation_split}"
        )
    if (validation_split or subset) and not (validation_split and subset):
        raise ValueError(
            "If `subset` is set, `validation_split` must be set, and inversely."
        )
    if subset not in ("training", "validation", "both", None):
        raise ValueError(
            '`subset` must be either "training", '
            f'"validation" or "both", received: {subset}'
        )
    if validation_split and shuffle and seed is None:
        raise ValueError(
            "If using `validation_split` and shuffling the data, you must "
            "provide a `seed` argument, to make sure that there is no "
            "overlap between the training and validation subset."
        )

def iter_valid_files(directory, follow_links, formats):
    if not follow_links:
        walk = file_utils.walk(directory)
    else:
        walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname

def labels_to_dataset(labels, label_mode, num_classes):
    """Create a tf.data.Dataset from the list/tuple of labels.

    Args:
      labels: list/tuple of labels to be converted into a tf.data.Dataset.
      label_mode: String describing the encoding of `labels`. Options are:
      - 'binary' indicates that the labels (there can be only 2) are encoded as
        `float32` scalars with values 0 or 1 (e.g. for `binary_crossentropy`).
      - 'categorical' means that the labels are mapped into a categorical
        vector.  (e.g. for `categorical_crossentropy` loss).
      num_classes: number of classes of labels.

    Returns:
      A `Dataset` instance.
    """

    if backend() == 'tensorflow':
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        if label_mode == "binary":
            label_ds = label_ds.map(
                lambda x: tf.expand_dims(tf.cast(x, "float32"), axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif label_mode == "categorical":
            label_ds = label_ds.map(
                lambda x: tf.one_hot(x, num_classes),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    elif backend() == 'torch':
        from torch.utils.data import TensorDataset
        import torch
        label_ds = TensorDataset(labels)
        if label_mode == "binary":
            label_ds = label_ds.map(
                lambda x: torch.unsqueeze(x.type(torch.IntTensor), axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif label_mode == "categorical":
            label_ds = label_ds.map(
                lambda x: torch.nn.functional.one_hot(x, num_classes),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif backend() == 'jax':
        NotImplementedError('Method for jax not yet Implemented.')

    return label_ds
    

def index_subdirectory(directory, class_indices, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.

    Args:
      directory: string, target directory.
      class_indices: dict mapping class names to their index.
      follow_links: boolean, whether to recursively follow subdirectories
        (if False, we only list top-level images in `directory`).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").

    Returns:
      tuple `(filenames, labels)`. `filenames` is a list of relative file
        paths, and `labels` is a list of integer labels corresponding to these
        files.
    """
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = file_utils.join(root, fname)
        relative_path = file_utils.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)
    return filenames, labels
    
    
def index_directory(
    directory,
    labels,
    formats,
    class_names=None,
    shuffle=True,
    seed=None,
    follow_links=False,
):
    """Make list of all files in `directory`, with their labels.

    Args:
      directory: Directory where the data is located.
          If `labels` is "inferred", it should contain
          subdirectories, each containing files for a class.
          Otherwise, the directory structure is ignored.
      labels: Either "inferred"
          (labels are generated from the directory structure),
          None (no labels),
          or a list/tuple of integer labels of the same size as the number of
          valid files found in the directory. Labels should be sorted according
          to the alphanumeric order of the image file paths
          (obtained via `os.walk(directory)` in Python).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
      class_names: Only valid if "labels" is "inferred". This is the explicit
          list of class names (must match names of subdirectories). Used
          to control the order of the classes
          (otherwise alphanumerical order is used).
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling.
      follow_links: Whether to visits subdirectories pointed to by symlinks.

    Returns:
      tuple (file_paths, labels, class_names).
        file_paths: list of file paths (strings).
        labels: list of matching integer labels (same length as file_paths)
        class_names: names of the classes corresponding to these labels, in
          order.
    """
    import numpy as np

    if labels != "inferred":
        # in the explicit/no-label cases, index from the parent directory down.
        subdirs = [""]
        class_names = subdirs
    else:
        subdirs = []
        for subdir in sorted(file_utils.listdir(directory)):
            if file_utils.isdir(file_utils.join(directory, subdir)):
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                subdirs.append(subdir)
        if not class_names:
            class_names = subdirs
        else:
            if set(class_names) != set(subdirs):
                raise ValueError(
                    "The `class_names` passed did not match the "
                    "names of the subdirectories of the target directory. "
                    f"Expected: {subdirs}, but received: {class_names}"
                )
    class_indices = dict(zip(class_names, range(len(class_names))))

    # Build an index of the files
    # in the different class subfolders.
    pool = multiprocessing.pool.ThreadPool()
    results = []
    filenames = []

    for dirpath in (file_utils.join(directory, subdir) for subdir in subdirs):
        results.append(
            pool.apply_async(
                index_subdirectory,
                (dirpath, class_indices, follow_links, formats),
            )
        )
    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res.get()
        labels_list.append(partial_labels)
        filenames += partial_filenames
    if labels not in ("inferred", None):
        if len(labels) != len(filenames):
            raise ValueError(
                "Expected the lengths of `labels` to match the number "
                "of files in the target directory. len(labels) is "
                f"{len(labels)} while we found {len(filenames)} files "
                f"in directory {directory}."
            )
        class_names = sorted(set(labels))
    else:
        i = 0
        labels = np.zeros((len(filenames),), dtype="int32")
        for partial_labels in labels_list:
            labels[i : i + len(partial_labels)] = partial_labels
            i += len(partial_labels)

    if labels is None:
        io_utils.print_msg(f"Found {len(filenames)} files.")
    else:
        io_utils.print_msg(
            f"Found {len(filenames)} files belonging "
            f"to {len(class_names)} classes."
        )
    pool.close()
    pool.join()
    file_paths = [file_utils.join(directory, fname) for fname in filenames]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)
    return file_paths, labels, class_names
