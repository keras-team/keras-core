import collections
import re

from keras_core.api_export import keras_core_export
from keras_core.backend.common import global_state


def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)


def uniquify(name):
    object_name_uids = global_state.get_global_attribute(
        "object_name_uids",
        default=collections.defaultdict(int),
        set_to_default=True,
    )
    if name in object_name_uids:
        unique_name = f"{name}_{object_name_uids[name]}"
    else:
        unique_name = name
    object_name_uids[name] += 1
    return unique_name


def to_snake_case(name):
    """Convert a string into snake_case format.

    1. Insert underscores before capitals preceded by a character and
    followed by lowercase (e.g., "MyName" to "my_name").
    2. Insert underscores between lowercase (or digits) and following
    capitals (e.g., "nameZ" to "name_z").
    3. Replace non-alphanumeric sequences and hyphens with "_"
    (e.g., "name!!name" to "name_name").
    4. Reduce consecutive underscores to one.
    5. Convert the entire string to lowercase.

    Args:
        name (str): The input string to be converted into snake_case format.

    Returns:
        str: The transformed string in snake_case format.

    Examples:
        >>> to_snake_case("MyName")
        "my_name"

        >>> to_snake_case("nameZ")
        "name_z"

        >>> to_snake_case("name!!name--name")
        "name_name_name"

        >>> to_snake_case("name__name")
        "name_name"
    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    name = re.sub(r"[\W-]+", "_", name)
    name = re.sub(r"__+", "_", name)

    return name


@keras_core_export("keras_core.backend.get_uid")
def get_uid(prefix=""):
    """Associates a string prefix with an integer counter.

    Args:
        prefix: String prefix to index.

    Returns:
        Unique integer ID.

    Example:

    >>> get_uid('dense')
    1
    >>> get_uid('dense')
    2
    """
    object_name_uids = global_state.get_global_attribute(
        "object_name_uids",
        default=collections.defaultdict(int),
        set_to_default=True,
    )
    object_name_uids[prefix] += 1
    return object_name_uids[prefix]


def reset_uids():
    global_state.set_global_attribute(
        "object_name_uids", collections.defaultdict(int)
    )


def get_object_name(obj):
    if hasattr(obj, "name"):  # Most Keras objects.
        return obj.name
    elif hasattr(obj, "__name__"):  # Function.
        return to_snake_case(obj.__name__)
    elif hasattr(obj, "__class__"):  # Class instance.
        return to_snake_case(obj.__class__.__name__)
    return to_snake_case(str(obj))
