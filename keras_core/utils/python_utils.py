import binascii
import codecs
import marshal
import os
import types as python_types


def default(method):
    """Decorates a method to detect overrides in subclasses."""
    method._is_default = True
    return method


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)


def func_dump(func):
    """Serializes a user-defined function.

    Args:
        func: the function to serialize.

    Returns:
        A tuple `(code, defaults, closure)`.
    """
    if os.name == "nt":
        raw_code = marshal.dumps(func.__code__).replace(b"\\", b"/")
        code = codecs.encode(raw_code, "base64").decode("ascii")
    else:
        raw_code = marshal.dumps(func.__code__)
        code = codecs.encode(raw_code, "base64").decode("ascii")
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
    """Deserializes a user defined function.

    Args:
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    Returns:
        A function object.
    """
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
        if isinstance(defaults, list):
            defaults = tuple(defaults)

    def ensure_value_to_cell(value):
        """Ensures that a value is converted to a python cell object.

        Args:
            value: Any value that needs to be casted to the cell type

        Returns:
            A value wrapped as a cell object (see function "func_load")
        """

        def dummy_fn():
            value  # just access it so it gets captured in .__closure__

        cell_value = dummy_fn.__closure__[0]
        if not isinstance(value, type(cell_value)):
            return cell_value
        return value

    if closure is not None:
        closure = tuple(ensure_value_to_cell(_) for _ in closure)
    try:
        raw_code = codecs.decode(code.encode("ascii"), "base64")
    except (UnicodeEncodeError, binascii.Error):
        raw_code = code.encode("raw_unicode_escape")
    code = marshal.loads(raw_code)
    if globs is None:
        globs = globals()
    return python_types.FunctionType(
        code, globs, name=code.co_name, argdefs=defaults, closure=closure
    )


def to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    Args:
        x: target object to be normalized.

    Returns:
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]
