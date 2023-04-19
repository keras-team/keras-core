import tensorflow as tf

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.stateless_scope import get_stateless_scope
from keras_core.backend.stateless_scope import in_stateless_scope
from keras_core.backend.tensorflow.basic import convert_to_tensor
from keras_core.utils.naming import auto_name


class Variable(KerasVariable):
    def __init__(self, value, dtype=None, trainable=True, name=None):
        self.name = name or auto_name(self.__class__.__name__)
        dtype = standardize_dtype(dtype)
        self.trainable = trainable
        self._value = tf.Variable(value, dtype=dtype)

    def assign(self, value):
        value = convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.value.shape:
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"Received: value.shape={value.shape}; "
                f"variable.shape={self.value.shape}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self.value.assign(value)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return value
        return self._value

    @property
    def dtype(self):
        return self.value.dtype.name

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def numpy(self):  # noqa: F811
        return self.value.numpy()

    # Overload native accessor.
    def __tf_tensor__(self, dtype=None, name=None):
        return tf.convert_to_tensor(self.value, dtype=dtype, name=name)

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __bool__(self):
        raise TypeError("A Keras Variable cannot be used as a boolean.")

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, other):
        return self.value.__eq__(convert_to_tensor(other, dtype=self.dtype))

    def __ne__(self, other):
        return self.value.__ne__(convert_to_tensor(other, dtype=self.dtype))

    def __lt__(self, other):
        return self.value.__lt__(convert_to_tensor(other, dtype=self.dtype))

    def __le__(self, other):
        return self.value.__le__(convert_to_tensor(other, dtype=self.dtype))

    def __gt__(self, other):
        return self.value.__gt__(convert_to_tensor(other, dtype=self.dtype))

    def __ge__(self, other):
        return self.value.__ge__(convert_to_tensor(other, dtype=self.dtype))

    def __add__(self, other):
        return self.value.__add__(convert_to_tensor(other, dtype=self.dtype))

    def __radd__(self, other):
        return self.value.__radd__(convert_to_tensor(other, dtype=self.dtype))

    def __sub__(self, other):
        return self.value.__sub__(convert_to_tensor(other, dtype=self.dtype))

    def __rsub__(self, other):
        return self.value.__rsub__(convert_to_tensor(other, dtype=self.dtype))

    def __mul__(self, other):
        return self.value.__mul__(convert_to_tensor(other, dtype=self.dtype))

    def __rmul__(self, other):
        return self.value.__rmul__(convert_to_tensor(other, dtype=self.dtype))

    def __div__(self, other):
        return self.value.__div__(convert_to_tensor(other, dtype=self.dtype))

    def __rdiv__(self, other):
        return self.value.__rdiv__(convert_to_tensor(other, dtype=self.dtype))

    def __truediv__(self, other):
        return self.value.__truediv__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __floordiv__(self, other):
        return self.value.__floordiv__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __divmod__(self, other):
        return self.value.__divmod__(convert_to_tensor(other, dtype=self.dtype))

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __mod__(self, other):
        return self.value.__mod__(convert_to_tensor(other, dtype=self.dtype))

    def __rmod__(self, other):
        return self.value.__rmod__(convert_to_tensor(other, dtype=self.dtype))

    def __pow__(self, other):
        return self.value.__pow__(convert_to_tensor(other, dtype=self.dtype))

    def __rpow__(self, other):
        return self.value.__rpow__(convert_to_tensor(other, dtype=self.dtype))

    def __matmul__(self, other):
        return self.value.__matmul__(convert_to_tensor(other, dtype=self.dtype))

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __and__(self, other):
        return self.value.__and__(convert_to_tensor(other, dtype=self.dtype))

    def __rand__(self, other):
        return self.value.__rand__(convert_to_tensor(other, dtype=self.dtype))

    def __or__(self, other):
        return self.value.__or__(convert_to_tensor(other, dtype=self.dtype))

    def __ror__(self, other):
        return self.value.__ror__(convert_to_tensor(other, dtype=self.dtype))

    def __xor__(self, other):
        return self.value.__xor__(convert_to_tensor(other, dtype=self.dtype))

    def __rxor__(self, other):
        return self.value.__rxor__(convert_to_tensor(other, dtype=self.dtype))

    def __lshift__(self, other):
        return self.value.__lshift__(convert_to_tensor(other, dtype=self.dtype))

    def __rlshift__(self, other):
        return self.value.__rlshift__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __rshift__(self, other):
        return self.value.__rshift__(convert_to_tensor(other, dtype=self.dtype))

    def __rrshift__(self, other):
        return self.value.__rrshift__(
            convert_to_tensor(other, dtype=self.dtype)
        )

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)
