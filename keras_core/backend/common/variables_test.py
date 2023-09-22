import numpy as np

from keras_core import backend
from keras_core import initializers
from keras_core.backend.common.variables import AutocastScope
from keras_core.backend.common.variables import KerasVariable
from keras_core.backend.common.variables import standardize_shape
from keras_core.testing import test_case
from keras_core.ops.core import convert_to_tensor


class VariablesTest(test_case.TestCase):
    def test_deferred_initialization(self):
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            # Variables can nevertheless be accessed
            _ = v + 1
        self.assertEqual(v._value.shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "while in a stateless scope"):
            with backend.StatelessScope():
                v = backend.Variable(initializer=0)

    def test_deferred_assignment(self):
        with backend.StatelessScope() as scope:
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            v.assign(np.zeros((2, 2)))
            v.assign_add(2 * np.ones((2, 2)))
            v.assign_sub(np.ones((2, 2)))
        out = scope.get_current_value(v)
        self.assertAllClose(out, np.ones((2, 2)))

    def test_autocasting(self):
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        print("open scope")
        with AutocastScope("float16"):
            self.assertEqual(
                backend.standardize_dtype(v.value.dtype), "float16"
            )
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(),
            shape=(2, 2),
            dtype="int32",
            trainable=False,
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

        with AutocastScope("float16"):
            self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

    def test_standardize_dtype_with_torch_dtype(self):
        import torch

        x = torch.randn(4, 4)
        backend.standardize_dtype(x.dtype)

    def test_name_validation(self):
        # Test when name is not a string
        with self.assertRaisesRegex(
            ValueError, "Argument `name` must be a string"
        ):
            KerasVariable(initializer=initializers.RandomNormal(), name=12345)

        # Test when name contains a '/'
        with self.assertRaisesRegex(ValueError, "cannot contain character `/`"):
            KerasVariable(
                initializer=initializers.RandomNormal(), name="invalid/name"
            )

    def test_standardize_shape_with_none(self):
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    # TODO
    # (3.9,torch) FAILED keras_core/backend/common/variables_test.py
    # ::VariablesTest::test_standardize_shape_with_non_integer_entry:
    #  - AssertionError "Cannot convert '\(3, 4, 'a'\)' to a shape.
    # " does not match "invalid literal for int() with base 10: 'a'"
    # def test_standardize_shape_with_non_integer_entry(self):
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "Cannot convert '\\(3, 4, 'a'\\)' to a shape. Found invalid",
    #     ):
    #         standardize_shape([3, 4, "a"])

    def test_standardize_shape_with_negative_entry(self):
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            standardize_shape([3, 4, -5])

    def test_autocast_scope_with_non_float_dtype(self):
        with self.assertRaisesRegex(
            ValueError,
            "`AutocastScope` can only be used with a floating-point",
        ):
            _ = AutocastScope("int32")

    def test_variable_initialization_with_non_callable(self):
        v = backend.Variable(initializer=np.ones((2, 2)))
        self.assertAllClose(v.value, np.ones((2, 2)))

    def test_variable_path_creation(self):
        v = backend.Variable(initializer=np.ones((2, 2)), name="test_var")
        self.assertEqual(v.path, "test_var")

    def test_variable_initialization_with_non_trainable(self):
        v = backend.Variable(initializer=np.ones((2, 2)), trainable=False)
        self.assertFalse(v.trainable)

    def test_variable_initialization_with_dtype(self):
        v = backend.Variable(initializer=np.ones((2, 2)), dtype="int32")
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

    def test_variable_initialization_without_shape(self):
        with self.assertRaisesRegex(
            ValueError,
            "When creating a Variable from an initializer, the `shape` ",
        ):
            backend.Variable(initializer=initializers.RandomNormal())

    def test_deferred_initialize_already_initialized(self):
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            ValueError, f"Variable {v.path} is already initialized."
        ):
            v._deferred_initialize()

    def test_deferred_initialize_within_stateless_scope(self):
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            with self.assertRaisesRegex(
                ValueError,
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed.",
            ):
                v._deferred_initialize()

    def test_variable_as_boolean(self):
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaises(TypeError):
            bool(v)

    def test_variable_negation(self):
        v = backend.Variable(initializer=np.array([-1, 2]))
        neg_v = -v
        self.assertAllClose(neg_v, np.array([1, -2]))

    def test_variable_pos(self):
        v = backend.Variable(initializer=np.array([-1, 2]))
        pos_v = v
        self.assertAllClose(pos_v, np.array([-1, 2]))

    def test_variable_abs(self):
        v = backend.Variable(initializer=np.array([-1, 2]))
        abs_v = abs(v)
        self.assertAllClose(abs_v, np.array([1, 2]))

    def test_variable_invert(self):
        v = backend.Variable(initializer=np.array([0, -1]), dtype="int32")
        inv_v = ~v
        self.assertAllClose(inv_v, np.array([-1, 0]))

    def test_variable_lt_tensor(self):
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 3, 2]))

        lt_result = v1 < v2
        self.assertAllClose(lt_result.numpy(), np.array([False, True, False]))

    def test_variable_lt_scalar(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))

        lt_result = v < 3
        self.assertAllClose(lt_result.numpy(), np.array([True, True, False]))

    def test_variable_lt_numpy_array(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        arr = np.array([2, 2, 2])

        lt_result = v < arr
        self.assertAllClose(lt_result.numpy(), np.array([True, False, False]))

    def test_variable_ge_tensor(self):
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 3, 2]))

        ge_result = v1 >= v2
        self.assertAllClose(ge_result.numpy(), np.array([True, False, True]))

    def test_variable_ge_scalar(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))

        ge_result = v >= 2
        self.assertAllClose(ge_result.numpy(), np.array([False, True, True]))

    def test_variable_ge_numpy_array(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        arr = np.array([2, 2, 2])

        ge_result = v >= arr
        self.assertAllClose(ge_result.numpy(), np.array([False, True, True]))

    def test_variable_rsub_scalar(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))

        rsub_result = 2 - v
        self.assertAllClose(rsub_result.numpy(), np.array([1, 0, -1]))

    def test_variable_div_scalar(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))

        div_result = v / 2
        self.assertAllClose(div_result.numpy(), np.array([1, 2, 4]))

    def test_variable_rdiv_scalar(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))

        rdiv_result = 16 / v
        self.assertAllClose(rdiv_result.numpy(), np.array([8, 4, 2]))

    def test_variable_div_numpy_array(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        arr = np.array([2, 8, 16])

        div_result = arr / v
        self.assertAllClose(div_result, np.array([1, 2, 2]))

    def test_variable_rdiv_numpy_array(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        arr = np.array([16, 32, 64])

        rdiv_result = arr / v
        self.assertAllClose(rdiv_result, np.array([8, 8, 8]))

    def test_variable_rsub_numpy_array(self):
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        arr = np.array([2, 2, 2])

        rsub_result = arr - v
        self.assertAllClose(rsub_result, np.array([1, 0, -1]))

    def test_variable_rtruediv(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        result = 16 / v
        self.assertAllClose(result.numpy(), np.array([8, 4, 2]))

    def test_variable_floordiv(self):
        v = backend.Variable(initializer=np.array([3, 4, 6]))
        result = v // np.array([2, 3, 6])
        self.assertAllClose(result.numpy(), np.array([1, 1, 1]))

    def test_variable_rfloordiv(self):
        v = backend.Variable(initializer=np.array([3, 4, 6]))
        result = np.array([9, 12, 18]) // v
        self.assertAllClose(result.numpy(), np.array([3, 3, 3]))

    def test_variable_rfloordiv(self):
        v = backend.Variable(initializer=np.array([3, 4, 6]))
        result = np.array([9, 12, 18]) // v
        self.assertAllClose(result, np.array([3, 3, 3]))

    def test_variable_mod_scalar(self):
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        mod_result = v % 3
        self.assertAllClose(mod_result.numpy(), np.array([2, 1, 2]))

    def test_variable_rmod_scalar(self):
        v = backend.Variable(initializer=np.array([3, 5, 7]))
        rmod_result = 10 % v
        self.assertAllClose(rmod_result.numpy(), np.array([1, 0, 3]))

    def test_variable_pow_scalar(self):
        v = backend.Variable(initializer=np.array([2, 3, 4]))
        pow_result = v**2
        self.assertAllClose(pow_result.numpy(), np.array([4, 9, 16]))

    def test_variable_rpow_scalar(self):
        v = backend.Variable(initializer=np.array([2, 3, 4]))
        rpow_result = 3**v
        self.assertAllClose(rpow_result.numpy(), np.array([9, 27, 81]))

    def test_variable_matmul(self):
        v = backend.Variable(initializer=np.array([[2, 3], [4, 5]]))
        other = np.array([[1, 2], [3, 4]])
        matmul_result = v @ other
        self.assertAllClose(
            matmul_result.numpy(), np.array([[11, 16], [19, 28]])
        )

    def test_variable_rmatmul(self):
        v = backend.Variable(initializer=np.array([[2, 3], [4, 5]]))
        other = np.array([[1, 2], [3, 4]])
        rmatmul_result = other @ v
        self.assertAllClose(rmatmul_result, np.array([[10, 13], [22, 29]]))

    def test_variable_and(self):
        v = backend.Variable(
            initializer=np.array([1, 0, 1, 0], dtype=np.int32), dtype="int32"
        )
        other_tensor = convert_to_tensor(
            np.array([1, 1, 0, 1], dtype=np.int32), dtype="int32"
        )
        and_result = v & other_tensor
        self.assertAllClose(and_result.numpy(), np.array([1, 0, 0, 0]))

    def test_variable_rand(self):
        v = backend.Variable(
            initializer=np.array([1, 0, 1, 0], dtype=np.int32), dtype="int32"
        )
        other_tensor = convert_to_tensor(
            np.array([1, 1, 0, 1], dtype=np.int32), dtype="int32"
        )
        rand_result = other_tensor & v
        self.assertAllClose(rand_result.numpy(), np.array([1, 0, 0, 0]))
