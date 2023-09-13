from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.testing import test_case
from keras_core.utils import dtype_utils


class DtypeSizeTests(test_case.TestCase):
    def test_bfloat16_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("bfloat16"), 16)

    def test_float16_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("float16"), 16)

    def test_float32_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("float32"), 32)

    def test_int32_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("int32"), 32)

    def test_float64_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("float64"), 64)

    def test_int64_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("int64"), 64)

    def test_uint8_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("uint8"), 8)

    def test_bool_dtype_size(self):
        self.assertEqual(dtype_utils.dtype_size("bool"), 1)

    def test_invalid_dtype_size(self):
        with self.assertRaises(ValueError):
            dtype_utils.dtype_size("unknown_dtype")


class IsFloatTests(test_case.TestCase):
    def test_is_float_float16(self):
        self.assertTrue(dtype_utils.is_float("float16"))

    def test_is_float_float32(self):
        self.assertTrue(dtype_utils.is_float("float32"))

    def test_is_float_float64(self):
        self.assertTrue(dtype_utils.is_float("float64"))

    def test_is_float_int32(self):
        self.assertFalse(dtype_utils.is_float("int32"))

    def test_is_float_bool(self):
        self.assertFalse(dtype_utils.is_float("bool"))

    def test_is_float_uint8(self):
        self.assertFalse(dtype_utils.is_float("uint8"))


class CastToCommonDtype(test_case.TestCase):
    def test_cast_to_common_dtype_float32_float64(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float32")
        tensor2 = KerasTensor([4, 5, 6], dtype="float64")
        casted_tensors = dtype_utils.cast_to_common_dtype([tensor1, tensor2])
        for tensor in casted_tensors:
            assert tensor.dtype == "float64"

    def test_cast_to_common_dtype_float16_float32_float64(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float16")
        tensor2 = KerasTensor([4, 5, 6], dtype="float32")
        tensor3 = KerasTensor([7, 8, 9], dtype="float64")
        casted_tensors = dtype_utils.cast_to_common_dtype(
            [tensor1, tensor2, tensor3]
        )
        for tensor in casted_tensors:
            assert tensor.dtype == "float64"

    def test_cast_to_common_dtype_float16_int16_float32(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float16")
        tensor2 = KerasTensor([4, 5, 6], dtype="int16")
        tensor3 = KerasTensor([7, 8, 9], dtype="float32")
        casted_tensors = dtype_utils.cast_to_common_dtype(
            [tensor1, tensor2, tensor3]
        )
        for tensor in casted_tensors:
            assert tensor.dtype == "float32"

    def test_cast_to_common_dtype_all_float32(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float32")
        tensor2 = KerasTensor([4, 5, 6], dtype="float32")
        tensor3 = KerasTensor([7, 8, 9], dtype="float32")
        casted_tensors = dtype_utils.cast_to_common_dtype(
            [tensor1, tensor2, tensor3]
        )
        for tensor in casted_tensors:
            assert tensor.dtype == "float32"

    def test_cast_to_common_dtype_float16_bfloat16(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float16")
        tensor2 = KerasTensor([4, 5, 6], dtype="bfloat16")
        casted_tensors = dtype_utils.cast_to_common_dtype([tensor1, tensor2])
        for tensor in casted_tensors:
            assert tensor.dtype == "float16"

    def test_cast_to_common_dtype_float16_uint8(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float16")
        tensor2 = KerasTensor([4, 5, 6], dtype="uint8")
        casted_tensors = dtype_utils.cast_to_common_dtype([tensor1, tensor2])
        for tensor in casted_tensors:
            assert tensor.dtype == "float16"

    def test_cast_to_common_dtype_mixed_types(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="float32")
        tensor2 = KerasTensor([4, 5, 6], dtype="int32")
        tensor3 = KerasTensor([7, 8, 9], dtype="bool")
        casted_tensors = dtype_utils.cast_to_common_dtype(
            [tensor1, tensor2, tensor3]
        )
        for tensor in casted_tensors:
            assert tensor.dtype == "float32"

    def test_cast_to_common_dtype_no_float(self):
        tensor1 = KerasTensor([1, 2, 3], dtype="int32")
        tensor2 = KerasTensor([4, 5, 6], dtype="uint8")
        casted_tensors = dtype_utils.cast_to_common_dtype([tensor1, tensor2])
        for tensor in casted_tensors:
            assert tensor.dtype == tensor.dtype
