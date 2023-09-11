import unittest

from keras_core.legacy.losses import Reduction


class TestReduction(unittest.TestCase):
    def test_all(self):
        self.assertEqual(
            Reduction.all(), ("auto", "none", "sum", "sum_over_batch_size")
        )

    def test_validate_valid_key(self):
        Reduction.validate("auto")
        Reduction.validate("none")
        Reduction.validate("sum")
        Reduction.validate("sum_over_batch_size")

    def test_validate_invalid_key(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid Reduction Key: invalid_key. Expected keys are"
        ):
            Reduction.validate("invalid_key")
