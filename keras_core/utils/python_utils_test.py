from keras_core import testing
from keras_core.utils import python_utils


class PythonUtilsTest(testing.TestCase):
    def test_func_dump_and_load(self):
        def my_function(x, y=1, **kwargs):
            return x + y

        serialized = python_utils.func_dump(my_function)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(2, y=3), 5)

    def test_removesuffix(self):
        x = "model.keras"
        self.assertEqual(python_utils.removesuffix(x, ".keras"), "model")
        self.assertEqual(python_utils.removesuffix(x, "model"), x)

    def test_removeprefix(self):
        x = "model.keras"
        self.assertEqual(python_utils.removeprefix(x, "model"), ".keras")
        self.assertEqual(python_utils.removeprefix(x, ".keras"), x)

    def test_func_load_defaults_as_tuple(self):
        # Using tuple as a default argument
        def dummy_function(x=(1, 2, 3)):
            pass

        serialized = python_utils.func_dump(dummy_function)
        deserialized = python_utils.func_load(serialized)
        # Ensure that the defaults are still a tuple
        self.assertIsInstance(deserialized.__defaults__[0], tuple)
        # Ensure that the tuple default remains unchanged
        self.assertEqual(deserialized.__defaults__[0], (1, 2, 3))

    def test_remove_long_seq_standard_case(self):
        sequences = [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]
        labels = [1, 2, 3, 4]
        new_sequences, new_labels = remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [[1], [2, 2], [3, 3, 3]])
        self.assertEqual(new_labels, [1, 2, 3])

    def test_remove_long_seq_all_below_maxlen(self):
        sequences = [[1], [2, 2], [3, 3, 3]]
        labels = [1, 2, 3]
        new_sequences, new_labels = remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [[1], [2, 2], [3, 3, 3]])
        self.assertEqual(new_labels, [1, 2, 3])

    def test_remove_long_seq_all_equal_maxlen(self):
        sequences = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        labels = [1, 2, 3]
        new_sequences, new_labels = remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        self.assertEqual(new_labels, [1, 2, 3])

    def test_remove_long_seq_all_above_maxlen(self):
        sequences = [[4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]
        labels = [4, 5, 6]
        new_sequences, new_labels = remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [])
        self.assertEqual(new_labels, [])

    def test_remove_long_seq_mixed_case(self):
        sequences = [[1], [2, 2], [3, 3, 3, 3], [3, 3, 3]]
        labels = [1, 2, 3, 4]
        new_sequences, new_labels = remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [[1], [2, 2], [3, 3, 3]])
        self.assertEqual(new_labels, [1, 2, 4])
