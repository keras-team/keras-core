import tree

from keras_core import testing
from keras_core.backend import KerasTensor
from keras_core.ops.symbolic_arguments import SymbolicArguments


class SymbolicArgumentsTest(testing.TestCase):
    def test_args(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        args = SymbolicArguments(
            (
                a,
                b,
            ),
            {},
        )

        self.assertEqual(args.keras_tensors, [a, b])
        self.assertEqual(args.__dict__["_flat_arguments"], [a, b])

    def test_conversion_fn(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        sym_args = SymbolicArguments(
            (
                a,
                b,
            ),
            {},
        )

        (value, _) = sym_args.convert(lambda x: x**2)
        args1 = value[0][0]
        args2 = value[0][1]

        self.assertIsInstance(args1, KerasTensor)
        self.assertNotEqual(args1, a)
        self.assertNotEqual(args2, b)

        mapped_value = tree.map_structure(lambda x: x**2, a)
        self.assertEqual(mapped_value.shape, args1.shape)
        self.assertEqual(mapped_value.dtype, args1.dtype)

    def test_fill_in(self):
        pass
