from keras_core import testing
from keras_core.backend import KerasTensor
from keras_core import ops, Layer
from keras_core.ops.node import Node


class DummyLayer(Layer):
    pass


class NodeTest(testing.TestCase):
    def test_something(self):
        a = KerasTensor(shape=(2, 3, 4))
        a_layer = DummyLayer()
        node = Node(a_layer, outputs=a, call_args=(), call_kwargs={})
        self.assertEqual(node.output_tensors[0], a)
        self.assertEqual(node.is_input, True)