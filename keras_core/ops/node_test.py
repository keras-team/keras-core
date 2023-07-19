import numpy as np

from keras_core import testing
from keras_core.backend import KerasTensor
from keras_core import ops, Layer
from keras_core.ops.node import Node


class DummyLayer(Layer):
    pass


class NodeTest(testing.TestCase):
    def test_simple_case(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        a_layer = DummyLayer()
        node = Node(a_layer, outputs=a, call_args=(), call_kwargs={})

        self.assertEqual(node.is_input, True)

        self.assertEqual(node.output_tensors[0], a)
        self.assertEqual(node.output_tensors[0].shape, shape)

    def test_single_wired_layers(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        a_layer = DummyLayer()
        node1 = Node(a_layer, outputs=a, call_args=(), call_kwargs={})

        b = KerasTensor(shape=shape)
        b_layer = DummyLayer()
        node2 = Node(b_layer, outputs=b, call_args=(a, ), call_kwargs={})

        self.assertEqual(node1.is_input, True)
        self.assertEqual(node2.is_input, False)

        self.assertEqual(node1.output_tensors[0], a)
        self.assertEqual(node1.output_tensors[0].shape, shape)

        self.assertEqual(b_layer.__dict__['_inbound_nodes'][0], node2)

        self.assertEqual(node2.parent_nodes[0], node1)

    def test_multi_wired_layers(self):
        pass

    def test_output_tensor_error(self):
        a = np.random.rand(2, 3, 4)
        a_layer = DummyLayer()
        with self.assertRaisesRegex(
                ValueError, "operation outputs must be tensors."
        ):
            Node(a_layer, outputs=a, call_args=(), call_kwargs={})
