from keras.src.engine import base_layer

from keras_core import testing
from keras_core.backend import KerasTensor


class DummyLayer(base_layer.Layer):
    pass


class NodeTest(testing.TestCase):

    def test_happy_path(self):
        a = KerasTensor(shape=(2, 3, 5))
        b = KerasTensor(shape=(2, 3, 5))

        a_layer = DummyLayer()
        # node = Node(a_layer, outputs=a)
        # self.assertEqual(node.output_tensors, a)
