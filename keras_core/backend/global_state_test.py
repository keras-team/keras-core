from keras_core.backend import global_state
from keras_core.testing import test_case
from keras_core.utils.naming import auto_name


class GlobalStateTest(test_case.TestCase):
    def test_clear_session(self):
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
        name1 = auto_name("somename")
        self.assertEqual(name1, "somename_1")
        global_state.clear_session()
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
