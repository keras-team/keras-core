import pytest
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


class IdentityTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_identity_basics(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors.")
        self.run_layer_test(
            layers.Identity,
            init_kwargs={},
            input_shape=(2, 3),
            input_sparse=sparse,
            expected_output_shape=(2, 3),
            expected_output_sparse=sparse,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            run_training_check=not sparse,
            supports_masking=True,
        )
