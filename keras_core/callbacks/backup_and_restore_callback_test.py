import pytest

from keras_core import callbacks
from keras_core import testing


class BackupAndRestoreCallbackTest(testing.TestCase):
    class InterruptingCallback(callbacks.Callback):
        """A callback to intentionally introduce interruption to
        training."""

        batch_count = 0
        epoch_int = 2
        steps_int = 5

        def on_epoch_end(self, epoch, log=None):
            if epoch == self.epoch_int:
                raise RuntimeError("EpochInterruption")

        def on_batch_end(self, batch, logs=None):
            self.batch_count += 1
            if self.batch_count == self.steps_int:
                raise RuntimeError("StepsInterruption")

    # Checking for invalid backup_dir
    def test_empty_backup_dir(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Empty " "`backup_dir`"
        ):
            callbacks.BackupAndRestoreCallback(backup_dir=None)

    # Checking save_freq and save_before_preemption both unset
    def test_save_set_error(self):
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="`save_freq` or "
            "`save_before_preemption` "
            ""
            "must be set",
        ):
            callbacks.BackupAndRestoreCallback(
                backup_dir="backup_dir",
                save_freq=None,
                save_before_preemption=False,
            )

    # Check invalid save_freq, both string and non integer
    def test_save_freq_unknown_error(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Unrecognized save_freq"
        ):
            callbacks.BackupAndRestoreCallback(
                backup_dir="backup_dir", save_freq="batch"
            )

        with self.assertRaisesRegex(
            ValueError, expected_regex="Unrecognized save_freq"
        ):
            callbacks.BackupAndRestoreCallback(
                backup_dir="backup_dir", save_freq=0.15
            )

    # Checking if after interruption, correct model params and weights are
    # loaded
    @pytest.mark.requires_trainable_backend
    def test_best_case(self):
        pass
