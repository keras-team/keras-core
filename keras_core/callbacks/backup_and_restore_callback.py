from keras_core.api_export import keras_core_export
from keras_core.callbacks.callback import Callback


@keras_core_export("keras_core.callbacks.BackupAndRestoreCallback")
class BackupAndRestoreCallback(Callback):
    """
    Callback to back up and restore the training state.

    BackupAndRestore callback is intended to recover training from an
    interruption that has happened in the middle of a Model.fit execution,
    by backing up the training states in a temporary checkpoint file (with
    the help of a tf.train.CheckpointManager), at the end of each epoch. Each
    backup overwrites the previously written checkpoint file, so at any given
    time there is at most one such checkpoint file for backup/restoring purpose.

    If training restarts before completion, the training state (which
    includes the Model weights and epoch number) is restored to the most
    recently saved state at the beginning of a new Model.fit run. At the
    completion of a Model.fit run, the temporary checkpoint file is deleted.

    Note that the user is responsible to bring jobs back after the
    interruption. This callback is important for the backup and restore
    mechanism for fault tolerance purpose, and the model to be restored from
    a previous checkpoint is expected to be the same as the one used to back
    up. If user changes arguments passed to compile or fit, the checkpoint
    saved for fault tolerance can become invalid.

    Args:
        backup_dir: String, path to store the checkpoint. e.g. backup_dir =
            os.path.join(working_dir, 'backup'). This is the directory in which
            the system stores temporary files to recover the model from jobs
            terminated unexpectedly.
        save_freq: 'epoch', integer, or False. When set to 'epoch'
            the callback saves the checkpoint at the end of each epoch. When set
            to an integer, the callback saves the checkpoint every save_freq
            batches. Set save_freq to False if only using preemption
            checkpointing (with save_before_preemption=True).
        delete_checkpoint: Boolean, default to True. This BackupAndRestore
            callback works by saving a checkpoint to back up the training state.
            If delete_checkpoint=True, the checkpoint will be deleted after
            training is finished. Use False if you'd like to keep the checkpoint
            for future usage.
        save_before_preemption: A boolean value instructing whether to turn on
            the automatic checkpoint saving for preemption/maintenance events.

    """

    def __int__(
        self,
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False,
    ):
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint
        self.save_before_preemption = save_before_preemption

        if not backup_dir:
            raise ValueError("Empty `backup_dir` argument passed")
        self.backup_dir = backup_dir

        if (not save_freq) and (not save_before_preemption):
            raise ValueError(
                "Either `save_freq` or `save_before_preemption` " "must be set."
            )

    def on_epoch_begin(self, epoch, logs=None):
        self.model


