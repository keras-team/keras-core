import collections
import numpy as np
import os

from keras_core import callbacks
from keras_core import layers
from keras_core.optimizers import schedules
from keras_core import models
from keras_core import optimizers
from keras_core import testing

# Note: this file and tensorboard in general has a dependency on tensorflow

# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple("_ObservedSummary", ("logdir", "tag"))


class _SummaryFile:
    """A record of summary tags and the files to which they were written.

    Fields `scalars`, `images`, `histograms`, and `tensors` are sets
    containing `_ObservedSummary` values.
    """

    def __init__(self):
        self.scalars = set()
        self.images = set()
        self.histograms = set()
        self.tensors = set()
        self.graph_defs = []
        self.convert_from_v2_summary_proto = False


def get_model_from_layers(model_layers, input_shape):
    model = models.Sequential()
    model.add(
        layers.InputLayer(
            input_shape=input_shape,
            dtype="float32",
        )
    )
    for layer in model_layers:
        model.add(layer)

def list_summaries(logdir):
    """Read all summaries under the logdir into a `_SummaryFile`.

    Args:
      logdir: A path to a directory that contains zero or more event
        files, either as direct children or in transitive subdirectories.
        Summaries in these events must only contain old-style scalars,
        images, and histograms. Non-summary events, like `graph_def`s, are
        ignored.

    Returns:
      A `_SummaryFile` object reflecting all summaries written to any
      event files in the logdir or any of its descendant directories.

    Raises:
      ValueError: If an event file contains an summary of unexpected kind.
    """
    result = _SummaryFile()
    for dirpath, _, filenames in os.walk(logdir):
        for filename in filenames:
            if not filename.startswith("events.out."):
                continue
            path = os.path.join(dirpath, filename)
            for event in tf.compat.v1.train.summary_iterator(path):
                if event.graph_def:
                    result.graph_defs.append(event.graph_def)
                if not event.summary:  # (e.g., it's a `graph_def` event)
                    continue
                for value in event.summary.value:
                    tag = value.tag
                    # Case on the `value` rather than the summary metadata
                    # because the Keras callback uses `summary_ops_v2` to emit
                    # old-style summaries. See b/124535134.
                    kind = value.WhichOneof("value")
                    container = {
                        "simple_value": result.scalars,
                        "image": result.images,
                        "histo": result.histograms,
                        "tensor": result.tensors,
                    }.get(kind)
                    if container is None:
                        raise ValueError(
                            "Unexpected summary kind %r in event file %s:\n%r"
                            % (kind, path, event)
                        )
                    elif kind == "tensor" and tag != "keras":
                        # Convert the tf2 summary proto to old style for type
                        # checking.
                        plugin_name = value.metadata.plugin_data.plugin_name
                        container = {
                            "images": result.images,
                            "histograms": result.histograms,
                            "scalars": result.scalars,
                        }.get(plugin_name)
                        if container is not None:
                            result.convert_from_v2_summary_proto = True
                        else:
                            container = result.tensors
                    container.add(_ObservedSummary(logdir=dirpath, tag=tag))
    return result

class TestTensorBoardV2(testing.TestCase):
    def setUp(self):
        super(TestTensorBoardV2, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_model(self, compile_model=True):
        layers = [
            layers.Conv2D(8, (3, 3)),
            layers.Flatten(),
            layers.Dense(1),
        ]
        model = get_model_from_layers(
            layers, input_shape=(10, 10, 1)
        )

        if compile_model:
            opt = optimizers.SGD(learning_rate=0.001)
            model.compile(
                opt, "mse"
            )
        return model

    def test_TensorBoard_default_logdir(self):
        """Regression test for cross-platform pathsep in default logdir."""
        os.chdir(self.get_temp_dir())

        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard()  # no logdir specified

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(logdir=".")
        train_dir = os.path.join(".", "logs", "train")
        validation_dir = os.path.join(".", "logs", "validation")
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=validation_dir, tag="evaluation_loss_vs_iterations"
                ),
            },
        )

    def test_TensorBoard_basic(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_across_invocations(self):
        """Regression test for summary writer resource use-after-free.

        See: <https://github.com/tensorflow/tensorflow/issues/25707>
        """
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        for _ in (1, 2):
            model.fit(
                x,
                y,
                batch_size=2,
                epochs=2,
                validation_data=(x, y),
                callbacks=[tb_cbk],
            )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_no_spurious_event_files(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        model.fit(x, y, batch_size=2, epochs=2, callbacks=[tb_cbk])

        events_file_run_basenames = set()
        for dirpath, _, filenames in os.walk(self.train_dir):
            if any(fn.startswith("events.out.") for fn in filenames):
                events_file_run_basenames.add(os.path.basename(dirpath))
        self.assertEqual(events_file_run_basenames, {"train"})

    def test_TensorBoard_batch_metrics(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir, update_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_learning_rate_schedules(self):
        model = self._get_model(compile_model=False)
        opt = optimizers.SGD(schedules.CosineDecay(0.01, 1))
        model.compile(opt, "mse")

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            callbacks=[callbacks.TensorBoard(self.logdir)],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
            },
        )

    def test_TensorBoard_global_step(self):
        model = self._get_model(compile_model=False)
        opt = gradient_descent.SGD(learning_rate_schedule.CosineDecay(0.01, 1))
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            verbose=0,
            callbacks=[
                keras.callbacks.TensorBoard(
                    self.logdir,
                    update_freq=1,
                    profile_batch=0,
                    write_steps_per_second=True,
                )
            ],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_steps_per_second"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="batch_steps_per_second"
                ),
            },
        )

    def test_TensorBoard_weight_histograms(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)
        model_type = test_utils.get_model_type()

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/histogram"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="kernel_0/histogram"
                ),
            },
        )

    def test_TensorBoard_weight_images(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, write_images=True
        )
        model_type = test_utils.get_model_type()

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/histogram"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="kernel_0/histogram"
                ),
            },
        )
        if summary_file.convert_from_v2_summary_proto:
            expected_image_summaries = {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/image"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image"),
            }
        else:
            expected_image_summaries = {
                _ObservedSummary(logdir=self.train_dir, tag="bias_0/image/0"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/0"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/1"),
                _ObservedSummary(logdir=self.train_dir, tag="kernel_0/image/2"),
            }
        self.assertEqual(
            self._strip_layer_names(summary_file.images, model_type),
            expected_image_summaries,
        )

    def test_TensorBoard_projector_callback(self):
        layers = [
            keras.layers.Embedding(10, 10, name="test_embedding"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(10,))
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            run_eagerly=test_utils.should_run_eagerly(),
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            embeddings_freq=1,
            embeddings_metadata={"test_embedding": "metadata.tsv"},
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        with open(os.path.join(self.logdir, "projector_config.pbtxt")) as f:
            self.assertEqual(
                f.readlines(),
                [
                    "embeddings {\n",
                    "  tensor_name: "
                    '"layer_with_weights-0/embeddings/.ATTRIBUTES/'
                    'VARIABLE_VALUE"\n',
                    '  metadata_path: "metadata.tsv"\n',
                    "}\n",
                ],
            )

    def test_custom_summary(self):
        if not tf.executing_eagerly():
            self.skipTest("Custom summaries only supported in V2 code path.")

        def scalar_v2_mock(name, data, step=None):
            """A reimplementation of the scalar plugin to avoid circular
            deps."""
            metadata = tf.compat.v1.SummaryMetadata()
            # Should match value in tensorboard/plugins/scalar/metadata.py.
            metadata.plugin_data.plugin_name = "scalars"
            with tf.summary.experimental.summary_scope(
                name, "scalar_summary", values=[data, step]
            ) as (tag, _):
                return tf.summary.write(
                    tag=tag,
                    tensor=tf.cast(data, "float32"),
                    step=step,
                    metadata=metadata,
                )

        class LayerWithSummary(keras.layers.Layer):
            def call(self, x):
                scalar_v2_mock("custom_summary", tf.reduce_sum(x))
                return x

        model = test_utils.get_model_from_layers(
            [LayerWithSummary()], input_shape=(5,), name="model"
        )

        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)
        x, y = np.ones((10, 5)), np.ones((10, 5))
        model.fit(
            x, y, batch_size=2, validation_data=(x, y), callbacks=[tb_cbk]
        )
        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
                _ObservedSummary(
                    logdir=self.train_dir,
                    tag="model/layer_with_summary/custom_summary",
                ),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="model/layer_with_summary/custom_summary",
                ),
            },
        )

    def _strip_layer_names(self, summaries, model_type):
        """Deduplicate summary names modulo layer prefix.

        This removes the first slash-component of each tag name: for
        instance, "foo/bar/baz" becomes "bar/baz".

        Args:
          summaries: A `set` of `_ObservedSummary` values.
          model_type: The model type currently being tested.

        Returns:
          A new `set` of `_ObservedSummary` values with layer prefixes
          removed.
        """
        result = set()
        for summary in summaries:
            if "/" not in summary.tag:
                raise ValueError(f"tag has no layer name: {summary.tag!r}")
            start_from = 2 if "subclass" in model_type else 1
            new_tag = "/".join(summary.tag.split("/")[start_from:])
            result.add(summary._replace(tag=new_tag))
        return result

    def test_TensorBoard_invalid_argument(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized arguments"):
            keras.callbacks.TensorBoard(wwrite_images=True)

    def test_TensorBoard_non_blocking(self):
        model = keras.Sequential([keras.layers.Dense(1)])
        tb = keras.callbacks.TensorBoard(self.logdir)
        self.assertTrue(tb._supports_tf_logs)
        cb_list = keras.callbacks.CallbackList(
            [tb], model=model, epochs=1, steps=100, verbose=0
        )

        tensor = tf.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, TensorBoard is causing a blocking "
                "NumPy conversion."
            )

        with tf.compat.v1.test.mock.patch.object(tensor, "numpy", mock_numpy):
            logs = {"metric": tensor}

            cb_list.on_train_begin(logs)
            cb_list.on_epoch_begin(0, logs)
            cb_list.on_train_batch_begin(0, logs)
            cb_list.on_train_batch_end(0, logs)
            cb_list.on_epoch_end(0, logs)
            cb_list.on_train_end(logs)

            cb_list.on_test_begin(logs)
            cb_list.on_test_batch_begin(0, logs)
            cb_list.on_test_batch_end(0, logs)
            cb_list.on_test_end(logs)

            cb_list.on_predict_begin(logs)
            cb_list.on_predict_batch_begin(logs)
            cb_list.on_predict_batch_end(logs)
            cb_list.on_predict_end(logs)


# Note that this test specifies model_type explicitly.
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2NonParameterizedTest(test_combinations.TestCase):
    def setUp(self):
        super(TestTensorBoardV2NonParameterizedTest, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_seq_model(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        opt = gradient_descent.SGD(learning_rate=0.001)
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())
        return model

    def _count_xplane_file(self, logdir):
        profile_dir = os.path.join(logdir, "plugins", "profile")
        count = 0
        for dirpath, dirnames, filenames in os.walk(profile_dir):
            del dirpath  # unused
            del dirnames  # unused
            for filename in filenames:
                if filename.endswith(".xplane.pb"):
                    count += 1
        return count

    def fitModelAndAssertKerasModelWritten(self, model):
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, write_graph=True, profile_batch=0
        )
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=3,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="keras"),
            },
        )
        if not model.run_eagerly:
            # There should be one train graph
            self.assertLen(summary_file.graph_defs, 1)
            for graph_def in summary_file.graph_defs:
                graph_def_str = str(graph_def)

                # All the model layers should appear in the graphs
                for layer in model.layers:
                    if "input" not in layer.name:
                        self.assertIn(layer.name, graph_def_str)

    def test_TensorBoard_writeSequentialModel_noInputShape(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_writeSequentialModel_withInputShape(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
            ]
        )
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_writeModel(self):
        inputs = keras.layers.Input([10, 10, 1])
        x = keras.layers.Conv2D(8, (3, 3), activation="relu")(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)(x)
        model = keras.models.Model(inputs=inputs, outputs=[x])
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        self.fitModelAndAssertKerasModelWritten(model)

    def test_TensorBoard_autoTrace(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=1, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_1"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_outerProfiler(self):
        """Runs a profiler session that interferes with the callback's one.

        The callback will not generate a profile but execution will proceed
        without crashing due to unhandled exceptions.
        """
        tf.profiler.experimental.start(logdir="")
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=1, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)
        tf.profiler.experimental.stop(save=False)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_1"),
            },
        )
        self.assertEqual(0, self._count_xplane_file(logdir=self.train_dir))

    def test_TensorBoard_autoTrace_tagNameWithBatchNum(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir, histogram_freq=1, profile_batch=2, write_graph=False
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_2"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileBatchRangeSingle(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="2,2",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                # Trace will be logged once at the batch it stops profiling.
                _ObservedSummary(logdir=self.train_dir, tag="batch_2"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileBatchRangeTwice(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="10,10",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=10,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        time.sleep(1)  # Avoids the second profile over-writing the first.

        model.fit(
            x,
            y,
            batch_size=3,
            epochs=10,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        self.assertEqual(2, self._count_xplane_file(logdir=self.logdir))

    # Test case that replicates a GitHub issue.
    # https://github.com/tensorflow/tensorflow/issues/37543
    def test_TensorBoard_autoTrace_profileTwiceGraphMode(self):
        tf.compat.v1.disable_eager_execution()
        inp = keras.Input((1,))
        out = keras.layers.Dense(units=1)(inp)
        model = keras.Model(inp, out)

        model.compile(gradient_descent.SGD(1), "mse")

        logdir = os.path.join(self.get_temp_dir(), "tb1")
        model.fit(
            np.zeros((64, 1)),
            np.zeros((64, 1)),
            batch_size=32,
            callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=1)],
        )
        # Verifies trace exists in the first logdir.
        self.assertEqual(1, self._count_xplane_file(logdir=logdir))
        logdir = os.path.join(self.get_temp_dir(), "tb2")
        model.fit(
            np.zeros((64, 1)),
            np.zeros((64, 1)),
            batch_size=32,
            callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=2)],
        )
        # Verifies trace exists in the second logdir.
        self.assertEqual(1, self._count_xplane_file(logdir=logdir))

    def test_TensorBoard_autoTrace_profileBatchRange(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch="1,3",
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=4,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                # Trace will be logged once at the batch it stops profiling.
                _ObservedSummary(logdir=self.train_dir, tag="batch_3"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))

    def test_TensorBoard_autoTrace_profileInvalidBatchRange(self):
        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="-1,3",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="1,None",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch="6,5",
                write_graph=False,
            )

        with self.assertRaises(ValueError):
            keras.callbacks.TensorBoard(
                self.logdir,
                histogram_freq=1,
                profile_batch=-1,
                write_graph=False,
            )

    def test_TensorBoard_autoTrace_profile_batch_largerThanBatchCount(self):
        model = self._get_seq_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = keras.callbacks.TensorBoard(
            self.logdir,
            histogram_freq=1,
            profile_batch=10000,
            write_graph=False,
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        # Enabled trace only on the 10000th batch, thus it should be empty.
        self.assertEmpty(summary_file.tensors)
        self.assertEqual(0, self._count_xplane_file(logdir=self.train_dir))