""" Benchmark normalization layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.normalization_benchmark \
    --benchmark_name=benchmark_batch_normalization \
    --num_samples=1000 \
    --batch_size=20 \
    --jit_compile=True 
```
"""

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

from absl import app
from absl import flags

FLAGS = flags.FLAGS

def benchmark_batch_normalization(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "BatchNormalization"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 4],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )


def benchmark_group_normalization(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "GroupNormalization"
    init_args = {
        "groups": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 4],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )


def benchmark_layer_normalization(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "LayerNormalization"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 4],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

BENCHMARK_NAMES = {
    "benchmark_batch_normalization": benchmark_batch_normalization,
    "benchmark_group_normalization": benchmark_group_normalization,
    "benchmark_layer_normalization": benchmark_layer_normalization,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    num_iterations = FLAGS.num_iterations
    jit_compile = FLAGS.jit_compile

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
            f"be one of {BENCHMARK_NAMES.keys()}"
        )
    benchmark_fn = BENCHMARK_NAMES[benchmark_name]
    benchmark_fn(num_samples, batch_size, num_iterations, jit_compile)


if __name__ == "__main__":
    app.run(main)