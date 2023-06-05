""" Benchmark core layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.core_benchmark \
    --benchmark_name=benchmark_dense \
    --num_samples=8192 \
    --batch_size=1024 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_dense(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Dense"
    init_args = {"units": 128}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[
            256,
            256,
        ],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_einsum_dense(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "EinsumDense"
    init_args = {
        "equation": "abc,cd->abd",
        "output_shape": (None, 128),
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[64, 32],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


BENCHMARK_NAMES = {
    "benchmark_dense": benchmark_dense,
    "benchmark_einsum_dense": benchmark_einsum_dense,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name is None:
        for name, benchmark_fn in BENCHMARK_NAMES:
            benchmark_fn(num_samples, batch_size, jit_compile)
        return

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
            f"be one of {BENCHMARK_NAMES.keys()}"
        )
    benchmark_fn = BENCHMARK_NAMES[benchmark_name]
    benchmark_fn(num_samples, batch_size, jit_compile)


if __name__ == "__main__":
    app.run(main)
