from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def benchmark_conv1D(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv1D"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 4],
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


def benchmark_conv2D(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv2D"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
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


def benchmark_conv3D(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv3D"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 32, 4],
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


def benchmark_depthwise_conv1D(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "DepthwiseConv1D"
    init_args = {
        "kernel_size": 16,
        "depth_multiplier": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 4],
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


def benchmark_depthwise_conv2D(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "DepthwiseConv2D"
    init_args = {
        "kernel_size": 16,
        "depth_multiplier": 2,
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


def benchmark_conv1D_transpose(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv1DTranspose"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 4],
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


def benchmark_conv2D_transpose(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv2DTranspose"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
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


def benchmark_conv3D_transpose(
    num_samples,
    batch_size,
    num_iterations=None,
    jit_compile=True,
):
    layer_name = "Conv3DTranspose"
    init_args = {
        "filters": 16,
        "kernel_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[32, 32, 32, 4],
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
    "benchmark_conv1D": benchmark_conv1D,
    "benchmark_conv2D": benchmark_conv2D,
    "benchmark_conv3D": benchmark_conv3D,
    "benchmark_depthwise_conv1D": benchmark_depthwise_conv1D,
    "benchmark_depthwise_conv2D": benchmark_depthwise_conv2D,
    "benchmark_conv1D_transpose": benchmark_conv1D_transpose,
    "benchmark_conv2D_transpose": benchmark_conv2D_transpose,
    "benchmark_conv3D_transpose": benchmark_conv3D_transpose,
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
