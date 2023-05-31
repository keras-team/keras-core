from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark


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


benchmark_conv2D(4000, 20, 199, jit_compile=False)
