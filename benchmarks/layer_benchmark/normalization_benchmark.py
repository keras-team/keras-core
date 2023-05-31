from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark


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
