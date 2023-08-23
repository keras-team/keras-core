from keras_core import backend
from keras_core.api_export import keras_core_export

if backend.backend() == "tensorflow":
    BackendVariable = backend.tensorflow.core.Variable
    backend_name_scope = backend.tensorflow.core.name_scope
elif backend.backend() == "jax":
    BackendVariable = backend.jax.core.Variable
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "torch":
    BackendVariable = backend.torch.core.Variable
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "numpy":
    from keras_core.backend.numpy.core import Variable as NumpyVariable

    BackendVariable = NumpyVariable
    backend_name_scope = backend.common.name_scope.name_scope
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")


@keras_core_export("keras_core.Variable")
class Variable(BackendVariable):
    pass


@keras_core_export("keras_core.name_scope")
class name_scope(backend_name_scope):
    pass
