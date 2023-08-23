import torch
import torch.nn as nn

from keras_core.layers import Layer
from keras_core.backend import Variable
from keras_core.api_export import keras_core_export


@keras_core_export(["keras_core.backend.torch.TorchModuleWarpper"])
class TorchModuleWarpper(Layer):
    """Torch module wrapper layer.

    `TorchModuleWarpper` is an abstraction that can be wrapped around a
    `torch.nn.Module` to make its parameters trackable as a
    `keras_core.layers.Layer`. It works with both vanilla and lazy PyTorch
    modules.

    Args:
        module: torch.nn.Module, A vanilla or lazy PyTorch neural network module.
        name: The name of the layer (string).

    References:
    - [PyTorch docs for `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
    - [PyTorch docs for `LazyModuleMixin`](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html)

    Examples:

    Here's an example of how the `TorchModuleWarpper` can be used with vanilla PyTorch
    modules.

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    import keras_core
    from keras_core.backend.torch import TorchModuleWarpper


    class Classifier(keras_core.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Wrap all `torch.nn.Module`s with `TorchModuleWarpper`
            self.conv1 = TorchModuleWarpper(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWarpper(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
            )
            self.pool = TorchModuleWarpper(
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.flatten = TorchModuleWarpper(nn.Flatten())
            self.dropout = TorchModuleWarpper(nn.Dropout(p=0.5))
            self.fc = TorchModuleWarpper(nn.Linear(1600, 10))

        def call(self, inputs):
            x = F.relu(self.conv1(inputs))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            return F.softmax(x, dim=1)


    model = Classifier()
    model.build((1, 28, 28))
    print("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(train_loader, epochs=5)
    ```

    Here's an example of how the `TorchModuleWarpper` can be used with PyTorch
    Lazy modules.

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    import keras_core
    from keras_core.backend.torch import TorchModuleWarpper


    class LazyClassifier(keras.Model):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # You can wrap all `torch.nn.Module`s with `TorchModuleWarpper`
            # irrespective of whether they are lazy or not.
            self.conv1 = TorchModuleWarpper(
                nn.LazyConv2d(out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWarpper(
                nn.LazyConv2d(out_channels=64, kernel_size=(3, 3))
            )
            self.pool = TorchModuleWarpper(nn.MaxPool2d(kernel_size=(2, 2)))
            self.flatten = TorchModuleWarpper(nn.Flatten())
            self.dropout = TorchModuleWarpper(nn.Dropout(p=0.5))
            self.fc = TorchModuleWarpper(nn.LazyLinear(10))

        def call(self, inputs):
            x = F.relu(self.conv1(inputs))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            return F.softmax(x, dim=1)


    model = Classifier()
    model.build((1, 28, 28))
    print("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(train_loader, epochs=5)
    ```

    """

    def __init__(self, module, name=None):
        super().__init__(name=name)
        self.module = module.to("cuda")
        self.lazy = isinstance(self.module, nn.modules.lazy.LazyModuleMixin)
        if not self.lazy:
            self.track_module_parameters()

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def track_module_parameters(self):
        for param in self.module.parameters():
            variable = Variable(
                initializer=param, trainable=param.requires_grad
            )
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self, input_shape):
        sample_input = torch.ones(*input_shape).to("cuda")
        _ = self.module(sample_input)
        self.track_module_parameters()

    def call(self, inputs, **kwargs):
        if not self.built:
            self.build(inputs.shape[1:])
        return self.module.forward(inputs, **kwargs)
