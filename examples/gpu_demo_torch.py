import torch
import torch.nn as nn
import torch.optim as optim
from keras_core import layers
import keras_core
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
learning_rate = 0.01
batch_size = 128
num_epochs = 1

def get_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras_core.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    return dataset

def get_model():
    # Create the Keras model
    model = keras_core.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ]
    )
    return model

#################################################################
######## Writing a torch training loop for a Keras model ########
#################################################################


def train(model, train_loader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print loss statistics
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 10}"
                )
                running_loss = 0.0

def setup(rank, world_size):
    # Device setup
    device = torch.device("cuda:{}".format(rank))
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(device)


def prepare(dataset, rank, world_size, batch_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # Create a DataLoader
    train_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader

def cleanup():
    # Cleanup
    dist.destroy_process_group()


def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)

    dataset = get_data()
    model = get_model()

    # prepare the dataloader
    dataloader = prepare(dataset, rank, world_size, batch_size)

    # Instantiate the torch optimizer
    print("Num params:", len(list(model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate the torch loss function
    loss_fn = nn.CrossEntropyLoss()

    # Put model on device
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train(ddp_model, train_loader, num_epochs, optimizer, loss_fn)

    cleanup()


################################################################
######## Using a Keras model or layer in a torch Module ########
################################################################

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = keras_core.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes),
            ]
        )

    def forward(self, x):
        return self.model(x)


# torch_module = MyModel()

# # Instantiate the torch optimizer
# print("Num params:", len(list(torch_module.parameters())))
# optimizer = optim.Adam(torch_module.parameters(), lr=learning_rate)

# # Instantiate the torch loss function
# loss_fn = nn.CrossEntropyLoss()

# train(torch_module, train_loader, num_epochs, optimizer, loss_fn)

if __name__ == "__main__":
    # GPU parameters
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        main,
        args=(world_size),
        nprocs=world_size
    )