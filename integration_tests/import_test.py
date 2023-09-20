import os
import subprocess
from keras_core import backend

BACKEND_REQ = {
    "tensorflow": "tensorflow-cpu>=2.13.0",
    "torch": "torch>=2.0.1+cpu torchvision>=0.15.1+cpu",
    "jax": "jax[cpu]",
}

commands = [
    "python3 -m venv test_env",
    "source ./test_env/bin/activate",
    # Installs only the backend's package
    "pip install "+ BACKEND_REQ[backend.backend()],
    "pip install -r requirements-common.txt",
    "deactivate",
]

for command in commands:
    subprocess.run(command, shell=True)