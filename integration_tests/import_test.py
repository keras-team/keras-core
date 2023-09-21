import os
import subprocess
from keras_core import backend

BACKEND_REQ = {
    "tensorflow": "tensorflow-cpu>=2.13.0",
    "torch": "torch>=2.0.1+cpu torchvision>=0.15.1+cpu",
    "jax": "jax[cpu]",
}

commands = [
    # Create and activate virtual environment
    "python3 -m venv test_env",
    "source ./test_env/bin/activate",

    # Installs the backend's package and common requirements
    "pip install "+ BACKEND_REQ[backend.backend()],
    "pip install -r requirements-common.txt",

    # Ensure other backends are uninstalled
    "pip uninstall "+(set(BACKEND_REQ.keys())-{backend.backend()})

    # Installs the Keras Core package
    "python3 pip_build.py --install",

    # Runs the example script
    "python3 integration_tests/basic_full_flow.py",

    # Exits virtual environment, deletes files, and any
    # miscellaneous install logs
    "exit",
    "rm -rf test_env",
    "rm -f *+cpu",
]

for command in commands:
    subprocess.run(command, shell=True)