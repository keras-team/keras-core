import os
import re
import subprocess
from keras_core import backend

BACKEND_REQ = {
    "tensorflow": "tensorflow-cpu>=2.13.0",
    "torch": "torch>=2.0.1+cpu torchvision>=0.15.1+cpu",
    "jax": "jax[cpu]",
}

other_backends = list(set(BACKEND_REQ.keys())-{backend.backend()})

subprocess.run("pip install tensorflow torch jax", shell=True)
subprocess.run("rm -rf tmp_build_dir", shell=True)
build_process = subprocess.run(
    "python3 pip_build.py",
    capture_output=True,
    text=True,
    shell=True,
    )
print(build_process.stdout)
match = re.search(
    r"\s[^\s]*\.whl",
    build_process.stdout,
)
if not match:
    raise ValueError("Installed package filepath could not be found.")
whl_path = match.group()

commands = [
    # Create and activate virtual environment
    "python3 -m venv test_env",
    "source ./test_env/bin/activate",

    # Installs the backend's package and common requirements
    "pip install "+ BACKEND_REQ[backend.backend()],
    "pip install -r requirements-common.txt",

    # Ensure other backends are uninstalled
    "pip uninstall -y "+ other_backends[0] +" "+ other_backends[1],

    # Copy over and install `.whl` package
    "pip3 install "+whl_path+" --force-reinstall --no-dependencies",

    # Runs the example script
    "python3 integration_tests/basic_full_flow.py",

    # Exits virtual environment, deletes files, and any
    # miscellaneous install logs
    "exit",
    "rm -rf test_env",
    "rm -rf tmp_build_dir",
    "rm -f *+cpu",
]

for command in commands:
    subprocess.run(command, shell=True)