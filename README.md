# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.



uv venv --python 3.11.5

source .venv/bin/activate
.venv/Scripts/Activate.ps1   

$env:Path += ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64"

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

 uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

uv run --no-sync nsys profile -o result_final python benchmark.py

python cs336_systems/benchmark.py --suite


uv run nsys profile --force-overwrite true -o result python benchmark.py





# Profile a single configuration
nsys profile -o small_forward --trace=cuda,nvtx python profile_benchmark.py --model_size small --context_length 128 --mode forward

# Export to CSV
nsys stats --report cuda_gpu_kern_sum --format csv small_forward.nsys-rep > small_forward_kernels.csv

# Analyze
python analyze_nsys.py small_forward_kernels.csv


WSL:
python3 -m venv cs336-env
source cs336-env/bin/activate

# Create virtual environment
python3 -m venv cs336-env

# Activate it
source cs336-env/bin/activate

# Install PyTorch with CUDA (adjust CUDA version to match your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install einops jaxtyping pandas

cd cs336_systems



# Make executable
chmod +x run_all_profiles.sh

./run_all_profiles.sh

sudo apt update
sudo apt install nsight-systems-cli



WSL

# Install the 2024.6.2 version (stable and recent)
sudo apt install -y nsight-systems-2024.6.2

# Verify it works
nsys --version

# Make sure you're in the virtual environment
source /mnt/c/Users/lilyx/git/stanford-cs336-assignment2-systems/cs336-env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip3 install einx einops jaxtyping pandas numpy


Powershell

which python
pip3 install einx einops jaxtyping pandas numpy
Remove-Item -Path .\nsys_profiles_win\ -Recurse -Force
