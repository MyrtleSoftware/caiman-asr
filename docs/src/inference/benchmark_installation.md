# Benchmark installation

Follow these installation instructions to run the benchmark scripts in a Linux-based OS.

## Install using a virtual environment

(1) Install the basic dependencies:

```bash
cd benchmark
sudo ./install_basic_deps.bash
```

(2) The following command will install the Python dependencies in a virtual environment:

```bash
./install_python_deps.bash
```

Note this requires Python >= 3.10.

If you don't have the correct version of Python, you can
use [uv](https://docs.astral.sh/uv/) to create
the virtual environment with the correct version of Python:

```bash
USE_UV=true ./install_python_deps.bash
```

(3) Launch the virtual environment. All commands are meant to be run inside the environment:

```bash
source .venv/bin/activate
```

## Install using Docker

(1) If you do not have docker installed, see [here](https://caiman-asr.myrtle.ai/training/installation.html).

(2) Build the benchmarking docker image (takes about 10 minutes):

```bash
cd benchmark
./build.bash
```
