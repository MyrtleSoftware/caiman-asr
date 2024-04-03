# Installation <a name="install"></a>

These steps have been tested on Ubuntu 18.04, 20.04 and 22.04.
Other Linux versions may work, since most processing takes place in a Docker container.
However, the install_docker.sh script is currently specific to Ubuntu.
Your machine does need NVIDIA GPU drivers installed.
Your machine does NOT need CUDA installed.

1. Clone the repository

```bash
git clone https://github.com/MyrtleSoftware/caiman-asr.git && cd caiman-asr
```

2. Install Docker

```bash
source training/install_docker.sh
```

3. Add your username to the docker group:

```bash
sudo usermod -a -G docker [user]
```

   Run the following in the same terminal window, and you might not have to log out and in again:

```bash
newgrp docker
```

4. Build the docker image

```bash
# Build from Dockerfile
cd training
./scripts/docker/build.sh
```

5. Start an interactive session in the Docker container mounting the volumes, as described in the next section.

```bash
./scripts/docker/launch.sh <DATASETS> <CHECKPOINTS> <RESULTS>
```


### Requirements

Currently, the reference uses CUDA-12.2.
Here you can find a table listing compatible drivers: <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver>

### Information about volume mounts

Setting up the training environment requires mounting the three directories:
`<DATASETS>`, `<CHECKPOINTS>`, and `<RESULTS>` for the training data, model checkpoints, and results, respectively.

The following table shows the mappings between directories on a host machine and inside the container.

| **Host machine** | **Inside container** |
| ------------------- | -----------------|
| training | /workspace/training |
| `<DATASETS>` | /datasets |
| `<CHECKPOINTS>` | /checkpoints |
| `<RESULTS>` | /results |


```admonish
The host directories passed to `./scripts/docker/launch.sh` must have absolute paths.
```

If your `<DATASETS>` directory contains symlinks to other drives (i.e. if your data is too large to fit on a single drive),
they will not be accessible from within the running container. In this case, you can pass the absolute paths to your drives
as the 4th, 5th, 6th, ... arguments to `./scripts/docker/launch.sh`.
This will enable the container to follow symlinks to these drives.

```admonish
During training, the model checkpoints are saved to the  `/results` directory so it is sometimes convenient to
load them from `/results` rather than from `/checkpoints`.
```

## Next Steps

Go to the [Data preparation](data_preparation.md) docs to see how to download and preprocess data in advance of training.
