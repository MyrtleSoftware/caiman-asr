# Convert PyTorch checkpoints to CAIMAN-ASR programs

Release name: `caiman-asr-server-<version>/compile-model-checkpoint`

This is a packaged version of the CAIMAN-ASR model compiler, which can be used to convert PyTorch
checkpoints to CAIMAN-ASR checkpoints. The CAIMAN-ASR checkpoint contains the instructions for the model
to enable CAIMAN-ASR acceleration. These instructions depend on the weights of the model, so when
the model is changed, the CAIMAN-ASR checkpoint needs to be recompiled.

The flow to deploy a trained CAIMAN-ASR model is:

1. Convert the training checkpoint to a hardware checkpoint following the steps in the [Exporting a checkpoint](../training/export_inference_checkpoint.md) section. Hardware checkpoints can be used with the CAIMAN-ASR server directly if you specify `--cpu-backend`.
2. Convert the hardware checkpoint to a CAIMAN-ASR checkpoint with the `compile-model.py` script in this directory.
   CAIMAN-ASR checkpoints can be used with the CAIMAN-ASR server with either of the CPU or FPGA backends.

## Usage

The program can be run with docker or directly if you install the dependencies.

### Docker

Install `docker` and run the following commands:

```bash
./build_docker.sh
./run_docker.sh path/to/hardware-checkpoint.pt output/path/to/caiman-asr-checkpoint.pt
```

### Without docker

Ensure that you are using Ubuntu 20.04 - there are libraries required by the CAIMAN-ASR assembler
that may not be present on other distributions.

```bash
pip3 install -r ./requirements.txt
./compile-model.py \
  --hardware-checkpoint path/to/hardware-checkpoint.pt \
  --mau-checkpoint output/path/to/caiman-asr-checkpoint.pt
```

These commands should be executed in the `compile-model-checkpoint` directory
otherwise the python script won't be able to find the `mau_model_compiler` binary.
