# Product Overview

The CAIMAN-ASR solution is used via a websocket interface.
Clients send audio data in chunks to the server, which returns transcriptions.

## Components of the solution

- CAIMAN-ASR bitstream: this is the bitstream that is flashed onto the FPGA.
  This bitstream supports all of the ML model architectures (for more details on the architectures, see the [Models](./training/model_yaml_configurations.md) section).
  This only needs to be reprogrammed when a new update is released.

- CAIMAN-ASR program: This contains the model weights and the instructions for a particular model architecture (e.g. base, large).
  It is loaded at runtime.
  The program is compiled from the `hardware-checkpoint` produced during training.
  For more details on how to compile the CAIMAN-ASR program, see [Compiling weights](./inference/compiling_weights.md).
  Pre-trained weights are provided for English-language transcription for the base and large architectures.

- CAIMAN-ASR server:
  This provides a websocket interface for using the solution.
  It handles loading the program and communicating to and from the card.
  One server controls one card; if you have multiple cards, you can run multiple servers and
  use load-balancing.

- ML training repository:
  This allows the user to [train](training/training.md) their own models, [validate](training/validation.md) on their own data, and [export](training/export_inference_checkpoint.md) model weights for the server.

An example configuration of CAIMAN-ASR is as follows:

![Example configuration.](./assets/caiman-structure.drawio.svg)
