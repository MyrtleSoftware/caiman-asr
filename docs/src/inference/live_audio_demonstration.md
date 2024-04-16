# Demonstrating live transcription using a local microphone

The live demo client is found in the `inference/live_demo_client` directory.

This script connects to a running CAIMAN-ASR server and streams audio from your microphone to the server for low-latency transcription.

## Step 1: Set up the CAIMAN-ASR server

Follow the instructions provided in [Inference Flow](./inference_flow.md) to set up the ASR server.

## Step 2: Set up the client

Locally, install dependencies:

```sh
sudo apt install portaudio19-dev python3-dev # dependencies of PyAudio
pip install pyaudio websocket-client
```

(Or if you are a Nix user, `nix develop` will install those)

Then run the client with `./live_client.py --host <host> --port <port>` where `<host>` and `<port>` are the host and port of the ASR server respectively.

## Troubleshooting

If the client raises `OSError: [Errno -9997] Invalid sample rate`, you may need to
use a different audio input device:

1. Run `./print_input_devices.py` to list available input devices
2. Try each device using (for example) `./live_client.py --input_device 5`
