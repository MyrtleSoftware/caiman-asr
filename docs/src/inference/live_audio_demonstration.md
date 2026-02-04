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

### Prefix (punctuation/casing)

Use `--prefix` to condition the RNN-T prediction network:

- `<pnc>`: punctuated and cased output
- `<nopnc>`: lowercase only (slightly more accurate)

For example,

```bash
python ./live_client.py --host <host> --port <port> --prefix="<pnc>"
```

Note: ensure the server is running a model that supports `prefix`.

### Keyword Boosting

CAIMAN-ASR supports keyword boosting of up to 100 keywords.
Keyword boosting is a technique to improve the recognition of domain-specific words/phrases and proper nouns,
by boosting the probability of specific tokens at inference time.

To provide keywords to the server, use the `--keywords` argument with a space-separated list of keywords e.g.

```bash
python ./live_client.py --keywords caiman asr myrtle
```

### Troubleshooting

If the client raises `OSError: [Errno -9997] Invalid sample rate`, you may need to
use a different audio input device:

1. Run `./print_input_devices.py` to list available input devices
2. Try each device using (for example) `./live_client.py --input_device 5`
