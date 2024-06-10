# CAIMAN-ASR server release bundle
Release name: `caiman-asr-server-<version>.run`

This release bundle contains all the software needed to run the Myrtle.ai CAIMAN-ASR server in a production environment.
This includes the server docker image, a simple Python client, and scripts to start and stop the server.
Additionally, it contains a script to compile a hardware checkpoint into a CAIMAN-ASR checkpoint.
Three model architectures are supported:

* `testing`
* `base`
* `large`

```admonish
The `testing` config is not recommended for production use.
See details [here](../training/model_yaml_configurations.md)
```

The CAIMAN-ASR server supports two backends: CPU and FPGA. The CPU backend is not real time, but
can be useful for testing on a machine without an Achronix Speedster7t PCIe card
installed. The FPGA backend is able to support 2000 concurrent transcription
streams per card with the `base` model and 800 with the `large` model.

## Quick start: CPU backend

1. Load the CAIMAN-ASR server docker image:

   ```
   docker load -i docker-asr-server.tgz
   ```

2. Start the CAIMAN-ASR server with the hardware checkpoint:

   ```
   ./start_asr_server.sh --rnnt-checkpoint compile-model-checkpoint/hardware_checkpoint.testing.example.pt --cpu-backend
   ```

3. Once the server prints "Server started on port 3030", you can start the simple client.
   This will send a librispeech example wav to the CAIMAN-ASR server and print the transcription:

   ```
   cd simple_client
   ./build.sh # only needed once to install dependencies
   ./run.sh
   cd ..
   ```
   To detach from the running docker container without killing it, use ctrl+p followed by ctrl+q.

4. Stop the CAIMAN-ASR server(s)

   ```
   ./kill_asr_servers.sh
   ```

## Quick start: FPGA backend

If you are setting up the server from scratch you will need to flash the Achronix Speedster7t FPGA
with the provided bitstream. If you have a demo system provided by Myrtle.ai or Achronix,
the bitstream will already be flashed.
See the [Programming the card](./programming_the_fpga.md) section for instructions on flashing the FPGA before continuing.

```admonish
Unlike with the CPU backend, you will need to compile the hardware checkpoint into a CAIMAN-ASR checkpoint (step 2
below). For more details on this process, see the [Compiling weights](./compiling_weights.md) section.
```

1. Load the CAIMAN-ASR server docker image

   ```
   docker load -i docker-asr-server.tgz
   ```

2. Compile an example hardware checkpoint to a CAIMAN-ASR checkpoint

    ```
    cd compile-model-checkpoint
    ./build_docker.sh
    ./run_docker.sh hardware_checkpoint.testing.example.pt caiman_asr_checkpoint.testing.example.pt
    cd ..
    ```

3. Start the CAIMAN-ASR server with the CAIMAN-ASR checkpoint (use `--card-id 1` to use the second card).
   `--license-dir` should point to the directory containing your license files. See the [Licensing](./licensing.md) section for more information.

   ```
   ./start_asr_server.sh --rnnt-checkpoint compile-model-checkpoint/caiman_asr_checkpoint.testing.example.pt --license-dir "./licenses/" --card-id 0
   ```
   To detach from the running docker container without killing it, use ctrl+p followed by ctrl+q.

4. Once the server prints "Server started on port 3030", you can start the simple client.
   This will send a librispeech example wav to the CAIMAN-ASR server and print the transcription:

   ```
   cd simple_client
   ./build.sh  # only needed once to install dependencies
   ./run.sh
   cd ..
   ```

5. Stop the CAIMAN-ASR server(s)

   ```
   ./kill_asr_servers.sh
   ```

## State resets
State resets improve the word error rate of the CAIMAN-ASR server on long utterances by resetting the hidden state of the model after a fixed duration.
This improves the accuracy but reduces the number of real-time streams that can be supported by about 25%.
If your audio is mostly short utterances (less than 60s), you can disable state resets to increase the number of real-time streams that can be supported.
State resets are switched on by default, but they can be disabled by passing the `--no-state-resets` flag to the `./start_server` script.

 More information about state resets can be found [here](../training/state_resets.md).

# Connecting to the websocket API
The websocket endpoint is at `ws://localhost:3030`.
See [Websocket API](./websocket_api.md) for full documentation of the websocket interface.

The code in `simple_client/simple_client.py` is a simple example of how to connect to the CAIMAN-ASR server using the websocket API.
The code snippets below are taken from this file, and demonstrate how to connect to the server in Python.

Initially the client needs to open a websocket connection to the server.
```python
ws = websocket.WebSocket()
ws.connect(
    "ws://localhost:3030/asr/v0.1/stream?content_type=audio/x-raw;format=S16LE;channels=1;rate=16000"
)
```

Then the client can send audio data to the server.
```python
for i in range(0, len(samples), samples_per_frame):
    payload = samples[i : i + samples_per_frame].tobytes()
    ws.send(payload, websocket.ABNF.OPCODE_BINARY)
```

The client can receive the server's response on the same websocket connection. Sending and receiving can be interleaved.
```python
msg = ws.recv()
print(json.loads(msg)["alternatives"][0]["transcript"], end="", flush=True)
```

When the audio stream is finished the client should send a blank frame to the server to signal the end of the stream.
```python
ws.send("", websocket.ABNF.OPCODE_BINARY)
```

The server will then send the final transcriptions and close the connection.

The server consumes audio in 60ms frames, so for optimal latency the client should send audio in 60ms frames.
If the client sends audio in smaller chunks the server will wait for a complete frame before processing it.
If the client sends audio in larger chunks there will be a latency penalty as the server waits for the next frame to arrive.

A more advanced client example in Rust is provided in `caiman-asr-client`; see [Testing inference performance](./testing_inference_performance.md) for more information.
