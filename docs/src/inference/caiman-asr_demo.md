# CAIMAN-ASR demo
Release name: `myrtle-asr-demo-<version>.run`

This software bundle is used for demonstrating the server.
It includes the asr server and a web interface which shows the live transcriptions and latency of the server.
This is not the right software to use for production installations; the docker container doesn't expose the server port so external clients cannot connect to it.
For production installations, use the `myrtle-asr-server-<version>.run` release. See instructions in the [CAIMAN-ASR server](./caiman-asr_server.md) section.

## Running the CAIMAN-ASR Demo Server

If you are setting up the server from scratch you will need to flash the Achronix Speedster7t FPGA
with the provided bitstream.
If you have a demo system provided by Myrtle.ai or Achronix,
the bitstream will already be flashed.
See the [Programming the card](./programming_the_fpga.md) section for instructions on flashing the FPGA.

1. Load the CAIMAN-ASR Demo Server Docker image:
    ```
    docker load -i docker-asr-demo.tgz
    ```

2. Start the server with:
    ```
    ./start_server <license directory> [card index]...
    ```

   where `<license directory>` is the path to the directory containing your Myrtle.ai licence
   and `[card index]` is an optional integer list argument specifying which card indices to use, e.g. `0 1 2 3`.
   The default is 0.

   The demo GUI webpage will then be served at <!-- markdown-link-check-disable -->[http://localhost](http://localhost).

The latency may be much higher than usual during start-up. Refreshing the
webpage will reset the scale on the latency chart.

To shut down the server you can use `ctrl+c` in the terminal where the server is running.
Alternatively, run the following:

```
./kill_server
```
