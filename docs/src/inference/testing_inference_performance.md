# Testing Inference Performance

Release name: `caiman-asr-client-<version>.run`

This is a simple client for testing and reporting the latency of the CAIMAN-ASR server. It spins up a configurable number
of concurrent connections that each run a stream in realtime.

## Running

A pre-compiled binary called `caiman-asr-client` is provided. The client documentation can be viewed with the
`--help` flag.

```
$ ./caiman-asr-client --help
This is a simple client for evaluation of the CAIMAN-ASR server.

It drives multiple concurrent real-time audio channels providing latency figures and transcriptions. In default mode, it spawns a single channel for each input audio file.

Usage: caiman-asr-client [OPTIONS] <INPUTS>...

Options:
      --perpetual
          Every channel drives multiple utterances in a loop. Each channel will only print a report for the first completed utterance

      --concurrent-connections <CONCURRENT_CONNECTIONS>
          If present, drive <CONCURRENT_CONNECTIONS> connections concurrently. If there are more connections than audio files, connections will wrap over the dataset

  -h, --help
          Print help (see a summary with '-h')

WebSocket connection:
      --host <HOST>
          The host to connect to. Note that when connecting to a remote host, sufficient network bandwidth is required when driving many connections

          [default: localhost]

      --port <PORT>
          Port that the CAIMAN-ASR server is listening on

          [default: 3030]

      --connect-timeout <CONNECT_TIMEOUT>
          The number of seconds to wait for the server to accept connections

          [default: 15]

      --quiet
          Suppress printing of transcriptions

Audio:
  <INPUTS>...
          The input wav files. The audio is required to be 16 kHz S16LE single channel wav
```

If you want to run it with many wav files you can use `find` to list all the wav files in a directory (this
will hit a command line limit if you have too many):

```
./caiman-asr-client $(find /path/to/wav -name '*.wav') --concurrent-connections 1000 --perpetual --quiet
```

## Building

If you want to build the client yourself you need the rust compiler. See <https://www.rust-lang.org/tools/install>

Once installed you can compile and run it with

```
$ cargo run --release -- my_audio.wav --perpetual --concurrent-connections 1000
```

If you want the executable you can run

```
$ cargo build --release
```

and the executable will be in `target/release/caiman-asr-client`.

## Latency

The CAIMAN-ASR server provides a response for every 60 ms of audio input, even if that response has no
transcription. We can use this to calculate the latency from sending the audio to getting back the
associated response.

To prevent each connection sending audio at the same time, the client waits a random length of time
(within the frame duration) before starting each connection. This provides a better model of real
operation where the clients would be connecting independently.
