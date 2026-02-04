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

      --prefix <PREFIX>
          Prefix to the prediction network to select the output style, either "<pnc>" for punctuated and cased output,
          or "<nopnc>" for lowercase only (more accurate).

          [default: None]


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

## How the latency is calculated

This client splits the audio in 60ms chunks and sends a chunk every 60ms, so it's mimicking live audio.
The latency is measured by the client, and it is the time from the chunk being sent to the response for that chunk being
received.
(The CAIMAN-ASR server sends a response for every 60ms chunk it receives, whether or not it has a token for this chunk,
so this is always well-defined.)

This means that the reported latency includes the following:

- The time for the packet to go from the client to the server across the network
- The time for the chunk to be processed by the server
- The time for the response to go from the server back to the client.

It doesn't include:

- The size of the chunk itself (60ms). If you imagine a live stream of audio that starts at t=0, then at t=60ms the
  first chunk of audio would be sent. Say that the time taken to receive a response from the server is 50ms, then the
  first response would be received at 110ms. But we don't say that the latency is 110ms because we don't include the
  size of the chunk in the calculation.

The latency reported by the client is the average and p99 for all the chunks in the previous (configurable) reporting
period.
The latency reported when the transcription comes back is the average and p99 for all the chunks in that transcription.

To prevent each connection sending audio at the same time, the client waits a random length of time
(within the frame duration) before starting each connection. This provides a better model of real
operation where the clients would be connecting independently.

## Troubleshooting

The client uses a file descriptor for each audio connection.
The number of file descriptors available to a process is limited by the operating system.
If you see the error `Too many open files`, you can increase the limit by adding the following lines
to `/etc/security/limits.conf`:

```
* soft nofile 40000
* hard nofile 40000
```

You can check the limit with:

```
ulimit -n
```
