#! /usr/bin/env python3
import argparse
import json
import wave
from itertools import count
from threading import Thread

import pyaudio
import websocket
from stack import PipeStack, Style

# Simple client for connecting to the CAIMAN-ASR server.
# It records microphone audio, sends it in chunks to the server,
# prints the transcription to stdout and saves the audio in output.wav.

# Instructions:
# The CAIMAN-ASR server must already be running!
# python3 ./client.py
# Start talking when you see "Recording" printed to stdout.
# ctrl+c to stop recording
# `aplay output.wav` to listen to the recording


SAMPLE_FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 1
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 60
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_LENGTH_MS // 1000


def listen(ws):
    msg = ws.recv()

    stack = PipeStack()
    prev_provisional = False

    while msg:
        json_msg = json.loads(msg)

        if prev_provisional:
            stack.pop()

        transcript = json_msg["alternatives"][0]["transcript"]

        prev_provisional = json_msg["is_provisional"]
        sty = Style.PARTIAL if prev_provisional else Style.FINAL

        stack.push(transcript, sty)

        msg = ws.recv()

    print("\n")


def save_wav(filename, frames, sample_size):
    print("Saving recorded audio in " + filename)
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_size)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


class Audio(object):
    def __init__(self, input_device):
        """
        A context manager for recording audio from a microphone using PyAudio.
        """
        self.input_device = input_device

        self.py_audio = pyaudio.PyAudio()

    def __enter__(self):
        self.stream = self.py_audio.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=SAMPLES_PER_FRAME,
            input=True,
            input_device_index=self.input_device,
        )
        return self.stream, self.py_audio.get_sample_size(SAMPLE_FORMAT)

    def __exit__(self, *_):
        self.stream.stop_stream()
        self.stream.close()
        self.py_audio.terminate()


def run_asr_server(args, stream):
    """
    Run the server, returns True if the server was terminated by the client.
    """

    # Set up websocket connection to asr server. CAIMAN-ASR SERVER MUST ALREADY BE RUNNING!
    ws = websocket.WebSocket(enable_multithread=True)

    ws.connect(
        f"ws://{args.host}:{args.port}/asr/v0.1/stream"
        f"?content_type=audio/x-raw;format=S16LE;channels={CHANNELS};rate={SAMPLE_RATE}",
        timeout=100,
    )

    listener = Thread(target=listen, args=(ws,))
    listener.start()

    frames = []
    eos = False

    print("Recording:")

    while True:
        try:
            data = stream.read(SAMPLES_PER_FRAME)
            ws.send(data, websocket.ABNF.OPCODE_BINARY)
            frames.append(data)
        except KeyboardInterrupt:
            ws.send("", websocket.ABNF.OPCODE_BINARY)
            break
        except ConnectionResetError:
            print("Connection terminated by server (EOS?)")
            eos = True
            break

    listener.join()

    return eos, frames


parser = argparse.ArgumentParser(description="CAIMAN-ASR client")
parser.add_argument(
    "--host", type=str, default="localhost", help="CAIMAN-ASR server host"
)
parser.add_argument("--port", type=int, default=3030, help="CAIMAN-ASR server port")
parser.add_argument(
    "--input_device",
    type=int,
    default=None,
    help="The index of the audio input device",
)

args = parser.parse_args()

with Audio(args.input_device) as (stream, sample_size):
    for iter in count():
        eos, frames = run_asr_server(args, stream)

        save_wav(f"output-{iter}.wav", frames, sample_size)

        if not eos:
            break
