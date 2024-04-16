#! /usr/bin/env python3
import argparse
import json
import wave
from threading import Thread

import pyaudio
import websocket

# Simple client for connecting to the CAIMAN-ASR server.
# It records microphone audio, sends it in chunks to the server,
# prints the transcription to stdout and saves the audio in output.wav.

# Instructions:
# The CAIMAN-ASR server must already be running!
# python3 ./client.py
# Start talking when you see "Recording" printed to stdout.
# ctrl+c to stop recording
# `aplay output.wav` to listen to the recording


def save_wav(filename):
    print("Saving recorded audio in " + filename)
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(samplerate)
    wf.writeframes(b"".join(frames))
    wf.close()


def listen(ws):
    msg = ws.recv()
    while msg:
        print(json.loads(msg)["alternatives"][0]["transcript"], end="", flush=True)
        msg = ws.recv()
    print()


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

# Set up portaudio
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
samplerate = 16000
frame_length_ms = 60
samples_per_frame = int(samplerate * frame_length_ms / 1000)
p = pyaudio.PyAudio()

stream = p.open(
    format=sample_format,
    channels=channels,
    rate=samplerate,
    frames_per_buffer=samples_per_frame,
    input=True,
    input_device_index=args.input_device,
)

# Set up websocket connection to asr server. CAIMAN-ASR SERVER MUST ALREADY BE RUNNING!
ws = websocket.WebSocket(enable_multithread=True)
ws.connect(
    f"ws://{args.host}:{args.port}/asr/v0.1/stream"
    f"?content_type=audio/x-raw;format=S16LE;channels={channels};rate={samplerate}",
    timeout=100,
)

listener = Thread(target=listen, args=(ws,))
listener.start()

print("Recording")
frames = []  # Used to store the recorded data, so it can be saved to a file

while True:
    try:
        data = stream.read(samples_per_frame)
        ws.send(data, websocket.ABNF.OPCODE_BINARY)
        frames.append(data)
    except KeyboardInterrupt:
        break

# send closing frame to asr server
ws.send("", websocket.ABNF.OPCODE_BINARY)

# wait for the listener thread to finish
listener.join()

print("Finished recording")

# Stop and close the portaudio stream and terminate the portaudio interface
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded data as a WAV file
filename = "output.wav"
save_wav(filename)
