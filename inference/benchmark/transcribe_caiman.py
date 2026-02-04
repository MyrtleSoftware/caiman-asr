#!/usr/bin/env python3
import argparse
import ast
import json
import threading
from pathlib import Path
from urllib.parse import urlencode

import websocket
from beartype import beartype
from beartype.typing import List
from ctm import CTM, CtmItem
from file_streamer import FileStreamer
from measures import calculate_latency, calculate_wer
from timer import Timer
from transcriber import Transcriber
from utils import (
    bold_green,
    bold_yellow,
    env_vars,
    ftrans_is_valid,
    load_manifest_or_fail,
    make_transcript_dir,
    maybe_download_librispeech,
    save_results_to_csv,
    shared_args,
)

from caiman_asr_train.latency.client import ServerResponse, get_word_timestamps


def caiman_parser():
    parser = argparse.ArgumentParser(description="Caiman-ASR parser")
    parser.add_argument(
        "--address",
        required=True,
        type=str,
        help="Server address",
    )
    parser.add_argument(
        "--port",
        default=3030,
        type=int,
        help="ASR service port",
    )
    parser.add_argument(
        "--pause_after_each",
        action="store_true",
        help="After streaming each audio, pause the program until the user presses Enter",
    )
    parser.add_argument(
        "--ignore_server_response",
        action="store_true",
        help="Stream audio to server but do not listen for responses",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        choices=["<pnc>", "<nopnc>"],
        help="Prefix to condition RNN-T prediction network: <pnc> or <nopnc>",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="List of keywords to be boosted.",
    )
    shared_args(parser)
    return parser


@beartype
class CaimanASR(Transcriber):
    def __init__(
        self,
        address: str,
        port: int,
        limit_to: int | None,
        trans_dir: Path,
        retries: int,
        data_dir: Path,
        play_audio: bool,
        pause_after_each: bool,
        ignore_server_response: bool,
        prefix: str | None,
        keywords: list[str] | None,
    ) -> None:
        super().__init__(
            limit_to, trans_dir, retries, data_dir, play_audio, "caiman-asr.trans"
        )

        base = f"ws://{address}:{port}/asr/v0.1/stream"
        content_type = "audio/x-raw;format=S16LE;channels=1;rate=16000"
        query = {
            "content_type": content_type,
        }
        if prefix is not None:
            query["prefix"] = prefix
        if keywords is not None:
            query["keywords"] = keywords
        query = urlencode(query, doseq=True, safe=";/")
        self.ws_config = f"{base}?{query}"

        self.ws = websocket.WebSocket()
        self.pause_after_each = pause_after_each
        self.ignore_server_response = ignore_server_response

    def on_message(self, msg) -> str | None:
        if not msg:
            return

        msg = json.loads(msg)
        trans = msg["alternatives"][0]["transcript"]
        if not trans:
            return

        # Deconstruct message and write to trans file
        start = msg["start"]
        end = msg["end"]
        conf = msg["alternatives"][0]["confidence"]
        if msg["is_provisional"] is True:
            return (
                f"partial_received;{self.timer.datetime()};"
                f"{[{'text': trans, 'start': start, 'end': end, 'confidence': conf}]};"
                f"N/A"
            )
        else:
            return (
                f"final_received;{self.timer.datetime()};"
                f"{[{'text': trans, 'start': start, 'end': end, 'confidence': conf}]};"
                f"N/A"
            )

    def attempt_to_transcribe(self, file: str, ftrans: Path, duration: float) -> None:
        messages = []
        self.ws.connect(self.ws_config)

        # Initialize FileStreamer and start streaming
        file_streamer = FileStreamer(
            file,
            self.data_dir,
            timer=self.timer,
            chunk_duration=0.06,
            play_audio=self.play_audio,
        )

        def fn():
            with file_streamer as streamer:
                for chunk in streamer.stream():
                    # Send chunk, receive response, save to trans
                    self.ws.send(chunk, websocket.ABNF.OPCODE_BINARY)

            # Send end-of-stream message
            self.ws.send("", websocket.ABNF.OPCODE_BINARY)

        t = threading.Thread(target=fn, daemon=True)
        t.start()

        if not self.ignore_server_response:
            while True:
                try:
                    msg = self.ws.recv()
                    trans = self.on_message(msg)
                    if trans is not None:
                        messages.append(trans)
                except websocket.WebSocketConnectionClosedException:
                    break

        t.join()

        # Close transcription
        file_streamer.close_trans()
        if self.pause_after_each:
            input("Press Enter to stream next audio")

        self.write_trans_to_file(ftrans, file_streamer, messages)

    def trans_to_ctm(self, manifest: List[dict]) -> CTM:
        """
        Convert the transcriptions to a CTM file.
        """
        ctm = CTM()
        invalid_files = 0
        for item in manifest:
            prev_len = len(ctm.items)
            file = item["files"][0]["fname"]
            ftrans = self.trans_dir / f"{Path(file).stem}.{self.suffix}"

            if not ftrans_is_valid(ftrans):
                invalid_files += 1
                continue

            with open(ftrans, "r") as fh:
                lines = fh.readlines()

            responses = []
            for line in lines:
                entries = line.strip().split(";")
                event_type = entries[0]

                # On first packet record when was it sent
                if event_type == "session_start":
                    dtime_start = entries[1]
                elif event_type == "partial_received":
                    responses.append(
                        ServerResponse(
                            *extract_text_and_timestamp(
                                entries, self.timer, dtime_start
                            ),
                            True,
                        )
                    )

                elif event_type == "final_received":
                    responses.append(
                        ServerResponse(
                            *extract_text_and_timestamp(
                                entries, self.timer, dtime_start
                            ),
                            False,
                        )
                    )
            words = get_word_timestamps(responses)

            ctm.add_items(
                [CtmItem(file, 1, w[0].strip(), w[1], 0.0, 1.0) for w in words if w[0]]
            )

            if prev_len == len(ctm.items):
                # The server predicted no tokens for the audio.
                # If no tokens are added to the ctm, the WER calculator
                # will skip this file, which is incorrect: The WER should
                # be penalized because the server predicted "" instead of
                # the nonempty ground truth. Hence add a dummy token to the
                # CTM. WER standardization will replace <token> with ""
                # and correctly calculate WER for this file.
                ctm.add_items([CtmItem(file, 1, "<empty>", 0.0, 0.0, 0.0)])
        if invalid_files == 0:
            print(bold_green("The server transcribed all audios without errors"))
        else:
            print(
                bold_yellow(
                    f"The server didn't transcribe {invalid_files} audios. "
                    "They will be excluded from the latency/WER calculations"
                )
            )

        return ctm


def main(args):
    assert args.append_results in ["caiman-base", "caiman-large"]

    tmp_dir, data_dir, dset, manifest_fpath, ref_ctm_fpath = env_vars(args)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # 1) Data-prep step
    if args.dset == "librispeech-dev-clean":
        maybe_download_librispeech(
            args.force_data_prep, manifest_fpath, dset, ref_ctm_fpath
        )

    transcript_dir = make_transcript_dir(args.run_name)
    hyp_ctm_fpath = transcript_dir / f"{dset}.caiman-asr.ctm"
    manifest = load_manifest_or_fail(manifest_fpath)
    transcriber = CaimanASR(
        args.address,
        args.port,
        args.limit_to,
        transcript_dir,
        args.retries,
        data_dir,
        play_audio=args.play_audio,
        pause_after_each=args.pause_after_each,
        ignore_server_response=args.ignore_server_response,
        prefix=args.prefix,
        keywords=args.keywords,
    )

    # 2) Transcription step
    if not args.skip_transcription:
        transcriber.transcribe(manifest, force=args.force_transcription)

    # 3) Evaluation step
    if not args.skip_evaluation:
        hyp_ctm = transcriber.trans_to_ctm(manifest)
        hyp_ctm.to_file(hyp_ctm_fpath)
        ref_ctm = CTM.from_file(ref_ctm_fpath)
        latency_metrics, utt_stats = calculate_latency(ref_ctm, hyp_ctm)
        wer_metrics = calculate_wer(ref_ctm, hyp_ctm)
        save_results_to_csv(
            wer_metrics,
            latency_metrics,
            utt_stats,
            "CAIMAN",
            transcript_dir,
            args.run_name,
            args.append_results,
            args.custom_timestamp,
        )


@beartype
def extract_text_and_timestamp(
    entries: list[str], timer: Timer, dtime_start: str
) -> tuple[str, float]:
    # Split line into data
    _, dtime_curr, trans, _ = entries

    # Convert string into list
    trans = ast.literal_eval(trans)[0]["text"]
    tstamp = timer.datetime_diff(dtime_curr, dtime_start)
    return trans, tstamp


if __name__ == "__main__":
    parser = caiman_parser()
    args = parser.parse_args()
    main(args)
