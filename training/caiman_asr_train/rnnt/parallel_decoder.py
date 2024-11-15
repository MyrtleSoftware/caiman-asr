import gc
from itertools import count

import torch
import torch.multiprocessing as mp
from beartype import beartype
from beartype.typing import Dict, List, Type
from psutil import cpu_count

from caiman_asr_train.rnnt.decoder import RNNTDecoder
from caiman_asr_train.rnnt.response import FrameResponses


@beartype
def ceil_div(x: int, *, by: int) -> int:
    """
    Integer division that rounds away from zero.
    """
    if x >= 0:
        return (x + by - 1) // by
    else:
        return (x - by + 1) // by


@beartype
def get_num_procs(args, world_size: int) -> int:
    """
    If args.beam_decoder_procs_per_gpu is less than 1 return a sensible
    default number of processes to use per GPU. Otherwise return the
    number of processes specified by the user.
    """
    if args.beam_decoder_procs_per_gpu < 1:
        # Clamp to 8 because empirical testing showed that
        # exceeding this did not improve performance.
        max_cpu_per_gpu = min(max(1, cpu_count(logical=False) // world_size), 8)

        max_useful_procs = ceil_div(
            args.val_batch_size, by=args.beam_min_decode_batch_size_per_proc
        )

        nprocs_per_gpu = min(max_cpu_per_gpu, max_useful_procs)

        if nprocs_per_gpu == 1:
            print("WARNING: Selecting only 1 decoder process per GPU")
        else:
            print(f"Selecting {nprocs_per_gpu} decoder processes per GPU")

        return nprocs_per_gpu
    else:
        return args.beam_decoder_procs_per_gpu


class ParallelDecoder(RNNTDecoder):
    @beartype
    def __init__(self, nprocs: int, min_batch: int, klass: Type[RNNTDecoder], **kwargs):
        """
        A wrapper around a decoder that uses multiprocessing to
        parallelize a batch of decoding. Exposes the RNNTDecoder API.

        nprocs: Number of processes to use.
        min_batch: The number of utterances to batch together for each worker.
        klass: The decoder class that each worker will use.
        kwargs: The arguments to pass to the decoder class, these will
            pickled and copies sent to each worker. The arguments "model"
            and "max_inputs_per_batch" with the same semantics as
            RNNTDecoder are required.
        """
        super().__init__(
            model=kwargs["model"],
            eos_strategy=kwargs["eos_strategy"],
            blank_idx=kwargs["blank_idx"],
            max_inputs_per_batch=kwargs["max_inputs_per_batch"],
        )

        self.min_batch = min_batch

        # pytorch can only send cuda tensors with spawn
        smp = mp.get_context("spawn")

        self.todo = smp.SimpleQueue()
        self.result = smp.SimpleQueue()

        args = (self.todo, self.result, torch.cuda.current_device(), klass, kwargs)

        self.procs = [
            smp.Process(target=self._worker_loop, args=args) for _ in range(nprocs)
        ]

        for p in self.procs:
            p.start()

        self.open = True

    @torch.no_grad()
    @beartype
    def _inner_decode(
        self, encs: torch.Tensor, enc_lens: torch.Tensor
    ) -> List[Dict[int, FrameResponses]]:
        B, _, _ = encs.shape

        # Over-chunk as heterogeneous utterance lengths can
        # lead to uneven work distribution
        num_chunks = 2 * len(self.procs)
        utts_per_chunk = B // num_chunks

        if utts_per_chunk < self.min_batch:
            num_chunks = max(1, B // self.min_batch)

        ch_encs = torch.chunk(encs, num_chunks)
        ch_lens = torch.chunk(enc_lens, num_chunks)

        assert len(ch_encs) == len(ch_lens)

        for i, enc, enc_len in zip(count(), ch_encs, ch_lens):
            self.todo.put((i, (enc, enc_len)))

        chunked_outs = dict(self.result.get() for _ in range(num_chunks))

        for ret in chunked_outs.values():
            if isinstance(ret, Exception):
                self.close()
                raise ret

        return [r for _, responses in sorted(chunked_outs.items()) for r in responses]

    @staticmethod
    @beartype
    def _worker_loop(todo, result, device, klass: Type[RNNTDecoder], kwargs):
        """
        The receive-compute-send loop for each worker process.
        """
        torch.cuda.set_device(device)

        decoder = klass(**kwargs)

        while True:
            id, args = todo.get()

            if args is None:
                # Release any shared resources held
                kwargs.clear()
                del decoder
                gc.collect()
                return

            try:
                result.put((id, decoder._inner_decode(*args)))
            except Exception as e:
                result.put((id, e))

    @beartype
    def close(self):
        """
        Close the decoder pool and clean up resources.
        """

        if not self.open:
            return

        print("Shutting down decoder pool")

        for _ in self.procs:
            self.todo.put((None, None))

        for p in self.procs:
            p.join()

        self.todo.close()
        self.result.close()

        self.open = False

    def __del__(self):
        self.close()
