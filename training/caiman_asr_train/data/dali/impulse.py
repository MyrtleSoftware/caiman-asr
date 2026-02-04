import os
from hashlib import sha256
from pathlib import Path

import torchaudio
from audiomentations import ApplyImpulseResponse
from beartype import beartype

from caiman_asr_train.train_utils.distributed import maybe_cached


@beartype
class ImpulseAug:
    def __init__(
        self,
        sample_rate: int,
        prob: float,
        impulse_dir: None | str | Path,
        impulse_cache_dir: None | str | Path,
        lru_cache_size=512,
    ):
        """
        Applies impulse response augmentations to an audio signal.

        Args:
            sample_rate: The sample rate of the audio signal.
            prob: The probability of applying the augmentation.
            impulse_dir: The directory containing impulse responses.
            impulse_cache_dir: The directory to cache resampled impulse responses.
            lru_cache_size: The size of audiomentations LRU cache.
        """
        self.sample_rate = sample_rate
        self.prob = prob

        assert 0 < sample_rate
        assert 0 <= prob <= 1

        if self.prob == 0.0:
            return

        assert impulse_dir is not None, "Impulse_dir must be provided"
        assert impulse_cache_dir is not None, "Impulse_cache_dir must be provided"

        impulse_dir = os.path.expanduser(impulse_dir)
        impulse_cache_dir = os.path.expanduser(impulse_cache_dir)

        # Get all files-names in dir_dir
        wavs = list(Path(impulse_dir).rglob("*.wav"))

        assert wavs, f"No impulse responses found in {impulse_dir}"

        # Build a hash
        stable = "".join(str(x) for x in sorted(wavs))
        stable = stable.encode("utf-8")
        stable = sha256(stable).hexdigest()

        cache_dir = Path(impulse_cache_dir) / stable

        def populate(cache_dir):
            os.makedirs(cache_dir)
            for d in wavs:
                # Resample with torch audio
                try:
                    wv, sr = torchaudio.load(d)
                except Exception as e:
                    print(f"WARNING: failed to load {d}: {e}")
                    continue

                out = cache_dir / d.name
                wv = torchaudio.functional.resample(wv, sr, sample_rate)
                torchaudio.save(out, wv, sample_rate)

        maybe_cached(populate, cache_dir)

        self.fn = ApplyImpulseResponse(
            ir_path=cache_dir,
            p=self.prob,
            lru_cache_size=lru_cache_size,
        )

    def __call__(self, wv):
        return self.fn(wv, sample_rate=self.sample_rate)
