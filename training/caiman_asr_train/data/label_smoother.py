import json
import os
from hashlib import sha256
from multiprocessing import Pool
from pathlib import Path

from beartype import beartype
from Levenshtein import distance as norm
from numpy import random
from psutil import cpu_count
from tqdm import tqdm

from caiman_asr_train.train_utils.distributed import maybe_cached


@beartype
class _LevenshteinFunctor:
    def __init__(self, words: list[str], levenshtein_threshold: float):
        """
        The call operator of this functor will return a word from `words`
        that is within `levenshtein_threshold` of the input word.

        This is needed (instead of a lambda) because it must be picklable.
        """
        self.words = words
        self.levenshtein_threshold = levenshtein_threshold

    def near(self, a: str, b: str) -> bool:
        cut = min(len(a), len(b))
        cut = self.levenshtein_threshold * cut
        cut = int(cut)
        cut = max(1, cut)
        return 0 < norm(a, b, score_cutoff=cut) <= cut

    def __call__(self, other: str) -> list[str]:
        return [w for w in self.words if self.near(w, other)]


class LabelSmoother:
    @beartype
    def __init__(
        self,
        dict_path: None | str | Path,
        dict_cache_dir: None | str | Path,
        charset: set[str],
        levenshtein_threshold: float,
        prob_smooth_word: float,
    ):
        """
        Args:
            dict_path: Path to unix dictionary file.
            dict_cache_dir: Directory to store cached word list.
            char_set: Set of characters that the words in the dictionary can contain.
            levenshtein_threshold: Minimum levenshtein ratio for two words to be
                considered similar.
            prob_smooth_word: Smoothing factor for label smoothing.
        """

        assert 0 <= prob_smooth_word <= 1
        assert 0 <= levenshtein_threshold <= 1

        self.prob_smooth_word = prob_smooth_word

        if prob_smooth_word == 0.0:
            return

        assert dict_path is not None, "Dictionary path must be provided."
        assert dict_cache_dir is not None, "Cache directory must be provided."

        with open(dict_path, "r") as f:
            words = set(f.read().splitlines())

        if charset == set(x.lower() for x in charset):
            words = {w.lower() for w in words}

        words = sorted({w for w in words if all(c in charset for c in w)})

        stable = str(levenshtein_threshold) + "i".join(words)
        stable = stable.encode("utf-8")
        stable = sha256(stable).hexdigest()

        dict_cache_dir = Path(dict_cache_dir) / f"{stable}/"
        dict_cache_dir = dict_cache_dir.expanduser()

        cache_file = dict_cache_dir / "words.json"

        def populate(_):
            # Must be picklable for multiprocessing.
            n = cpu_count(logical=False)
            fn = _LevenshteinFunctor(words, levenshtein_threshold)
            bar = tqdm(words, desc=f"Building word cache ({n})")

            with Pool(n) as p:
                sim_words = dict(zip(bar, p.imap(fn, words)))

            os.makedirs(dict_cache_dir)
            with open(cache_file, "w") as f:
                json.dump(sim_words, f, indent=2)

        maybe_cached(populate, dict_cache_dir)

        with open(cache_file, "r") as f:
            self.sim_words = json.load(f)

        print(f"Loaded {len(self.sim_words)} words from {dict_path}")

    @beartype
    def __call__(self, word: str) -> str:
        if (
            self.prob_smooth_word == 0.0
            or random.random_sample() > self.prob_smooth_word
        ):
            return word

        opts = self.sim_words.get(word, [])

        if not opts:
            return word

        return random.choice(opts)
