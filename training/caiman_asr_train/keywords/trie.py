from dataclasses import dataclass, field

from beartype import beartype
from beartype.typing import Generic, Iterable, Optional, TypeAlias, TypeVar

T = TypeVar("T")


@beartype
def exists(x: Optional[T]) -> bool:
    return x is not None


@beartype
def unwrap(x: Optional[T]) -> T:
    assert x is not None, f"Expected non-None value, got {x}"
    return x


@beartype
@dataclass
class _Edge:
    """An edge in a FSM's graph.

    Attributes:
        idx: Target index
        wgt: Weight of the edge
    """

    idx: int
    wgt: float = 0.0


@beartype
@dataclass
class _Inst(Generic[T]):
    """An instruction in a FSM.

    Attributes:
        term: Set if the instruction is a terminal state of the FSM
        inst: Mapping of possible next instructions
    """

    term: float | None = None
    inst: dict[T, _Edge] = field(default_factory=dict)

    def __repr__(self):
        if exists(self.term):
            return f"{self.inst} term={self.term}"
        return f"{self.inst}"


@beartype
def _build(vocab: Iterable[Iterable[T]]) -> list[_Inst[T]]:
    """
    Compile a Sequence of keywords into a dense trie structure.

    The weights of the edges are not set by this function nor
    are the term flags/weights.
    """
    fsa = [_Inst[T]()]

    for it in map(iter, vocab):
        idx = 0
        tok = next(it)

        while True:
            if tok is None:
                break

            if tok in fsa[idx].inst.keys():
                # Follow the prefix
                idx = fsa[idx].inst[tok].idx
                tok = next(it, None)
            else:
                # Point to new node
                n_idx = len(fsa)

                fsa.append(_Inst())
                fsa[idx].inst[tok] = _Edge(idx=n_idx)

                idx = n_idx
                tok = next(it, None)

    return fsa


@beartype
def _weight(
    trie: list[_Inst[T]], vocab: Iterable[tuple[Iterable[T], float]]
) -> list[_Inst[T]]:
    """
    Assign weights to the edges of the trie.

    These weights are the score delta of incurred moving along the edge.

    Returns:
        The updated trie
    """
    for word, w in vocab:
        idx = 0
        # How much has been accumulated along the real path
        acc_real = 0.0
        # How much has been accumulated along the theoretical path
        acc_imag = 0.0

        for tok in word:
            # The weight is the cumulative delta
            trie[idx].inst[tok].wgt = w + trie[idx].inst[tok].wgt

            acc_real += trie[idx].inst[tok].wgt
            acc_imag += w

            idx = trie[idx].inst[tok].idx

        assert trie[idx].term is None, "Duplicate keyword"
        # The theoretical value for terminating here
        trie[idx].term = acc_imag

    return trie


class Keywords(Generic[T]):
    # Decoding state: key = instruction index, value = accumulated weight
    State: TypeAlias = dict[int, float]

    @beartype
    def __init__(self, vocab: Iterable[tuple[Iterable[T], float]]):
        """
        A collection of keywords that can be used to score a decoding state.

        Args:
            vocab: pairs of keywords and their weights
        """

        v = next(zip(*vocab), [])

        assert len(set(v)) == len(v), "Duplicate keywords"

        self.inst = _weight(_build(v), vocab)

    def __repr__(self):
        return "\n".join(map(str, self.inst))

    @beartype
    @classmethod
    def init(cls) -> State:
        """
        Get a start of sentence state.
        """
        return {0: 0.0}

    @beartype
    def steps(self, toks: Iterable[T], state: State) -> tuple[float, State]:
        """
        Compute the cumulative score delta of step() called on each token in
        toks.

        Returns:
            cumulative score, final state
        """
        acc = 0.0
        for tok in toks:
            tmp, state = self.step(tok, state)
            acc += tmp
        return acc, state

    @beartype
    def step(self, tok: T, state: State) -> tuple[float, State]:
        """
        Compute the score delta of advancing state by one token tok.

        Returns:
           score delta, new state
        """

        # If string check for single char, probably meant to call steps()
        if isinstance(tok, str):
            assert len(tok) == 1, f"Did you mean to call steps() with: {tok}?"

        # Every step can be a new start
        assert 0 in state, "All states should have the initial inst"
        new_state = Keywords.init()

        delta = 0.0

        for i, acc in state.items():
            if exists(self.inst[i].term):
                # If this is a terminal node we commit the
                # accumulated score.
                acc = acc - unwrap(self.inst[i].term)

            edge = self.inst[i].inst.get(tok, None)

            if edge is None:
                # Thread dies, any uncommitted score is lost.
                delta -= acc
            else:
                # Thread continues in keyword
                new_state[edge.idx] = acc + edge.wgt
                delta += edge.wgt

        return delta, new_state
