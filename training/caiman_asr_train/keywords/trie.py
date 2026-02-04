import math
from collections import deque
from dataclasses import dataclass, field

from beartype import beartype
from beartype.typing import Generic, Iterable, List, Optional, Tuple, TypeAlias, TypeVar

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
    """A goto edge in the trie"""

    idx: int


@beartype
@dataclass
class _Inst(Generic[T]):
    """An instruction in a FSM.

    Node in the goto trie (before AC failure links are added).

    Attributes:
        term: Set if the instruction is a terminal state of the FSM
        inst: Mapping of possible next instructions
    """

    term: float | None = None  # terminal => commit reward
    inst: dict[T, _Edge] = field(default_factory=dict)  # goto edges

    def __repr__(self):
        if exists(self.term):
            return f"{self.inst} term={self.term}"
        return f"{self.inst}"


@beartype
def _build(vocab: Iterable[Iterable[T]]) -> List[_Inst[T]]:
    """
    Compile a sequence of keywords (each an iterable of tokens) into a dense trie.

    This function does not mark terminal nodes; `_mark_terminals()` must be called
    after `_build()` to flag phrase ends.
    """
    fsa = [_Inst[T]()]  # root = 0

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
def _mark_terminals(
    trie: List[_Inst[T]], vocab: Iterable[Iterable[T]]
) -> List[_Inst[T]]:
    """
    Mark terminal nodes for each keyword (used to zero backoff after completion).
    """
    for word in vocab:
        idx = 0
        for tok in word:
            idx = trie[idx].inst[tok].idx
        # Just need a non-None sentinel; numeric value is unused by scoring
        assert trie[idx].term is None, "Duplicate keyword"
        trie[idx].term = 0.0
    return trie


@beartype
def _build_ac(
    trie: List[_Inst[T]],
    c_0: float = 0.3,
    beta: float = 0.9,
    context_score: float = 1.0,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Build an Aho–Corasick automaton with depth-shaped node potentials.

    Scoring:
        - Let
            shape(depth=0) = 0
            shape(depth=1) = c_0
            shape(depth>=2) = c_0 * beta + log(depth)
        - Node potential:
            S[node] = context_score * shape(depth[node])
        - Goto transition u -> v:
            edge_delta[v] = S[v] - S[fail[v]]
        - Failure transition s -> f, add backoff_w[s], where
            backoff_w[s] = 0.0            if s is terminal
                         = S[f] - S[s]    otherwise
    """

    def shape(depth: int) -> float:
        if depth == 0:
            return 0.0
        elif depth == 1:
            return c_0
        return c_0 * beta + math.log(depth)

    N = len(trie)
    fail = [0] * N
    S = [0.0] * N
    backoff_w = [0.0] * N
    edge_delta = [0.0] * N
    depth = [0] * N

    q = deque()

    # Root children: parent is root (0)
    for ch, e in trie[0].inst.items():
        v = e.idx
        fail[v] = 0
        depth[v] = 1
        S[v] = context_score * shape(depth[v])
        edge_delta[v] = S[v] - S[fail[v]]  # = S[v]
        backoff_w[v] = 0.0 if exists(trie[v].term) else (S[fail[v]] - S[v])
        q.append(v)

    # BFS over the rest
    while q:
        u = q.popleft()
        for ch, e in trie[u].inst.items():
            v = e.idx

            # Failure link for v based on u's fail chain
            f = fail[u]
            while ch not in trie[f].inst and f != 0:
                f = fail[f]
            f = trie[f].inst[ch].idx if ch in trie[f].inst else 0
            fail[v] = f

            # Depth, potentials, deltas
            depth[v] = depth[u] + 1
            S[v] = context_score * shape(depth[v])
            edge_delta[v] = S[v] - S[fail[v]]  # failure based delta
            backoff_w[v] = 0.0 if exists(trie[v].term) else (S[fail[v]] - S[v])

            q.append(v)

    return fail, S, backoff_w, edge_delta


@beartype
class Keywords(Generic[T]):
    # External decoding state is an integer AC node id (root=0).
    State: TypeAlias = int

    def __init__(
        self,
        vocab: Iterable[Iterable[T]],
        c_0: float = 0.3,
        beta: float = 0.9,
        context_score: float = 1.0,
    ):
        """
        Keyword biasing via AC automaton with depth-shaped potentials.

        Args:
            vocab: Iterable of token sequences (e.g. list of strings)
            c_0: Start-of-keyword boost (depth 1) [default 0.3].
            beta: Extra front-loading from depth 2; later steps taper via log(depth)
                  [default 0.9].
            context_score: Global bias strength [default 1.0].
        """
        v = list(vocab)
        assert len(set(v)) == len(v), "Duplicate keywords"

        # Build goto trie + weights
        self.inst = _mark_terminals(_build(v), v)

        # Build AC failure links and scoring tables
        self.fail, self.S, self.backoff_w, self.edge_delta = _build_ac(
            self.inst,
            c_0=c_0,
            beta=beta,
            context_score=context_score,
        )

    def __repr__(self):
        return "\n".join(map(str, self.inst))

    @classmethod
    def init(cls) -> State:
        """Start state (root)."""
        return 0

    def steps(self, toks: Iterable[T], state: State) -> Tuple[float, State]:
        """
        Compute cumulative score delta of calling step() on each token in `toks`.
        """
        acc = 0.0
        for tok in toks:
            tmp, state = self.step(tok, state)
            acc += tmp
        return acc, state

    def step(self, tok: T, state: State) -> Tuple[float, State]:
        """
        Advance the AC state by one character `tok`.
        Returns (score_delta, new_state).
        """
        # If string check for single char, probably meant to call steps()
        if isinstance(tok, str):
            assert len(tok) == 1, f"Did you mean to call steps() with: {tok}?"

        delta = 0.0

        # Follow failure links until we can consume tok (or we're at root)
        while tok not in self.inst[state].inst and state != 0:
            delta += self.backoff_w[state]  # undo the excess partial boost
            state = self.fail[state]

        # Consume goto edge if present; otherwise stay at root (no score change)
        e = self.inst[state].inst.get(tok, None)
        if e is not None:
            state = e.idx
            delta += self.edge_delta[
                state
            ]  # add forward gain relative to its fail state

        return delta, state
