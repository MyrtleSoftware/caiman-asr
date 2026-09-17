"""
Tests for the kernel-agnostic autotune sweep: the structural TIR hash, the
per-config bench cache (incremental re-bench on set changes), and
``autotune_select``'s infeasibility contract.

All CUDA-free (pure functions / fakes), so it runs anywhere. These tests pin the
generic ``caiman_asr_train.autotune`` package alone and must not import
``caiman_asr_train.tilelstm``.
"""

from caiman_asr_train.autotune import sweep


def _use_tmp_bench_cache(monkeypatch, tmp_path):
    """Isolate the autotuner's on-disk bench cache in ``tmp_path`` with caching ENABLED
    (clear the two disable-cache env knobs)."""
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CAIMAN_AUTOTUNE_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)


def test_tir_hash_is_structural():
    import tvm
    from tvm import tirx as tir  # tilelang's bundled TVM names the TIR namespace `tirx`

    def loop(
        extent, var_name
    ):  # a bare `for <var_name> in range(extent): pass` PrimFunc
        i = tir.Var(var_name, "int32")
        body = tir.For(
            i,
            tir.const(0, "int32"),
            tir.const(extent, "int32"),
            tir.ForKind.SERIAL,
            tir.Evaluate(tir.const(0, "int32")),
        )
        return tir.PrimFunc([], body)

    # Name-INVARIANT: the key tracks the lowered structure, not the var names (which drift
    # with TVM's global counter as other configs elaborate), so a warm re-tune still hits.
    assert sweep._tir_hash(loop(4, "i")) == sweep._tir_hash(loop(4, "j"))
    # A real structural change (the loop extent) yields a fresh key -> automatic re-bench.
    assert sweep._tir_hash(loop(4, "i")) != sweep._tir_hash(loop(8, "i"))

    def reads_free(name):
        # A PrimFunc that READS a var it neither binds nor takes as a param -- i.e. a FREE
        # variable, a distinct object on each call.
        v = tir.Var(name, "int32")
        return tir.PrimFunc([], tir.Evaluate(v))

    # FREE vars must hash by STRUCTURE, not object identity -- kernels that mint fresh
    # dynamic-shape symbols on every elaboration would otherwise get a different key in
    # every process (identity-keying is the DEFAULT of structural_hash) => a fresh
    # process re-tune misses every entry. _tir_hash passes map_free_vars=True to defeat
    # this; this is the regression guard. The default tells the two apart (proving they
    # are genuinely identity-keyed free vars); _tir_hash must NOT.
    f1, f2 = reads_free("n"), reads_free("n")
    assert tvm.ir.structural_hash(f1) != tvm.ir.structural_hash(
        f2
    )  # identity-keyed default
    assert sweep._tir_hash(f1) == sweep._tir_hash(
        f2
    )  # map_free_vars=True -> structure-keyed (the fix)


def test_bench_cache_roundtrip(tmp_path, monkeypatch):
    _use_tmp_bench_cache(monkeypatch, tmp_path)
    bench_dir = sweep._bench_cache_dir(
        ("device", 256)
    )  # cache_key = (device, tune_batch)
    cfg = {"block_M": 64}
    ok, fail = "tirhash_ok", "tirhash_fail"  # bench entries are keyed by the TIR hash
    assert sweep._load_bench_entry(bench_dir, ok) is None  # not yet benched
    sweep._store_bench_entry(bench_dir, ok, cfg, 0.42)
    sweep._store_bench_entry(bench_dir, fail, cfg, None)  # a cached failure
    assert sweep._load_bench_entry(bench_dir, ok)["latency_ms"] == 0.42
    assert sweep._load_bench_entry(bench_dir, fail)["latency_ms"] is None


def test_autotune_select_benches_only_the_delta(tmp_path, monkeypatch):
    # The per-config bench cache makes a set change re-bench only the DELTA: adding one
    # config triggers exactly one more par_compile; an unchanged set triggers none.
    _use_tmp_bench_cache(monkeypatch, tmp_path)

    funcs_per_call = []

    class _Kernel:
        def __call__(self, *a, **k):
            return None

    class _Factory:
        out_idx = None
        execution_backend = "tvm_ffi"
        target = target_host = None
        verbose = False
        pass_configs: dict = {}
        compile_flags: list = []

        def get_tir(self, **cfg):
            return repr(sorted(cfg.items()))  # stand-in TIR; distinct per config

        def compile(self, **cfg):
            return _Kernel()

    def fake_par_compile(funcs, **kw):
        funcs_per_call.append(list(funcs))
        return [_Kernel() for _ in funcs]

    monkeypatch.setattr(sweep, "_par_compile", fake_par_compile)
    monkeypatch.setattr(sweep, "lean_bench", lambda thunk, n_warmup=3, n_repeat=10: 1.0)
    # structural_hash needs a real PrimFunc; here the stand-in TIR is its own key.
    monkeypatch.setattr(sweep, "_tir_hash", lambda tir: tir)

    def run(cfgs):
        return sweep.autotune_select(
            factory=_Factory(),
            configs=cfgs,
            make_thunk=lambda k, c: (lambda: None),
            validate=lambda k, c: None,
            finalize=lambda k, c: k,
            cache_key=("delta_test",),
            label="fwd_c",
            H=1024,
            tune_batch=256,
        )

    a, b = {"block_M": 64}, {"block_M": 32}
    run([a])
    assert [len(f) for f in funcs_per_call] == [1]  # benched A
    run([a, b])
    assert [len(f) for f in funcs_per_call] == [1, 1]  # benched only B (A cached)
    run([a, b])
    assert [len(f) for f in funcs_per_call] == [1, 1]  # all cached -> no new compile


def test_autotune_select_returns_none_when_no_config_feasible(monkeypatch):
    """
    When every candidate fails to compile/launch, autotune_select returns None (not
    raise) so the caller skips the backend -- infeasibility is signalled by None, there is
    no 'no feasible' exception any more. CUDA-free: par_compile is faked to fail all.
    """
    monkeypatch.setenv("CAIMAN_AUTOTUNE_DISABLE_CACHE", "1")
    monkeypatch.setattr(sweep, "_par_compile", lambda funcs, **kw: [None] * len(funcs))
    # structural_hash needs a real PrimFunc; here the stand-in TIR is its own key.
    monkeypatch.setattr(sweep, "_tir_hash", lambda tir: tir)

    class _Factory:
        out_idx = None
        execution_backend = "tvm_ffi"
        target = target_host = None
        verbose = False
        pass_configs: dict = {}
        compile_flags: list = []

        def get_tir(self, **cfg):
            return repr(sorted(cfg.items()))  # stand-in TIR; distinct per config

    result = sweep.autotune_select(
        factory=_Factory(),
        configs=[{"block_M": 64}, {"block_M": 32}],
        make_thunk=lambda k, c: lambda: None,
        validate=lambda k, c: None,
        finalize=lambda k, c: k,
        cache_key=("test_no_feasible",),
        label="fwd_c",
        H=1024,
        tune_batch=256,
    )
    assert result is None
