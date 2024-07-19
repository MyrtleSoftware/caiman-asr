def test_all_submodule_imports():
    from caiman_asr_train import (
        args,
        data,
        evaluate,
        export,
        latency,
        log,
        rnnt,
        setup,
        unittesting,
    )


def test_train_import():
    from caiman_asr_train import train


def test_val_import():
    from caiman_asr_train import val


def test_val_multiple_import():
    from caiman_asr_train import val_multiple


def test_other_imports():
    from caiman_asr_train.data import generate_mel_stats
    from caiman_asr_train.export.hardware_ckpt import create_hardware_ckpt
