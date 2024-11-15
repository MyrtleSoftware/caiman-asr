from caiman_asr_train.utils.iter import flat, repeat_like


def test_flat():
    assert flat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flat([[1, 2], [3, 4]], _if=lambda x: x % 2 == 0) == [2, 4]
    assert flat([(1, 3), (2,)]) == [1, 3, 2]


def test_repeat_like():
    assert repeat_like([1, 2], _as=[[1, 2], [3, 4]]) == [1, 1, 2, 2]
