from caiman_asr_train.utils.iter import flat, lmap, lstarmap, lstarmap_zip, repeat_like


def test_flat():
    assert flat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flat([[1, 2], [3, 4]], _if=lambda x: x % 2 == 0) == [2, 4]
    assert flat([(1, 3), (2,)]) == [1, 3, 2]


def test_repeat_like():
    assert repeat_like([1, 2], _as=[[1, 2], [3, 4]]) == [1, 1, 2, 2]


def test_lmap():
    assert lmap(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]


def test_lstar_map():
    a = [1, 2, 3]
    b = [4, 5, 6]

    assert lstarmap(lambda x, y: x + y, zip(a, b)) == [5, 7, 9]

    assert lstarmap_zip(lambda x, y: x + y, a, b) == [5, 7, 9]
