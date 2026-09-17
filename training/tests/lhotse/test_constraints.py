from caiman_asr_train.lhotse.config import AreaBatch
from caiman_asr_train.lhotse.constraints import AreaConstraint, _find_smallest_bucket


def test_find_smallest_bucket():
    area_bins = [(1.0, 1), (1.0, 2), (2.0, 4)]

    bins = AreaConstraint(area_bins, AreaBatch(), strict=False).bins

    assert _find_smallest_bucket(bins, 0.5, 1, strict=True) == 0
    assert _find_smallest_bucket(bins, 1.0, 1, strict=True) == 0
    assert _find_smallest_bucket(bins, 1.0, 2, strict=True) == 1
    assert _find_smallest_bucket(bins, 1.5, 3, strict=True) == 2

    assert _find_smallest_bucket(bins, 0.5, 3, strict=True) is None
    assert _find_smallest_bucket(bins, 0.5, 3, strict=False) == 2
