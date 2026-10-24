import numpy as np

from lobsurv.confirm import bootstrap_pooled_diff, part_days, rolling_from_raw
from lobsurv.scoring import score


def test_part_days_splits_val_and_test_at_their_boundary():
    offsets = {"train": 0, "val": 1000, "test": 1600}
    segments = [0, 500, 1000, 1300, 1600, 1900]                 # two days per split
    val = part_days(offsets, segments, 2200, "val", context=10, horizon=100)
    test = part_days(offsets, segments, 2200, "test", context=10, horizon=100)
    np.testing.assert_array_equal(val[0], [1010, 1110])          # 1300 - 100 = 1200 is excluded
    np.testing.assert_array_equal(val[1], [1310, 1410])          # ends before the test split starts
    np.testing.assert_array_equal(test[0], [1610, 1710])
    np.testing.assert_array_equal(test[1], [1910, 2010])         # ends before the stream does
    for days, lo, hi in ((val, 1000, 1600), (test, 1600, 2200)):
        for d in days:
            assert d.min() - 10 >= lo and d.max() + 100 <= hi


def test_part_days_matches_the_validation_only_layout():
    offsets = {"train": 0, "val": 1000}
    segments = [0, 500, 1000, 1300]
    days = part_days(offsets, segments, 1600, "val", context=10, horizon=100)
    expected = [np.arange(a + 10, b - 100, 100) for a, b in ((1000, 1300), (1300, 1600))]
    for d, e in zip(days, expected):
        np.testing.assert_array_equal(d, e)


def test_rolling_from_raw_calibrates_each_day_on_the_day_before():
    rng = np.random.default_rng(0)
    cal_p = rng.random((50, 3))
    eval_p = rng.random((40, 2, 3))
    day = np.repeat([0, 1], 20)
    p = rolling_from_raw(cal_p, eval_p, day)
    sc, cal0, cal1 = score(eval_p), score(cal_p), score(eval_p)[:20, 0]
    for i in (0, 25):
        cal = cal0 if day[i] == 0 else cal1
        assert np.isclose(p[i, 1], (1 + np.sum(cal >= sc[i, 1])) / (len(cal) + 1))


def test_bootstrap_interval_covers_a_known_difference():
    rng = np.random.default_rng(1)
    diffs = [rng.normal(0.2, 1, 3000), rng.normal(0.0, 1, 3000)]
    est, lo, hi = bootstrap_pooled_diff(diffs, reps=500)
    assert lo < est < hi and lo < 0.1 < hi and lo > 0.0
    est0, lo0, hi0 = bootstrap_pooled_diff([np.zeros(10), np.zeros(10)], reps=50)
    assert est0 == lo0 == hi0 == 0.0
