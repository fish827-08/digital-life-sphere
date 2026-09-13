"""田字格信号场单元测试（L2）。"""
import numpy as np
import pytest

from world.signal_field import SignalField
from world.sphere_world import SphereWorld


@pytest.fixture
def world():
    return SphereWorld(rows=60, cols=120)


@pytest.fixture
def sf(world):
    return SignalField(world, duration=5)


def test_initial_all_zero(sf):
    assert sf.active_count() == 0
    assert sf.total_marks() == 0
    assert (sf._marks == 0).all()


def test_write_read_single(sf):
    sf.write(100, 0b1010)  # 10
    assert sf.read(100) == 10
    assert sf.active_count() == 1
    assert sf.total_marks() == 2  # 2 个 1


def test_write_overwrites(sf):
    sf.write(100, 0b1111)
    sf.write(100, 0b0001)
    assert sf.read(100) == 1


def test_write_zero_clears(sf):
    sf.write(100, 0b1111)
    sf.write(100, 0)
    assert sf.read(100) == 0
    assert sf.active_count() == 0


def test_pattern_truncated_to_4_bits(sf):
    sf.write(100, 0xFF)  # 255，应截断为 0x0F=15
    assert sf.read(100) == 15


def test_tick_expires_after_duration(sf):
    sf.write(100, 0b1010)
    for _ in range(5):
        assert sf.read(100) == 0b1010
        sf.tick()
    assert sf.read(100) == 0
    assert sf.active_count() == 0


def test_tick_refreshes_age_on_rewrite(sf):
    sf.write(100, 0b1010)
    sf.tick()
    sf.tick()
    sf.write(100, 0b1010)  # 重写，年龄重置
    for _ in range(5):
        assert sf.read(100) == 0b1010
        sf.tick()
    assert sf.read(100) == 0


def test_write_many_read_many(sf, world):
    cells = np.array([10, 20, 30], dtype=np.int64)
    patterns = np.array([1, 2, 3], dtype=np.uint8)
    sf.write_many(cells, patterns)
    np.testing.assert_array_equal(sf.read_many(cells), [1, 2, 3])
    assert sf.active_count() == 3


def test_clear(sf):
    sf.write(100, 0b1111)
    sf.clear(100)
    assert sf.read(100) == 0


def test_snapshot_shape(sf, world):
    snap = sf.snapshot()
    assert snap.shape == (world.rows, world.cols)


def test_multiple_cells_independent(sf):
    sf.write(100, 0b0001)
    sf.write(200, 0b1110)
    assert sf.read(100) == 1
    assert sf.read(200) == 14
    assert sf.active_count() == 2
    assert sf.total_marks() == 1 + 3  # 1 + 3 = 4
