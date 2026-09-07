"""快照机制测试（v2，适配 L10a + 统计数组）。

核心：save_snapshot → load_snapshot → 继续跑，结果与"不保存连续跑"逐位一致。
覆盖：基本保存/恢复、续跑逐位一致、版本/配置/gene_count 校验、L10a 数组、RNG 状态。
"""
import os
import tempfile

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from simulation.genes import Gene


def _make_config(seed=42, fruit_enabled=False):
    cfg = SimConfig(seed=seed)
    cfg.world.rows = 60
    cfg.world.cols = 120
    cfg.population.initial_count = 100
    cfg.resources.distribution = "patchy"
    cfg.simulation.use_sim_core = True
    cfg.fruit.enabled = fruit_enabled
    return cfg


class TestSnapshotBasic:
    """基本保存/恢复功能。"""

    def test_save_and_load_basic(self):
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(200):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            e2 = SphereEngine.load_snapshot(path)
            P = len(e1._id)
            assert P == len(e2._id)
            assert e1._tick == e2._tick
            assert e1._next_id == e2._next_id
            assert e1._max_generation == e2._max_generation
            np.testing.assert_array_equal(e1._id[:P], e2._id[:P])
            np.testing.assert_array_equal(e1._flat[:P], e2._flat[:P])
            np.testing.assert_array_equal(e1._energy[:P], e2._energy[:P])
            np.testing.assert_array_equal(e1._genes[:P], e2._genes[:P])
        finally:
            os.unlink(path)

    def test_l10a_arrays_saved(self):
        """L10a 果实场数组必须保存/恢复。"""
        cfg = _make_config(fruit_enabled=True)
        e1 = SphereEngine(cfg)
        for _ in range(500):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            np.testing.assert_array_equal(e1._fruit_grid, e2._fruit_grid)
            P = len(e1._id)
            np.testing.assert_array_equal(e1._fruit_charge[:P], e2._fruit_charge[:P])
            np.testing.assert_array_equal(e1._seed_carried[:P], e2._seed_carried[:P])
        finally:
            os.unlink(path)

    def test_run_stats_saved(self):
        """运行统计（_run_born/_run_died/_run_deaths）必须保存/恢复。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(500):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            assert e1._run_born == e2._run_born
            assert e1._run_died == e2._run_died
            assert dict(e1._run_deaths) == dict(e2._run_deaths)
        finally:
            os.unlink(path)

    def test_rng_state_saved(self):
        """RNG 状态必须保存/恢复（可复现的关键）。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(100):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            assert e1.rng.bit_generator.state == e2.rng.bit_generator.state
        finally:
            os.unlink(path)

    def test_resource_state_saved(self):
        """资源场状态必须保存/恢复。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(300):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            np.testing.assert_array_equal(e1.resources._grid, e2.resources._grid)
            np.testing.assert_array_equal(e1.resources._capacity, e2.resources._capacity)
            np.testing.assert_array_equal(e1.resources._patch_mask, e2.resources._patch_mask)
        finally:
            os.unlink(path)


class TestSnapshotReproducibility:
    """核心：恢复后续跑与不保存连续跑逐位一致。"""

    def test_continue_after_load_bitwise_equal(self):
        """保存→恢复→跑100tick，与不保存连续跑100tick逐位一致。"""
        cfg = _make_config()
        e_continuous = SphereEngine(cfg)
        e_snapshot = SphereEngine(cfg)

        # 两个引擎用相同配置，前 300 tick 应该完全一致
        for _ in range(300):
            e_continuous.step()
            e_snapshot.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snapshot.save_snapshot(path)
            e_loaded = SphereEngine.load_snapshot(path)

            # 继续跑 100 tick
            for _ in range(100):
                e_continuous.step()
                e_loaded.step()

            P = len(e_continuous._id)
            assert P == len(e_loaded._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._flat[:P], e_loaded._flat[:P])
            np.testing.assert_array_equal(e_continuous._genes[:P], e_loaded._genes[:P])
            np.testing.assert_array_equal(e_continuous._age[:P], e_loaded._age[:P])
            np.testing.assert_array_equal(e_continuous._valence[:P], e_loaded._valence[:P])
            np.testing.assert_array_equal(e_continuous._trust[:P], e_loaded._trust[:P])
        finally:
            os.unlink(path)

    def test_continue_with_l10a_bitwise_equal(self):
        """L10a 开启时，恢复后续跑逐位一致。"""
        cfg = _make_config(fruit_enabled=True)
        e_continuous = SphereEngine(cfg)
        e_snapshot = SphereEngine(cfg)

        for _ in range(300):
            e_continuous.step()
            e_snapshot.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snapshot.save_snapshot(path)
            e_loaded = SphereEngine.load_snapshot(path)

            for _ in range(100):
                e_continuous.step()
                e_loaded.step()

            P = len(e_continuous._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._fruit_grid, e_loaded._fruit_grid)
            np.testing.assert_array_equal(e_continuous._fruit_charge[:P], e_loaded._fruit_charge[:P])
        finally:
            os.unlink(path)

    def test_multiple_save_load_cycles(self):
        """多次保存/恢复循环后仍逐位一致。"""
        cfg = _make_config()
        e_continuous = SphereEngine(cfg)
        e_loaded = SphereEngine(cfg)

        for _ in range(100):
            e_continuous.step()
            e_loaded.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            # 3 次保存/恢复循环，每次循环后跑 50 tick
            for cycle in range(3):
                e_loaded.save_snapshot(path)
                e_loaded = SphereEngine.load_snapshot(path)
                for _ in range(50):
                    e_continuous.step()
                    e_loaded.step()

            P = len(e_continuous._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._flat[:P], e_loaded._flat[:P])
        finally:
            os.unlink(path)


class TestSnapshotValidation:
    """版本/配置/gene_count 校验。"""

    def test_version_mismatch_raises(self):
        """快照版本不兼容时必须报错。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            # 手动修改版本号
            data = dict(np.load(path, allow_pickle=True))
            data["snapshot_version"] = np.array(999)
            np.savez_compressed(path, **data)

            with pytest.raises(ValueError, match="版本"):
                SphereEngine.load_snapshot(path)
        finally:
            os.unlink(path)

    def test_config_fingerprint_mismatch_raises(self):
        """提供 config 时，指纹不匹配必须报错。"""
        cfg1 = _make_config(seed=42)
        cfg2 = _make_config(seed=999)  # 不同 seed → 不同指纹
        e1 = SphereEngine(cfg1)
        e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            with pytest.raises(ValueError, match="指纹"):
                SphereEngine.load_snapshot(path, config=cfg2)
        finally:
            os.unlink(path)

    def test_load_without_config_recovers_config(self):
        """不提供 config 时，从快照中恢复配置。"""
        cfg = _make_config(seed=42)
        e1 = SphereEngine(cfg)
        for _ in range(50):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)  # config=None
            assert e2.config.seed == 42
            assert e2._tick == e1._tick
        finally:
            os.unlink(path)

    def test_snapshot_file_is_compressed(self):
        """快照文件使用 npz 压缩，大小应合理。"""
        cfg = _make_config()
        cfg.population.initial_count = 500
        e1 = SphereEngine(cfg)
        for _ in range(100):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            size = os.path.getsize(path)
            # N=500, expectation (500,120) = 60000 floats = 480KB，压缩后应 < 1MB
            assert size < 5 * 1024 * 1024, f"快照过大: {size} bytes"
            assert size > 1024, f"快照过小: {size} bytes"
        finally:
            os.unlink(path)
