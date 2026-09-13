"""C5 L10a 果实-种子传播：函数级对拍测试（Python 参考 vs Rust 下沉）。

覆盖 fruit_charge（植物蓄力→结果）和 eat_fruit（动物吃果实→能量转移）。
同输入下 Python 参考实现与 Rust sim_core 逐位一致。
"""
import pytest
import numpy as np
import sim_core

from simulation.genes import Gene

GENE_COUNT = 24


def _python_fruit_charge(flat, genes, fruit_charge, fruit_grid,
                          g19_idx, g8_idx, plant_threshold, charge_rate,
                          fruit_threshold, fruit_ratio):
    """Python 参考：复制 sphere_engine._step_fruit_charge 精确逻辑。"""
    fruit_charge = fruit_charge.copy()
    fruit_grid = fruit_grid.copy()
    P = len(flat)
    g19 = genes[:, g19_idx]
    g8 = genes[:, g8_idx]
    plant_mask = g19 >= plant_threshold
    if not plant_mask.any():
        return fruit_charge, fruit_grid
    fruit_charge[plant_mask] += charge_rate * g8[plant_mask]
    ripe = plant_mask & (fruit_charge >= fruit_threshold)
    if not ripe.any():
        return fruit_charge, fruit_grid
    ripe_flat = flat[ripe]
    ripe_charge = fruit_charge[ripe]
    fruit_add = ripe_charge * fruit_ratio
    np.add.at(fruit_grid, ripe_flat, fruit_add)
    fruit_charge[ripe] = 0.0
    return fruit_charge, fruit_grid


def _python_eat_fruit(flat, genes, energy, fruit_grid,
                       g19_idx, plant_threshold, eat_rate, digest_ratio):
    """Python 参考：复制 sphere_engine._step_eat_fruit 精确逻辑。"""
    energy = energy.copy()
    fruit_grid = fruit_grid.copy()
    P = len(flat)
    g19 = genes[:, g19_idx]
    animal_mask = g19 < plant_threshold
    if not animal_mask.any():
        return energy, fruit_grid
    animal_flat = flat[animal_mask]
    cell_fruit = fruit_grid[animal_flat]
    has_fruit = cell_fruit > 0
    if not has_fruit.any():
        return energy, fruit_grid
    eat_amount = cell_fruit[has_fruit] * eat_rate
    # 注意：布尔索引链式返回副本，必须用 flatnonzero 拿原始索引
    animal_idx = np.flatnonzero(animal_mask)
    eat_idx = animal_idx[has_fruit]
    energy[eat_idx] += eat_amount * digest_ratio
    eat_flat = animal_flat[has_fruit]
    np.add.at(fruit_grid, eat_flat, -eat_amount)
    fruit_grid[fruit_grid < 0] = 0.0
    return energy, fruit_grid


@pytest.fixture
def setup():
    rng = np.random.default_rng(42)
    n = 100
    n_cells = 7200
    flat = rng.integers(0, n_cells, size=n).astype(np.int64)
    genes = rng.uniform(0, 1, size=(n, GENE_COUNT)).astype(np.float64)
    # 确保有植物和动物
    genes[:30, int(Gene.ROOTING)] = rng.uniform(0.6, 1.0, size=30)  # 植物
    genes[30:60, int(Gene.ROOTING)] = rng.uniform(0.0, 0.4, size=30)  # 动物
    genes[60:, int(Gene.ROOTING)] = rng.uniform(0, 1, size=40)  # 混合
    genes[:, int(Gene.PHOTOSYNTHESIS)] = rng.uniform(0, 1, size=n)
    energy = rng.uniform(0, 300, size=n).astype(np.float64)
    fruit_charge = rng.uniform(0, 8, size=n).astype(np.float64)
    fruit_grid = rng.uniform(0, 20, size=n_cells).astype(np.float64)
    return {
        "flat": flat, "genes": genes, "energy": energy,
        "fruit_charge": fruit_charge, "fruit_grid": fruit_grid,
        "g19_idx": int(Gene.ROOTING), "g8_idx": int(Gene.PHOTOSYNTHESIS),
    }


class TestFruitChargeBitwise:
    """fruit_charge 植物蓄力→结果 逐位对拍。"""

    def test_fruit_charge_bitwise_equal(self, setup):
        s = setup
        py_fc, py_fg = _python_fruit_charge(
            s["flat"], s["genes"], s["fruit_charge"], s["fruit_grid"],
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        rs_fc = s["fruit_charge"].copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.fruit_charge(
            s["flat"], s["genes"].reshape(-1), GENE_COUNT,
            rs_fc, rs_fg,
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        np.testing.assert_array_equal(py_fc, rs_fc)
        np.testing.assert_array_equal(py_fg, rs_fg)

    def test_fruit_charge_no_plants(self, setup):
        """全动物（无植物）→ 无变化。"""
        s = setup
        genes = s["genes"].copy()
        genes[:, s["g19_idx"]] = 0.0  # 全动物
        py_fc, py_fg = _python_fruit_charge(
            s["flat"], genes, s["fruit_charge"], s["fruit_grid"],
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        rs_fc = s["fruit_charge"].copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.fruit_charge(
            s["flat"], genes.reshape(-1), GENE_COUNT,
            rs_fc, rs_fg,
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        np.testing.assert_array_equal(py_fc, rs_fc)
        np.testing.assert_array_equal(py_fg, rs_fg)
        # 无植物时 fruit_charge 应完全不变
        np.testing.assert_array_equal(py_fc, s["fruit_charge"])

    def test_fruit_charge_release_trigger(self, setup):
        """蓄力刚好达到阈值→触发释放。"""
        s = setup
        fc = s["fruit_charge"].copy()
        fc[:5] = 10.0  # 刚好等于阈值
        genes = s["genes"].copy()
        genes[:5, s["g19_idx"]] = 0.8  # 植物
        genes[:5, s["g8_idx"]] = 0.0  # 光合为0，蓄力不增加
        py_fc, py_fg = _python_fruit_charge(
            s["flat"], genes, fc, s["fruit_grid"],
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        rs_fc = fc.copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.fruit_charge(
            s["flat"], genes.reshape(-1), GENE_COUNT,
            rs_fc, rs_fg,
            s["g19_idx"], s["g8_idx"], 0.5, 0.1, 10.0, 0.8,
        )
        np.testing.assert_array_equal(py_fc, rs_fc)
        np.testing.assert_array_equal(py_fg, rs_fg)
        # 释放后蓄力清零
        assert (py_fc[:5] == 0.0).all()


class TestEatFruitBitwise:
    """eat_fruit 动物吃果实→能量转移 逐位对拍。"""

    def test_eat_fruit_bitwise_equal(self, setup):
        s = setup
        py_e, py_fg = _python_eat_fruit(
            s["flat"], s["genes"], s["energy"], s["fruit_grid"],
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        rs_e = s["energy"].copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.eat_fruit(
            s["flat"], s["genes"].reshape(-1), GENE_COUNT,
            rs_e, rs_fg,
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        np.testing.assert_array_equal(py_e, rs_e)
        np.testing.assert_array_equal(py_fg, rs_fg)

    def test_eat_fruit_no_animals(self, setup):
        """全植物（无动物）→ 无变化。"""
        s = setup
        genes = s["genes"].copy()
        genes[:, s["g19_idx"]] = 1.0  # 全植物
        py_e, py_fg = _python_eat_fruit(
            s["flat"], genes, s["energy"], s["fruit_grid"],
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        rs_e = s["energy"].copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.eat_fruit(
            s["flat"], genes.reshape(-1), GENE_COUNT,
            rs_e, rs_fg,
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        np.testing.assert_array_equal(py_e, rs_e)
        np.testing.assert_array_equal(py_fg, rs_fg)
        np.testing.assert_array_equal(py_e, s["energy"])

    def test_eat_fruit_empty_grid(self, setup):
        """果实场全空→无变化。"""
        s = setup
        empty_grid = np.zeros_like(s["fruit_grid"])
        py_e, py_fg = _python_eat_fruit(
            s["flat"], s["genes"], s["energy"], empty_grid,
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        rs_e = s["energy"].copy()
        rs_fg = empty_grid.copy()
        sim_core.eat_fruit(
            s["flat"], s["genes"].reshape(-1), GENE_COUNT,
            rs_e, rs_fg,
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        np.testing.assert_array_equal(py_e, rs_e)
        np.testing.assert_array_equal(py_fg, rs_fg)
        np.testing.assert_array_equal(py_e, s["energy"])

    def test_eat_fruit_energy_conservation(self, setup):
        """能量守恒：果实减少量 × digest_ratio = 能量增加量。"""
        s = setup
        fg_before = s["fruit_grid"].sum()
        e_before = s["energy"].sum()
        rs_e = s["energy"].copy()
        rs_fg = s["fruit_grid"].copy()
        sim_core.eat_fruit(
            s["flat"], s["genes"].reshape(-1), GENE_COUNT,
            rs_e, rs_fg,
            s["g19_idx"], 0.5, 0.05, 0.7,
        )
        fg_delta = fg_before - rs_fg.sum()
        e_delta = rs_e.sum() - e_before
        # e_delta ≈ fg_delta × digest_ratio（浮点误差范围内）
        np.testing.assert_allclose(e_delta, fg_delta * 0.7, rtol=1e-10, atol=1e-10)
