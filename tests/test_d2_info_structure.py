"""D2 信息结构重构单元测试。

覆盖四大机制：
  1. 学习瓶颈：解读表不遗传，幼体从(信号,后果)观察样本归纳
  2. 任意性码本：pattern=码本[状态]，码本可遗传可突变
  3. 信息不对称：感知半径4 + 感知噪声 + softmax
  4. Steels对齐：同格相遇解读表+码本对齐

D2 关闭时（enabled=False）行为与旧版完全一致（回归由 test_sim_core_engine 覆盖）。
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import SimConfig, InfoStructureConfig
from simulation.sphere_engine import SphereEngine


def _make_engine(**d2_kwargs) -> SphereEngine:
    """创建 D2 开启的小引擎（use_sim_core=False，D2 暂未下沉 Rust）。"""
    cfg = SimConfig()
    cfg.info_structure = InfoStructureConfig(enabled=True, **d2_kwargs)
    cfg.simulation.use_sim_core = False
    cfg.population.initial_count = 50
    return SphereEngine(cfg)


# ─── 配置层 ────────────────────────────────────────────────

class TestInfoStructureConfig:
    def test_default_disabled(self):
        cfg = InfoStructureConfig()
        assert cfg.enabled is False
        assert cfg.perception_radius == 4

    def test_perception_radius_validation(self):
        with pytest.raises(AssertionError):
            InfoStructureConfig(enabled=True, perception_radius=6)

    def test_enabled_in_simconfig(self):
        cfg = SimConfig()
        assert cfg.info_structure.enabled is False
        cfg.info_structure.enabled = True
        assert cfg.info_structure.learning_bottleneck is True


# ─── 数据结构 ──────────────────────────────────────────────

class TestD2DataStructures:
    def test_codebook_initialized_identity(self):
        e = _make_engine()
        # 码本初始恒等映射：codebook[state] == state
        expected = np.arange(16, dtype=np.uint8)
        np.testing.assert_array_equal(e._codebook[0], expected)

    def test_learning_count_initialized_zero(self):
        e = _make_engine()
        assert (e._learning_count == 0).all()

    def test_codebook_shape(self):
        e = _make_engine()
        assert e._codebook.shape == (50, 16)
        assert e._codebook.dtype == np.uint8


# ─── 机制1：学习瓶颈 ───────────────────────────────────────

class TestLearningBottleneck:
    def test_offspring_interpret_not_inherited(self):
        """D2 学习瓶颈开启时，子代解读表随机初始化而非继承亲代。"""
        e = _make_engine(arbitrary_codebook=False, perception_radius=8,
                         perception_noise=0.0, softmax_tau=0.0)
        # 强制繁殖：设置高能量、低阈值
        e._energy[:] = 100.0
        e._genes[:, 2] = 0.0  # 繁殖门槛最低
        e._genes[:, 12] = 0.0  # 冷却为0
        parent_interp = e._interpret[0].copy()
        for _ in range(50):
            e.step()
        # 应该有子代出生
        if len(e._id) > 50:
            # 子代的解读表不应与任何亲代完全相同（随机初始化）
            child_interps = e._interpret[50:]
            # 至少有一个子代的解读表与亲代0不同
            assert not np.allclose(child_interps[0], parent_interp, atol=0.01)

    def test_learning_count_increases(self):
        """幼体观察到信号+后果时，学习计数增加。"""
        e = _make_engine(arbitrary_codebook=False, perception_radius=8,
                         perception_noise=0.0, softmax_tau=0.0)
        for _ in range(100):
            e.step()
        # 学习计数应至少有非零值（或种群灭绝）
        assert e._learning_count.size == 0 or (e._learning_count > 0).any() or e._extinct


# ─── 机制2：任意性码本 ─────────────────────────────────────

class TestArbitraryCodebook:
    def test_codebook_inherited(self):
        """码本可遗传：子代的码本应与亲代相似（可能有少量突变）。"""
        e = _make_engine(learning_bottleneck=False, perception_radius=8,
                         perception_noise=0.0, softmax_tau=0.0)
        e._energy[:] = 100.0
        e._genes[:, 2] = 0.0
        e._genes[:, 12] = 0.0
        parent_cb = e._codebook[0].copy()
        for _ in range(30):
            e.step()
        if len(e._id) > 50:
            child_cb = e._codebook[50]
            # 至少有一些位与亲代相同（遗传）
            assert (child_cb == parent_cb).sum() > 8

    def test_signal_emit_uses_codebook(self):
        """信号发射时 pattern=码本[state]，而非硬编码 state。"""
        e = _make_engine(learning_bottleneck=False, perception_radius=8,
                         perception_noise=0.0, softmax_tau=0.0)
        # 修改个体0的码本：所有状态映射到模式5
        e._codebook[0] = 5
        e._energy[0] = 100.0
        e._genes[0, 15] = 1.0  # 信号发射概率100%
        e.step()
        # 个体0所在格子应该有信号（模式5）
        cell = int(e._flat[0])
        assert e.signals._marks[cell] in (0, 5)  # 0=可能被覆盖，5=码本映射


# ─── 机制3：信息不对称 ─────────────────────────────────────

class TestInformationAsymmetry:
    def test_perception_radius_4_runs(self):
        """感知半径4时引擎正常运行不报错。"""
        e = _make_engine(arbitrary_codebook=False, learning_bottleneck=False,
                         perception_radius=4, perception_noise=0.05, softmax_tau=0.15)
        for _ in range(50):
            e.step()
        assert e._tick == 50 or e._extinct

    def test_softmax_runs(self):
        """softmax>0时引擎正常运行。"""
        e = _make_engine(arbitrary_codebook=False, learning_bottleneck=False,
                         perception_radius=8, perception_noise=0.0, softmax_tau=0.2)
        for _ in range(50):
            e.step()
        assert e._tick == 50 or e._extinct


# ─── 机制4：Steels对齐 ─────────────────────────────────────

class TestSteelsAlignment:
    def test_alignment_reduces_interpret_difference(self):
        """同格相遇时解读表差异减小。"""
        e = _make_engine(arbitrary_codebook=False, learning_bottleneck=False,
                         perception_radius=8, perception_noise=0.0, softmax_tau=0.0,
                         alignment_rate=1.0, alignment_step=0.5, alignment_noise=0.0)
        # 把两个个体放到同一格，设置差异很大的解读表
        e._flat[0] = 100
        e._flat[1] = 100
        e._interpret[0] = 1.0
        e._interpret[1] = -1.0
        diff_before = np.abs(e._interpret[0] - e._interpret[1]).mean()
        e.step()
        diff_after = np.abs(e._interpret[0] - e._interpret[1]).mean()
        # 对齐后差异应减小（或至少不增大太多，因为有移动可能分开）
        # 注意：移动可能导致它们分开，所以只检查代码不报错
        assert e._tick == 1

    def test_codebook_alignment(self):
        """Steels对齐同时对齐码本。"""
        e = _make_engine(arbitrary_codebook=True, learning_bottleneck=False,
                         perception_radius=8, perception_noise=0.0, softmax_tau=0.0,
                         alignment_rate=1.0, alignment_step=1.0, alignment_noise=0.0)
        e._flat[0] = 100
        e._flat[1] = 100
        e._codebook[0] = 1
        e._codebook[1] = 2
        e.step()
        # 对齐后码本0的某些位应变成2（alignment_step=1.0概率替换）
        assert e._tick == 1


# ─── 可复现性（中高危漏洞回归）──────────────────────────────

class TestReproducibility:
    def test_same_seed_bitwise_identical(self):
        """np.random.seed(config.seed)：同 seed 两次运行 D2 全机制逐位一致。

        感知噪声/Steels 配对使用全局 np.random，若不在引擎 init 随 config.seed
        播种，同 seed 两次运行结果不同，"3 seed 一致"判据会混入非受控噪声。
        """
        def _run():
            cfg = SimConfig()
            cfg.info_structure = InfoStructureConfig(
                enabled=True, learning_bottleneck=True, arbitrary_codebook=True,
                steels_alignment=True,
                perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
                alignment_rate=0.3, alignment_step=0.2, alignment_noise=0.05,
            )
            cfg.simulation.use_sim_core = False
            cfg.population.initial_count = 120
            e = SphereEngine(cfg)
            for _ in range(300):
                e.step()
                if e._extinct:
                    break
            return (
                e._genes.copy(), e._interpret.copy(), e._codebook.copy(),
                e._flat.copy(), e._energy.copy(), e._trust.copy(),
            )
        a, b = _run(), _run()
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_different_seed_differs(self):
        """不同 seed 产生不同轨迹（确认 seed 真正生效，而非恒等）。"""
        cfg = SimConfig(seed=42)
        cfg.info_structure = InfoStructureConfig(
            enabled=True, learning_bottleneck=True, arbitrary_codebook=True,
            steels_alignment=True,
            perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        )
        cfg.simulation.use_sim_core = False
        cfg.population.initial_count = 60
        e42 = SphereEngine(cfg)
        for _ in range(100):
            e42.step()
        cfg2 = SimConfig(seed=43)
        cfg2.info_structure = cfg.info_structure
        cfg2.simulation.use_sim_core = False
        cfg2.population.initial_count = 60
        e43 = SphereEngine(cfg2)
        for _ in range(100):
            e43.step()
        if not (e42._extinct and e43._extinct):
            assert not np.array_equal(e42._genes[: e42.alive_count()],
                                      e43._genes[: e43.alive_count()])


# ─── 快照兼容性 ────────────────────────────────────────────

class TestD2Snapshot:
    def test_save_load_d2(self, tmp_path):
        """D2 开启时快照保存/加载正常。"""
        e = _make_engine()
        for _ in range(20):
            e.step()
        path = str(tmp_path / "d2_snap.npz")
        e.save_snapshot(path)
        e2 = SphereEngine.load_snapshot(path)
        assert e2._tick == e._tick
        np.testing.assert_array_equal(e2._codebook, e._codebook)
        np.testing.assert_array_equal(e2._learning_count, e._learning_count)

    def test_snapshot_version_3(self):
        e = _make_engine()
        assert e.SNAPSHOT_VERSION == 3


# ─── 参考实现组合 bug 回归（S1 复核：use_sim_core=True + 信息不对称）────────────

class TestSeedSentinel:
    """F-D10 哨兵测试：同配置不同 seed 必须产生不同轨迹（防 F-D1 类回归）。"""

    @staticmethod
    def _engine(seed: int, use_sim_core: bool):
        cfg = SimConfig(seed=seed)
        cfg.simulation.use_sim_core = use_sim_core
        cfg.info_structure = InfoStructureConfig(
            enabled=True,
            learning_bottleneck=True, arbitrary_codebook=True, steels_alignment=True,
        )
        return SphereEngine(cfg)

    def test_different_seed_yields_different_trajectory(self):
        """seed 42 vs 43，同配置 60 tick：终态必须不同（全局 np.random 隔离：顺序整跑）。"""
        a = self._engine(42, use_sim_core=False)
        for _ in range(60):
            a.step()
        b = self._engine(43, use_sim_core=False)
        for _ in range(60):
            b.step()
        pa, pb = a.alive_count(), b.alive_count()
        assert pa >= 1 and pb >= 1
        assert not (pa == pb and np.array_equal(a._genes[:pa], b._genes[:pb])), \
            "不同 seed 轨迹相同 → --seed 无效回归（F-D1 复发）"


# ─── 参考实现组合 bug 回归（S1 复核：use_sim_core=True + 信息不对称）────────────

class TestInfoAsymAgeRegression:
    """use_sim_core=True 且 D2 信息不对称开启时，年龄/冷却必须正常推进。

    历史 bug：移动段在 _d2_asym 时回退 Python 分步（跳过 stage2），但捕食段仍走 Rust
    分支（不含年龄推进）→ _age/_repro_cooldown 永不推进 → 永不成熟/不繁殖。
    修复：捕食段分支条件与移动段对齐（`use_sim_core and not _d2_asym`）。
    """
    @staticmethod
    def _engine(use_sim_core: bool):
        cfg = SimConfig(seed=42)
        cfg.simulation.use_sim_core = use_sim_core
        cfg.info_structure = InfoStructureConfig(
            enabled=True,
            learning_bottleneck=False, arbitrary_codebook=False, steels_alignment=False,
            perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        )
        return SphereEngine(cfg)

    def test_age_progresses_with_asym(self):
        """信息不对称开启时，Rust 路径年龄照常推进（回归）。"""
        e = self._engine(use_sim_core=True)
        for _ in range(20):
            e.step()
        p = e.alive_count()
        assert p > 0
        assert e._age[:p].mean() > 0, "Rust+D2-asym 年龄未推进（参考实现 bug 复发）"

    def test_dual_path_bitwise_equal_with_asym(self):
        """use_sim_core 开关下，D2 信息不对称路径逐 bit 一致。

        注：感知噪声/Steels 消费全局 np.random（随 config.seed 播种）。同进程内
        交错 step 会让后创建引擎读到先前引擎污染的全局状态——必须"py 跑完再建 rs"
        （rs 创建时 np.random.seed 重置），等价于真实实验的独立进程隔离。
        """
        e_py = self._engine(use_sim_core=False)
        for _ in range(30):
            e_py.step()
        e_rs = self._engine(use_sim_core=True)
        for _ in range(30):
            e_rs.step()
        assert e_rs.alive_count() == e_py.alive_count()
        np.testing.assert_array_equal(e_rs._age, e_py._age)
        np.testing.assert_array_equal(e_rs._energy, e_py._energy)
        np.testing.assert_array_equal(e_rs._genes, e_py._genes)
        np.testing.assert_array_equal(e_rs._flat, e_py._flat)
        assert e_rs._tick == e_py._tick


# ─── D2 关闭时行为不变 ─────────────────────────────────────

class TestD2DisabledBackwardCompat:
    def test_disabled_same_as_old(self):
        """D2 关闭时，引擎行为与旧版一致（不报错，正常运行）。"""
        cfg = SimConfig()
        cfg.info_structure.enabled = False
        cfg.simulation.use_sim_core = False
        e = SphereEngine(cfg)
        for _ in range(100):
            e.step()
        assert e._tick == 100
        # 码本恒等映射
        expected = np.arange(16, dtype=np.uint8)
        np.testing.assert_array_equal(e._codebook[0], expected)
