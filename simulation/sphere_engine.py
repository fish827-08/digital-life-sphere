"""SphereEngine：球面世界的生物引擎（模块二 · 文件 2/2）。

模块职责
--------
上文说清了"土地/天气/食物"（模块一），这份文件让**生物**在这上面
活起来：每个 tick，它们进食、转化能量、移动、衰老，可能死亡、繁殖。
所有个体用一张"大数据表"（NumPy 数组）同时推进，速度远快于逐个体循环。

规则语义（与主人确认过的版本）
-------------------------------
1. 代谢是"胃→能量的转化"：吃进胃里的食物不立即变能量；
   每 tick 按【代谢速率】把一部分胃转化为能量。
   - 转化总量 = 同样的食物最终都给同样能量（不会因为温度少给）；
   - 转化【速度】受温度和基因影响：合适温度快、冷的地方慢；
   - 吃得多先存在胃里，以后有空再慢慢转化（饱食度=胃容量上限）。
2. 移动是生物自己的决定（基因概率），温度只影响移动的【代价】：
   冷的地方移动更费能；爱扎堆的（g13）会朝同伴多的邻居走，
   独行侠（g13<0.5）专挑冷清的邻居走。
3. 基础维持消耗（体温/活动）每 tick 必扣；
   恒温个体（g9≈1）再多扣一点（恒温维持费），换取低温不减速。
4. 部分个体会光合（基因 g8）：白天按所在格光照获得少量能量（产能刻意小）。
5. 有觅食基因（g10）的个体，自己格不够吃时可以补吃一格外邻格的食物。
6. 繁殖时按基因 g7 决定把多少比例的能量/胃粮分给子代（默认 0.5 对半）；
   生完要休息（g12 冷却期），冷却没有过去之前攒再多能量也不生。
7. 有【成熟年龄】：未到寿命的一定比例（默认 15%，由 g3 寿命决定）绝不能繁衍，
   防止一出生就疯狂生。
8. 能量【需求随年龄变化】：幼体在长身体（维持费 ×1.6）、成年（×1）、
   老年器官退化（×1.4）——需要的能量不一样。

坐标：flat（平铺索引）代表"在哪个格子"，不用 x/y——
经度环绕、极点坍缩全部由 SphereWorld 内部处理，引擎只管搬格子。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.lifecycle import DeathCause
from simulation.config import SimConfig
from simulation.genes import Gene
from simulation.tick import TickStats
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.signal_field import SignalField
from world.sphere_world import SphereWorld


@dataclass(frozen=True)
class PopulationView:
    """一组只读的种群统计快照（per-tick 观察窗口，够 observer 用）。"""

    tick: int
    alive_count: int
    max_generation: int
    total_energy: float
    total_food_in_stomach: float


class SphereEngine:
    """球面世界引擎：NumPy 数组整群推进。

    实例属性（__slots__ 说明）
    -------------------------
    config : SimConfig
        全部参数（网格/光照/资源/生物/基因/种群/运行）。
    rng : np.random.Generator
        随机数源（用 config.seed 初始化，保证可复现）。
    world : SphereWorld
        土地（模块一文件1）：格子、面积、邻居。
    light : LightAndTemperature
        天气（模块一文件2）：光照、温度、活性。
    resources : ResourceField
        食物（模块一文件3）：存量、容量、再生。
    _flat : NDArray[int64]
        每个存活个体的位置（平铺索引）。
    _energy : NDArray[float64]
        每个个体"可用能量"（能用来移动/维持/繁殖的能量）。
    _stomach : NDArray[float64]
        每个个体"胃里的食物"（等待被代谢转化为能量）。
    _genes : NDArray[float64], 形状 (N, gene_count)
        每个个体的基因链。
    _age : NDArray[int64]
        每个个体已活的 tick 数。
    _generation : NDArray[int64]
        每个个体是第几代。
    _parent : NDArray[int64]
        每个个体的亲代 id（-1 = 初始个体）。
    _id : NDArray[int64]
        每个个体的唯一编号。
    _next_id : int
        下一个可用编号。
    _max_generation : int
        全种群最深的世代。
    _tick : int / _extinct / _finished :
        运行状态。
    _history : list[TickStats]
        逐 tick 统计（history_limit>0 时只留尾部）。
    _run_born / _run_died / _run_deaths :
        运行期累计账本（长程实验防历史无限增长）。
    """

    __slots__ = (
        "config", "rng", "world", "light", "resources", "signals",
        "_flat", "_energy", "_stomach", "_genes", "_age", "_generation",
        "_parent", "_id", "_next_id", "_max_generation",
        "_repro_cooldown",
        "_valence", "_arousal", "_expectation", "_baseline", "_trust",
        "_work_memory", "_mem_ptr", "_interpret", "_nb_table",
        "_tick", "_extinct", "_finished", "_history",
        "_history_limit", "_run_born", "_run_died", "_run_deaths",
        "_use_sim_core", "_sim_core",
    )

    # ---- 性状解码表（基因位 → 行为） --------------------------------
    # 基因位与行为的一一映射就定义在本文件（引擎热路径），不依赖其他模块：
    # g0 移动概率   g1 代谢倍率   g2 繁殖阈值   g3 寿命
    # g4 进食量倍率 g5 饱食度上限 g6 移动能耗倍率 g7 繁殖投入比例
    # g8 光合利用   g9 恒温指数   g10 邻格觅食（g≥0.5 能吃邻格）
    # g11 温度偏好 g12 繁殖冷却（间隔=g×60 tick） g13 群居性（g≥0.5 聚群，<0.5 避群）
    # （完整说明见 MODULES.md「模块二 · 基因表」）

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # 模块三：use_sim_core=True 时把种群数值管线下沉到 Rust（sim_core）
        self._use_sim_core = config.simulation.use_sim_core
        if self._use_sim_core:
            try:
                import sim_core  # 本地扩展，运行时注入
            except ImportError as exc:  # pragma: no cover - 构建问题，非逻辑错误
                raise RuntimeError(
                    "use_sim_core=True 但 sim_core 未安装：请先在 sim_core/ 运行"
                    " `.venv\\Scripts\\python -m maturin develop`"
                ) from exc
            self._sim_core = sim_core
            # Rust 基因索引常量 ↔ simulation.genes 注册表对照：防双写漂移。
            # native_gene_indicators() 返回 [(Rust 语义名, 值), ...]，
            # 任一项与 Python Gene 枚举不一致 ⇒ 引擎初始化直接报错。
            _native = getattr(sim_core, "native_gene_indicators", None)
            if _native is not None:
                _rust_const = {n: int(v) for n, v in _native()}
                _py_const = {g.name: int(g) for g in Gene}
                _drift = [
                    f"sim_core.{n}={v} != Gene.{n}={_py_const[n]}"
                    for n, v in _rust_const.items()
                    if n in _py_const and _py_const[n] != v
                ]
                if _drift:
                    raise RuntimeError(
                        "基因双写漂移：" + "; ".join(_drift)
                        + " —— 请同步 simulation/genes.py 与 sim_core/src/genes.rs"
                    )
        else:
            self._sim_core = None
        self.world = SphereWorld(
            rows=config.world.rows, cols=config.world.cols
        )
        self.light = LightAndTemperature(
            self.world,
            rotation_period=config.light.rotation_period,
            t_equator=config.light.t_equator,
            t_pole=config.light.t_pole,
            day_boost=config.light.day_boost,
            lat_base_ref=config.light.lat_base_ref,
        )
        self.resources = ResourceField(
            self.world, self.light,
            capacity_per_area=config.resources.capacity_per_area,
            regrowth_rate=config.resources.regrowth_rate,
            temp_sensitivity=config.resources.temp_sensitivity,
            distribution=config.resources.distribution,
            patch_count=config.resources.patch_count,
            patch_radius=config.resources.patch_radius,
            patch_capacity_mult=config.resources.patch_capacity_mult,
            patch_regrowth_mult=config.resources.patch_regrowth_mult,
            background_fill=config.resources.background_fill,
            initial_fill=config.resources.initial_fill,
            # 用 config.seed 派生 patch 中心：可复现，且独立 rng 不消费引擎 self.rng
            patch_seed=config.seed,
        )
        # 田字格信号场（L2/L3）：生物可写入/读取 16 种标记模式
        self.signals = SignalField(self.world, duration=50)

        # 预计算统一邻居表（L6 Rust 下沉用）：普通格 8 邻，极点格 cols 邻，
        # 统一到 nb_stride 列，未用位置填 -1。世界不变，只需构建一次。
        n_cells = self.world.n_cells
        nb_stride = max(8, self.world.cols)
        self._nb_table = np.full((n_cells, nb_stride), -1, dtype=np.int64)
        for c in range(n_cells):
            nbs = self.world.neighbors(c)
            self._nb_table[c, :len(nbs)] = nbs

        n = config.population.initial_count
        self._flat = np.zeros(n, dtype=np.int64)
        self._energy = np.full(
            n, config.organisms.initial_energy, dtype=np.float64
        )
        self._stomach = np.zeros(n, dtype=np.float64)
        self._genes = self.rng.uniform(
            config.genome.gene_min,
            config.genome.gene_max,
            size=(n, config.genome.gene_count),
        )
        self._age = np.zeros(n, dtype=np.int64)
        self._generation = np.zeros(n, dtype=np.int64)
        self._parent = np.full(n, -1, dtype=np.int64)
        self._repro_cooldown = np.zeros(n, dtype=np.float64)
        self._id = np.arange(n, dtype=np.int64)
        self._next_id = n
        self._max_generation = 0

        # 愉悦度系统（L2）：四数组 + 预测误差驱动
        # _valence: 情绪效价 -1~1（瞬态，RPE 响应）
        # _arousal: 唤醒度 0~1（意外事件→高唤醒）
        # _expectation: (N, 120) 预期表（120 情境，EWMA 学习）
        # _baseline: 基线慢漂移（习惯化）
        pcfg = self.config.pleasure
        self._valence = np.zeros(n, dtype=np.float64)
        self._arousal = np.full(n, 0.5, dtype=np.float64)
        self._baseline = np.zeros(n, dtype=np.float64)
        # 信任度（L5）：对信号的信任，初始 0.5（中性），真信号+假信号-
        self._trust = np.full(n, 0.5, dtype=np.float64)
        # 工作记忆（L5）：4 槽，存食物丰富格子位置（-1=空），round-robin 写入
        self._work_memory = np.full((n, 4), -1, dtype=np.int64)
        self._mem_ptr = 0
        # 信号解读表（L5 文化传递）：(N,16)，对 16 种信号模式的响应倾向
        # 正值=移向，负值=逃避，0=忽略；初始随机，幼体向周围成体学习
        self._interpret = self.rng.normal(0.0, 0.3, size=(n, 16))
        # 乐观初始化：0.8 × max_reward，逼生物探索（预期高→现实可能超预期→愉悦）
        self._expectation = np.full(
            (n, pcfg.expectation_size),
            pcfg.optimism * pcfg.max_reward,
            dtype=np.float64,
        )

        # 出生位置：均匀随机格（不做地形障碍过滤，球面无障碍）
        self._flat = self.rng.integers(0, self.world.n_cells, size=n).astype(
            np.int64
        )

        self._tick = 0
        self._extinct = False
        self._finished = False
        self._history: list[TickStats] = []
        self._history_limit = config.simulation.history_limit
        self._run_born = 0
        self._run_died = 0
        self._run_deaths: Counter = Counter()

    # ---- 只读状态（给观察者用） ------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    def alive_count(self) -> int:
        return len(self._id)

    def population_view(self) -> PopulationView:
        return PopulationView(
            tick=self._tick,
            alive_count=len(self._id),
            max_generation=self._max_generation,
            total_energy=float(self._energy.sum()),
            total_food_in_stomach=float(self._stomach.sum()),
        )

    @property
    def extinct(self) -> bool:
        return self._extinct

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def history(self) -> list[TickStats]:
        return self._history

    @property
    def total_born(self) -> int:
        """累计出生数（与 history_limit 无关，全局计数）。"""
        return self._run_born

    @property
    def total_died(self) -> int:
        """累计死亡数（与 history_limit 无关，全局计数）。"""
        return self._run_died

    def death_cause_totals(self) -> Counter:
        """累计死因分布（与 history_limit 无关，全局计数）。"""
        return self._run_deaths.copy()

    # ---- 主循环 ----------------------------------------------------------

    def step(self) -> TickStats:
        """推进一个 tick，返回统计快照。与旧引擎的 step() 对齐。

        注意：step() 会先推进内部时钟（_tick += 1）再执行整套流程，
        所以"手动逐 tick 调度"与 run() 的行为完全一致（环境时间都会前进）。
        """
        if self._finished:
            raise RuntimeError("模拟已结束，无法继续 step()")
        self._tick += 1
        stats = self._advance_one_tick()
        self._populate_history(stats)
        return stats

    def _advance_one_tick(self) -> TickStats:
        # 顺序契约：先资源再生，再种群行动
        # uniform：Rust regrow（3.2）；patchy：Rust regrow_patchy（L7e，含空间倍率守恒）。
        #   旧版 .pyd 没有 regrow_patchy 时特性检测回退 Python，保证双路径不炸。
        if self._use_sim_core and self.resources.distribution == "uniform":
            # 3.4：资源再生长沉到 Rust（与 ResourceField.regrow 逐位等价，
            # 默认 temp_sensitivity=1.0 时严格一致；≠1 有 ≤1-ULP 差异）
            self._sim_core.regrow(
                self.resources._grid, self.resources._capacity,
                self.light.temperature(
                    np.arange(self.world.n_cells), self._tick
                ),
                self.resources.regrowth_rate,
                self.resources.temp_sensitivity,
            )
        elif (
            self._use_sim_core
            and self.resources.distribution == "patchy"
            and hasattr(self._sim_core, "regrow_patchy")
        ):
            # L7e：patchy 再生下沉 Rust（斑块格×patch_mult / 背景格×bg_mult，守恒）
            self._sim_core.regrow_patchy(
                self.resources._grid, self.resources._capacity,
                self.light.temperature(
                    np.arange(self.world.n_cells), self._tick
                ),
                self.resources._patch_mask.astype(np.uint8),
                self.resources._patch_regrowth_mult,
                self.resources._bg_regrowth_mult,
                self.resources.regrowth_rate,
                self.resources.temp_sensitivity,
            )
        else:
            self.resources.regrow(self._tick)
        # 信号场时间推进（标记衰减、过期清零）
        self.signals.tick()
        born, died, deaths = self._step_population()
        return TickStats(
            tick=self._tick,
            population=len(self._id),
            born=born,
            died=died,
            deaths_by_cause=deaths,
            total_energy=float(self._energy.sum()),
            total_resource=self.resources.total(),
        )

    def run(self, ticks: Optional[int] = None) -> list[TickStats]:
        """一直跑到 ticks 或自然结束，返回历史统计。"""
        limit = self.config.simulation.ticks if ticks is None else ticks
        while not self._finished and self._tick < limit:
            self.step()
            self._finished = self._end_condition_met()
        return list(self._history)

    def _populate_history(self, stats: TickStats) -> None:
        self._history.append(stats)
        # 全局累计（与 history_limit 无关：观察台/存档用）
        self._run_born += stats.born
        self._run_died += stats.died
        self._run_deaths.update(stats.deaths_by_cause)
        if self._history_limit > 0:
            overflow = len(self._history) - self._history_limit
            if overflow > 0:
                del self._history[:overflow]

    def _end_condition_met(self) -> bool:
        if self._extinct and self.config.simulation.stop_on_extinction:
            return True
        return self._tick >= self.config.simulation.ticks

    # ---- 数值平衡：寿命与昼夜 ---------------------------------------------

    def _lifespan(self, g3: NDArray[np.float64]) -> NDArray[np.float64]:
        """寿命（tick）= 一昼夜 × (1 + g3×7) → 最短 1 昼夜，最长 8 昼夜。

        用"昼夜长度"做基准而非绝对 tick 数：昼夜设置（rotation_period）
        调整时，寿命自动跟着缩放，保证生物始终能经历若干完整昼夜。
        g3=0 最少活满一昼夜（对昼夜有反应）；g3=1 最多八昼夜（慢热长寿）。
        """
        day = float(self.config.light.rotation_period)
        return day * (1.0 + g3 * 7.0)

    # ---- 单 tick 种群推进（核心热循环） -----------------------------------

    def _step_population(self) -> tuple[int, int, Counter]:
        """整群推进一步。返回 (born, died, deaths)。"""
        ocfg, gcfg = self.config.organisms, self.config.genome
        ccfg = self.config.culture          # 隐式选择压参数化（A2）：信任学习幅值
        P = len(self._id)
        if P == 0:
            self._extinct = True
            return 0, 0, Counter()

        # 温度相关量（一次算出全种群的那份，避免反复调用）
        activity = self.light.activity_factor(self._flat, self._tick)

        genes = self._genes[:P]
        energy = self._energy[:P]
        stomach = self._stomach[:P]
        # 愉悦度：记录 tick 开始时的能量（步骤 1~8 会修改 energy）
        energy_before = energy.copy() if self.config.pleasure.enabled else None

        # 0) 个体化温度响应（恒温基因 g9 + 温度偏好基因 g11）：
        #    冷血（g9=0）完全随环境：低温时消化慢、行动贵；
        #    恒温（g9=1）内部温度恒定：低温不再压制，但每 tick 多扣维持费。
        homeo = genes[:, Gene.HOMEOTHERM]
        eff_activity = activity + homeo * (1.0 - activity)
        #    温度偏好（g11）：冷血个体只在自己偏好的温度附近才满速，
        #    偏离越远越慢（偏离 15° 活力掉到 ≈6 成）；恒温者不受影响。
        #    偏好温度 = -10 + g11×50，落在 [-10°C, 40°C] 覆盖全地图谱系。
        pref_temp = -10.0 + genes[:, Gene.TEMP_PREF] * 50.0
        grid_temp = self.light.temperature(self._flat, self._tick)
        niche_match = np.exp(-0.5 * ((grid_temp - pref_temp) / 15.0) ** 2)
        eff_activity = np.where(
            homeo >= 0.5,
            eff_activity,
            eff_activity * (0.4 + 0.6 * niche_match),
        )
        cold_penalty = 2.0 - eff_activity  # 冷 => 活动代价高（恒温者不受影响）

        # 1)~3) 光合 / 代谢 / 维持 —— 双路径（Rust stage1 逐位等价，或 Python 原实现）
        day_len = float(self.config.light.rotation_period)
        if self._use_sim_core:
            self._sim_core.step_vectors_stage1(
                energy, stomach, genes, eff_activity,
                self.light.illumination(self._flat, self._tick),
                self._age[:P],
                ocfg.photo_max, ocfg.base_metabolism, ocfg.eat_efficiency,
                ocfg.growth_mult, ocfg.senile_mult,
                ocfg.maturity_fraction, ocfg.senile_fraction,
                ocfg.homeo_upkeep, day_len,
            )
        else:
            # 1) 光合收入（g8）：少量、随光照。
            #    收入 = 光照(所在格,当前tick) × g8 × photo_max。
            #    植物化增强（g19）在 stage1 之后统一补，确保双路径一致。
            energy += (
                self.light.illumination(self._flat, self._tick)
                * genes[:, Gene.PHOTOSYNTHESIS]
                * ocfg.photo_max
            )

            # 2) 代谢转化：胃 → 能量（总量不变，快慢受温度+基因影响）
            #    转化速率 = base_metabolism × metabolic_mult × eff_activity
            #    metabolic_mult = 0.5 + g1×1.5（基因放大代谢快慢）
            metab_mult = 0.5 + genes[:, Gene.METABOLIC] * 1.5
            digest_rate = ocfg.base_metabolism * metab_mult * eff_activity
            # 每 tick 最多转化这么多；不得超出胃里有的
            digest = np.minimum(stomach, digest_rate)
            energy += digest * ocfg.eat_efficiency  # 同样食物 → 同样能量
            stomach -= digest

            # 3) 基础维持消耗（体温/活动，必扣，与温度无关基础价）
            #    随年龄的"需求"阶段：幼体在长身体（growth_mult 倍）→
            #    成年（1 倍）→ 老年器官退化（senile_mult 倍）。
            #    成熟/老年年龄按各自寿命（g3）的比例划分，寿命长的物种成熟和老化都更晚。
            life_span = self._lifespan(genes[:, Gene.LIFE_GENE])
            age_f = self._age[:P].astype(np.float64)
            maturity_age = ocfg.maturity_fraction * life_span
            senile_age = ocfg.senile_fraction * life_span
            age_mult = np.ones(P, dtype=np.float64)
            age_mult[age_f < maturity_age] = ocfg.growth_mult
            age_mult[age_f >= senile_age] = ocfg.senile_mult
            energy -= ocfg.base_metabolism * metab_mult * age_mult
            #    + 恒温维持费（g9）：恒温个体每 tick 另付 homeo_upkeep
            energy -= ocfg.homeo_upkeep * homeo

        # 3.5) 植物化光合增强（g19）：统一在 stage1 之后补，确保 Rust/Python 双路径一致
        #     扎根个体（g19 高）额外获得 光照×g8×g19×photo_max 的光合收入
        energy += (
            self.light.illumination(self._flat, self._tick)
            * genes[:, Gene.PHOTOSYNTHESIS]
            * genes[:, Gene.ROOTING]
            * ocfg.photo_max
        )

        # 4) 进食：从格子里吃进胃（先吃后扣基础维持，保证当天能吃到）
        #    饱食度：胃容量上限（基础 = max_energy/eat_efficiency/2，g5 缩放 0.5~2 倍）
        #    进食量：每 tick 最多 eat_amount（g4 缩放 0.5~1.5 倍）
        eat_mult = 0.5 + genes[:, Gene.EAT_AMOUNT] * 1.0
        cap_mult = 0.5 + genes[:, Gene.STOMACH_CAP] * 1.5
        stomach_cap = (
            ocfg.max_energy / max(1e-9, ocfg.eat_efficiency) * 0.5 * cap_mult
        )
        if (stomach < stomach_cap).any():
            eaters = np.flatnonzero(stomach < stomach_cap)
            want = np.minimum(
                ocfg.eat_amount * eat_mult[eaters],
                stomach_cap[eaters] - stomach[eaters],
            )
            # 3.5：批量进食双路径（Rust consume_many 与 numpy consume_many 逐位等价）
            if self._use_sim_core:
                taken = np.empty(len(eaters), dtype=np.float64)
                self._sim_core.consume_many(
                    self.resources._grid, self._flat[eaters], want, taken
                )
            else:
                taken = self.resources.consume_many(self._flat[eaters], want)
            stomach[eaters] += taken
            # 邻格觅食（g10）：自己格不够吃的个体，随机吃一格外邻格
            short = want - taken
            hung = np.flatnonzero(short > 1e-9)
            hunter = self._genes[hung, Gene.FORAGE_NEIGHBOR] >= 0.5
            if hunter.any():
                hf = hung[hunter]
                targets = np.empty(int(hunter.sum()), dtype=np.int64)
                for j, fi in enumerate(hf):
                    nb = self.world.neighbors(int(self._flat[fi]))
                    targets[j] = nb[int(self.rng.integers(0, len(nb)))]
                if self._use_sim_core:
                    taken2 = np.empty(len(hf), dtype=np.float64)
                    self._sim_core.consume_many(
                        self.resources._grid, targets, short[hf], taken2
                    )
                else:
                    taken2 = self.resources.consume_many(targets, short[hf])
                stomach[hf] += taken2

        # 4.4) 工作记忆写入（L5）：食物丰富的格子记入 4 槽（round-robin）
        cur_flat = self._flat[:P]
        food_rich = (
            self.resources._grid[cur_flat]
            > 0.5 * self.resources._capacity[cur_flat]
        )
        if food_rich.any():
            rich_idx = np.flatnonzero(food_rich)
            ptr = int(self._mem_ptr)
            self._work_memory[rich_idx, ptr] = cur_flat[rich_idx]
            self._mem_ptr = (ptr + 1) % 4

        # 4.5) 信号发射（g15）：以基因概率在当前格写入田字格标记，耗能
        #     模式 = 状态哈希（能量2位 + 食物1位 + 邻居1位 = 4位=16种）
        #     use_sim_core=True：下沉到 Rust signal_emit（L7a C1），
        #     rand_emit 在 Python 侧预生成，保证 RNG 消费顺序与纯 Python 一致。
        signal_gene = genes[:, Gene.SIGNAL_STRENGTH]
        rand_emit = self.rng.random(P)  # 双路径共用：Python 直接用，Rust 传入
        if self._use_sim_core:
            SIGNAL_COST = 0.1
            dens = np.bincount(self._flat[:P], minlength=self.world.n_cells).astype(np.float64)
            self._sim_core.signal_emit(
                self._flat[:P], energy, signal_gene.copy(), rand_emit,
                dens, self.resources._grid, self.resources._capacity,
                self.signals._marks, self.signals._age,
                SIGNAL_COST, ocfg.max_energy, self.signals.duration,
            )
        else:
            emitters = np.flatnonzero(rand_emit < signal_gene)
            if len(emitters):
                SIGNAL_COST = 0.1  # 发射成本（降低，让信号基因不被纯成本淘汰）
                can_afford = energy[emitters] >= SIGNAL_COST
                emitters = emitters[can_afford]
                if len(emitters):
                    energy[emitters] -= SIGNAL_COST
                    e_flat = self._flat[emitters]
                    # 模式编码：能量档(2位,bit3-2) + 食物(1位,bit1) + 邻居(1位,bit0)
                    e_bin = np.clip(
                        (energy[emitters] / max(1e-9, ocfg.max_energy) * 4).astype(np.int64), 0, 3
                    )
                    f_bit = (
                        self.resources._grid[e_flat]
                        > 0.5 * self.resources._capacity[e_flat]
                    ).astype(np.int64)
                    dens = np.bincount(self._flat[:P], minlength=self.world.n_cells)
                    n_bit = (dens[e_flat] > 1).astype(np.int64)
                    patterns = (e_bin * 4 + f_bit * 2 + n_bit).astype(np.uint8)
                    self.signals.write_many(e_flat, patterns)

        # 5) 移动：g19 植物化降低移动概率（g0 × (1-g19)）
        #    use_sim_core=True：移动决策下沉到 Rust（step_movement），移动耗能在 Rust 内扣；
        #    stage2 只做年龄/冷却（moved_raw 全 False，不重复扣移动耗能）。
        move_prob = genes[:, Gene.MOVE_PROB] * (1.0 - genes[:, Gene.ROOTING])
        move_cost_ind = ocfg.move_cost * cold_penalty * (0.5 + genes[:, Gene.MOVE_COST])
        if self._use_sim_core:
            moved_raw = self.rng.random(P) < move_prob
            mi = np.flatnonzero(moved_raw & (energy >= move_cost_ind))
            # stage2：moved_raw 全 False（移动耗能改由 step_movement 扣），年龄/冷却正常
            out_moved = np.empty(P, dtype=bool)
            out_starved = np.empty(P, dtype=bool)
            out_expired = np.empty(P, dtype=bool)
            out_repro = np.empty(P, dtype=bool)
            self._sim_core.step_vectors_stage2(
                energy, self._age[:P], self._repro_cooldown[:P], genes,
                np.zeros(P, dtype=bool), move_cost_ind,
                out_moved, out_starved, out_expired, out_repro,
                day_len, ocfg.maturity_fraction, ocfg.max_energy,
            )
            if len(mi) > 0:
                # 预计算环境量
                food_ratio = self.resources._grid / np.maximum(
                    self.resources._capacity, 1e-9
                )
                sig_present = (self.signals._marks > 0).astype(np.float64)
                densities = np.bincount(
                    self._flat, minlength=self.world.n_cells
                ).astype(np.float64)
                signal_marks = self.signals._marks.astype(np.uint8)
                # 预生成随机选择（得分无差异时用），按移动个体顺序
                rand_choice = self.rng.integers(
                    0, 1_000_000, size=len(mi), dtype=np.int64
                )
                # Rust 移动决策（含移动扣费）
                self._sim_core.step_movement(
                    self._flat[:P], energy, genes, self._trust[:P],
                    self._work_memory[:P].reshape(-1), self._interpret[:P],
                    food_ratio, sig_present, densities, signal_marks,
                    self._nb_table.reshape(-1),
                    mi.astype(np.int64), rand_choice, move_cost_ind,
                    self.world.n_cells, self._nb_table.shape[1],
                )
                # 5.6) 信任学习：移动到有信号的格子后验证真假
                target_cells = self._flat[mi]
                had_signal = sig_present[target_cells] > 0
                has_food = food_ratio[target_cells] > ccfg.food_threshold
                true_sig = had_signal & has_food
                false_sig = had_signal & ~has_food
                self._trust[mi[true_sig]] = np.minimum(
                    1.0, self._trust[mi[true_sig]] + ccfg.trust_true
                )
                self._trust[mi[false_sig]] = np.maximum(
                    0.0, self._trust[mi[false_sig]] - ccfg.trust_false
                )
        else:
            moved = self.rng.random(P) < move_prob
            moved &= energy >= move_cost_ind  # 付得起才走
            Nm = int(moved.sum())
            if Nm:
                mi = np.flatnonzero(moved)
                food_ratio = self.resources._grid / np.maximum(
                    self.resources._capacity, 1e-9
                )
                sig_present = (self.signals._marks > 0).astype(np.float64)
                densities = np.bincount(
                    self._flat, minlength=self.world.n_cells
                ).astype(np.float64)
                rand_choice = self.rng.integers(
                    0, 1_000_000, size=Nm, dtype=np.int64
                )
                targets = np.empty(Nm, dtype=np.int64)
                for i, idx in enumerate(mi):
                    nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    if len(nb) == 1:
                        targets[i] = nb[0]
                        continue
                    perc = genes[idx, Gene.PERCEPTION]
                    soc = (genes[idx, Gene.SOCIABILITY] - 0.5) * 2.0
                    score = perc * (
                        food_ratio[nb] * 0.5 + sig_present[nb] * 0.5 * self._trust[idx]
                    ) + soc * densities[nb]
                    valid_mem = self._work_memory[idx][self._work_memory[idx] >= 0]
                    if len(valid_mem) > 0:
                        mem_in_nb = np.isin(nb, valid_mem)
                        score = score + 0.3 * perc * mem_in_nb.astype(np.float64)
                    nb_sigs = self.signals._marks[nb]
                    if (nb_sigs > 0).any():
                        interp = np.array(
                            [self._interpret[idx, int(s)] if s > 0 else 0.0 for s in nb_sigs],
                            dtype=np.float64,
                        )
                        score = score + 0.4 * perc * interp
                    if score.max() - score.min() < 1e-9:
                        targets[i] = nb[int(rand_choice[i] % len(nb))]
                    else:
                        targets[i] = nb[int(np.argmax(score))]
                self._flat[mi] = targets
                energy[mi] -= move_cost_ind[mi]
                # 5.6) 信任学习
                target_cells = self._flat[mi]
                had_signal = sig_present[target_cells] > 0
                has_food = food_ratio[target_cells] > ccfg.food_threshold
                true_sig = had_signal & has_food
                false_sig = had_signal & ~has_food
                self._trust[mi[true_sig]] = np.minimum(
                    1.0, self._trust[mi[true_sig]] + ccfg.trust_true
                )
                self._trust[mi[false_sig]] = np.maximum(
                    0.0, self._trust[mi[false_sig]] - ccfg.trust_false
                )

        # 5.5) 捕食（g16）+ 6.5) 文化学习（L5）：use_sim_core=True 时合并为一次 Rust 调用，
        #     共用一次 cell→个体 CSR 构建，消除重复开销。use_sim_core=False 时分步执行。
        predation_mask = np.zeros(P, dtype=bool)
        pcfg = self.config.predation           # 隐式选择压参数化（A2）：捕食段参数
        attack_gene = genes[:, Gene.AGGRESSION]
        hunger = np.clip(1.0 - energy / max(ocfg.max_energy, 1e-9), 0.0, 1.0)
        attack_prob = attack_gene * pcfg.attack_prob_coef * hunger
        attackers = np.flatnonzero(
            (attack_gene > pcfg.attack_gene_gate) & (self.rng.random(P) < attack_prob)
        )
        # 文化学习需要的成熟年龄（年龄已在 stage2 推进，use_sim_core=True 时）
        age_f = self._age[:P].astype(np.float64)
        life_span = self._lifespan(genes[:, Gene.LIFE_GENE])
        maturity_age = ocfg.maturity_fraction * life_span

        if self._use_sim_core:
            # 合并调用：捕食 + 文化学习，共用一次 CSR
            if len(attackers) > 0:
                rand_prey = self.rng.integers(0, 1_000_000, size=len(attackers), dtype=np.int64)
                rand_success = self.rng.random(len(attackers))
            else:
                rand_prey = np.zeros(0, dtype=np.int64)
                rand_success = np.zeros(0, dtype=np.float64)
            self._sim_core.predation_and_culture(
                energy, stomach, predation_mask, self._interpret,
                self._flat[:P], genes, self._age[:P], maturity_age,
                attackers.astype(np.int64), rand_prey, rand_success,
                self._nb_table.reshape(-1),
                self.world.n_cells, self._nb_table.shape[1],
                ocfg.max_energy, ocfg.eat_efficiency, 0.1,
                pcfg.attack_cost, pcfg.success_gene_gain,
                pcfg.success_floor, pcfg.success_ceil,
                pcfg.transfer_ratio, pcfg.stomach_transfer,
            )
        else:
            # ── Python 分步：捕食 ──
            if len(attackers) > 0:
                rand_prey = self.rng.integers(0, 1_000_000, size=len(attackers), dtype=np.int64)
                rand_success = self.rng.random(len(attackers))
                for k, idx in enumerate(attackers):
                    if energy[idx] <= pcfg.attack_cost:
                        continue
                    nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    nb_mask = np.isin(self._flat[:P], nb) & (np.arange(P) != idx)
                    prey_candidates = np.flatnonzero(nb_mask)
                    if len(prey_candidates) == 0:
                        continue
                    prey = int(prey_candidates[int(rand_prey[k] % len(prey_candidates))])
                    if predation_mask[prey]:
                        continue
                    energy[idx] -= pcfg.attack_cost
                    success_rate = np.clip(
                        (energy[idx] / max(energy[idx] + energy[prey], 1e-9))
                        * (0.5 + attack_gene[idx] * pcfg.success_gene_gain),
                        pcfg.success_floor, pcfg.success_ceil,
                    )
                    if rand_success[k] < success_rate:
                        predation_mask[prey] = True
                        energy[idx] += energy[prey] * pcfg.transfer_ratio
                        stomach[idx] = np.minimum(
                            stomach[idx] + stomach[prey] * pcfg.stomach_transfer,
                            ocfg.max_energy / max(1e-9, ocfg.eat_efficiency),
                        )
                        stomach[prey] = 0.0
            # ── Python 分步：年龄推进（Rust 路径已由 stage2 就地 +1）──
            self._age[:P] += 1
            age_f = self._age[:P].astype(np.float64)
            # ── Python 分步：文化学习 ──
            juvenile = age_f < maturity_age
            if juvenile.any():
                j_idx = np.flatnonzero(juvenile)
                for idx in j_idx:
                    nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    nb_mask = np.isin(self._flat[:P], nb)
                    adult_nb = nb_mask & (age_f >= maturity_age)
                    adult_idx = np.flatnonzero(adult_nb)
                    if len(adult_idx) > 0:
                        mean_interpret = self._interpret[adult_idx].mean(axis=0)
                        self._interpret[idx] += 0.1 * (mean_interpret - self._interpret[idx])

        # 7) 死亡判定：饿死（energy<=0）→ 老死（age>=寿命）→ 被捕食
        # 注意：统一用 Python 计算（不用 Rust 的 out_starved/out_expired），
        # 因为捕食（步骤 5.5）在 Rust stage2 之后才执行，会改变 energy。
        starved = energy <= 0.0
        expired = (~starved) & (age_f >= life_span)
        dead = starved | expired | predation_mask
        deaths: Counter = Counter()
        n_starved = int(starved.sum())
        n_expired = int(expired.sum())
        n_predation = int(predation_mask.sum())
        if n_starved:
            deaths[DeathCause.STARVATION] = n_starved
        if n_expired:
            deaths[DeathCause.OLD_AGE] = n_expired
        if n_predation:
            deaths[DeathCause.PREDATION] = n_predation

        # 8) 繁殖：冷却期（g12）倒数；能量 ≥ 阈值（0.25+g2×0.65），且种群未满
        # 注意：统一用 Python 计算繁殖判定（不用 Rust 的 out_repro），
        # 因为捕食（5.5）和植物化（3.5）在 Rust stage2 之后才执行，会改变 energy。
        if not self._use_sim_core:
            self._repro_cooldown[:P] = np.maximum(
                0.0, self._repro_cooldown[:P] - 1.0
            )
        repro_thr = (0.25 + genes[:, Gene.REPRO_THRESHOLD] * 0.65) * ocfg.max_energy
        repro = (
            (~dead)
            & (energy >= repro_thr)
            & (self._repro_cooldown[:P] <= 0.0)
            & (age_f >= maturity_age)   # 未到成熟年龄不生（长大后才能繁衍）
        )
        K = min(int(repro.sum()), self.config.population.max_count - P)
        born = 0
        if K > 0:
            ri = np.flatnonzero(repro)[:K]
            child_genes = self._genes[ri].copy()
            mut = self.rng.random((K, gcfg.gene_count)) < gcfg.mutation_rate
            if mut.any():
                sigma = gcfg.mutation_sigma * (gcfg.gene_max - gcfg.gene_min)
                noise = self.rng.normal(0.0, sigma, size=(K, gcfg.gene_count))
                child_genes = np.clip(
                    child_genes + np.where(mut, noise, 0.0),
                    gcfg.gene_min, gcfg.gene_max,
                )
            # 传代投入比例（g7）：亲代把多少比例的能量/胃粮分给子代（0.3~0.7）
            split = 0.3 + genes[ri, Gene.PARENTAL_INVEST] * 0.4
            child_energy = energy[ri] * split
            child_stomach = stomach[ri] * split
            energy[ri] -= child_energy
            stomach[ri] -= child_stomach
            # 生完进入冷却（g12）：间隔 = g12 × 60 tick，冷却没到攒再多也不生
            self._repro_cooldown[ri] = genes[ri, Gene.REPRO_COOLDOWN] * 60.0

            ids = np.arange(self._next_id, self._next_id + K, dtype=np.int64)
            self._next_id += K
            self._id = np.concatenate([self._id, ids])
            self._flat = np.concatenate([self._flat, self._flat[ri]])
            self._energy = np.concatenate([self._energy, child_energy])
            self._stomach = np.concatenate([self._stomach, child_stomach])
            self._genes = np.concatenate([self._genes, child_genes])
            self._age = np.concatenate([self._age, np.zeros(K, dtype=np.int64)])
            self._repro_cooldown = np.concatenate(
                [self._repro_cooldown, np.zeros(K, dtype=np.float64)]
            )
            new_gen = self._generation[ri] + 1
            self._generation = np.concatenate([self._generation, new_gen])
            self._parent = np.concatenate([self._parent, self._id[ri]])
            # 愉悦度：子代继承亲代 expectation + 噪声（文化传递载体）
            pcfg = self.config.pleasure
            child_exp = self._expectation[ri].copy()
            child_exp += self.rng.normal(
                0.0, pcfg.inheritance_noise, size=child_exp.shape
            )
            child_exp = np.clip(child_exp, 0.0, pcfg.max_reward)
            self._expectation = np.concatenate([self._expectation, child_exp])
            self._valence = np.concatenate(
                [self._valence, np.zeros(K, dtype=np.float64)]
            )
            self._arousal = np.concatenate(
                [self._arousal, np.full(K, 0.5, dtype=np.float64)]
            )
            self._baseline = np.concatenate(
                [self._baseline, self._baseline[ri] * 0.5]
            )
            # 信任度：子代半继承亲代，回归中性 0.5
            child_trust = self._trust[ri] * 0.8 + 0.5 * 0.2
            self._trust = np.concatenate([self._trust, child_trust])
            # 工作记忆：子代继承亲代的食物位置记忆（文化传递的一部分）
            self._work_memory = np.concatenate(
                [self._work_memory, self._work_memory[ri].copy()]
            )
            # 信号解读表：子代继承亲代 + 噪声（文化传递的核心载体）
            child_interp = self._interpret[ri].copy()
            child_interp += self.rng.normal(0.0, 0.1, size=child_interp.shape)
            self._interpret = np.concatenate([self._interpret, child_interp])
            born = K
            new_max = int(self._generation.max())
            if new_max > self._max_generation:
                self._max_generation = new_max

        # 8.5) 愉悦度更新（RPE 预测误差驱动）：对前 P 个亲代计算
        if self.config.pleasure.enabled and energy_before is not None:
            self._update_pleasure(P, energy_before)

        # 9) 清理尸体（只清理本 tick 行动的旧行；子代不受影响）
        if dead.any():
            keep = ~dead
            n_tail = len(self._id) - P
            self._id = np.concatenate([self._id[:P][keep], self._id[P:]])
            self._flat = np.concatenate([self._flat[:P][keep], self._flat[P:]])
            self._energy = np.concatenate(
                [self._energy[:P][keep], self._energy[P:]]
            )
            self._stomach = np.concatenate(
                [self._stomach[:P][keep], self._stomach[P:]]
            )
            self._genes = np.concatenate([self._genes[:P][keep], self._genes[P:]])
            self._age = np.concatenate([self._age[:P][keep], self._age[P:]])
            self._generation = np.concatenate(
                [self._generation[:P][keep], self._generation[P:]]
            )
            self._parent = np.concatenate([self._parent[:P][keep], self._parent[P:]])
            self._repro_cooldown = np.concatenate(
                [self._repro_cooldown[:P][keep], self._repro_cooldown[P:]]
            )
            self._valence = np.concatenate(
                [self._valence[:P][keep], self._valence[P:]]
            )
            self._arousal = np.concatenate(
                [self._arousal[:P][keep], self._arousal[P:]]
            )
            self._baseline = np.concatenate(
                [self._baseline[:P][keep], self._baseline[P:]]
            )
            self._expectation = np.concatenate(
                [self._expectation[:P][keep], self._expectation[P:]]
            )
            self._trust = np.concatenate(
                [self._trust[:P][keep], self._trust[P:]]
            )
            self._work_memory = np.concatenate(
                [self._work_memory[:P][keep], self._work_memory[P:]]
            )
            self._interpret = np.concatenate(
                [self._interpret[:P][keep], self._interpret[P:]]
            )

        if len(self._id) == 0:
            self._extinct = True
        return born, n_starved + n_expired + n_predation, Counter(deaths)

    # ---- 愉悦度系统（L2） --------------------------------------------------

    def _update_pleasure(self, P: int, energy_before: NDArray[np.float64]) -> None:
        """对前 P 个亲代更新愉悦度（RPE 预测误差驱动）。

        核心：愉悦 = 实际获得 − 预期获得。
        1. 情境编码（120 种：能量5×食物4×邻居3×信号2）
        2. 事件收益（Δ能量 + 社会增益，信息增益待信号场接入）
        3. RPE = 收益 − expectation[情境]
        4. 更新 valence（瞬态）/arousal（唤醒）/expectation（EWMA 学习）/baseline（习惯化）

        只更新前 P 个个体（本 tick 开始时存在的亲代），子代不参与本 tick。
        """
        pcfg = self.config.pleasure
        max_e = self.config.organisms.max_energy
        idx = np.arange(P)
        flat = self._flat[:P]
        energy_now = self._energy[:P]

        if self._use_sim_core:
            # L7a C2：愉悦度更新下沉 Rust，双路径逐位一致
            densities = np.bincount(flat, minlength=self.world.n_cells).astype(np.float64)
            self._sim_core.pleasure_update(
                flat, energy_now, energy_before,
                densities, self.resources._grid, self.resources._capacity,
                self.signals._marks,
                self._valence[:P], self._arousal[:P],
                self._expectation[:P].reshape(-1), self._baseline[:P],
                max_e, pcfg.alpha, pcfg.valence_decay, pcfg.arousal_decay,
                pcfg.baseline_rate, pcfg.max_reward,
                pcfg.w_energy, pcfg.w_info, pcfg.w_social,
            )
            return

        # 1) 情境编码
        # 能量档：energy/max_energy → 0~4
        e_bin = np.clip((energy_now / max_e) * 5, 0, 4).astype(np.int64)
        # 食物档：所在格食物/capacity → 0~3
        food_ratio = np.clip(
            self.resources._grid[flat] / np.maximum(self.resources._capacity[flat], 1e-9),
            0.0, 1.0,
        )
        f_bin = np.clip((food_ratio * 4).astype(np.int64), 0, 3)
        # 邻居档：所在格密度 → 0(无)/1(1-2)/2(3+)
        densities = np.bincount(flat, minlength=self.world.n_cells)
        n_count = densities[flat]
        n_bin = np.where(n_count == 0, 0, np.where(n_count <= 2, 1, 2))
        # 信号档：第一版信号场未接入引擎，全 0（待 L3 信号基因接入后填）
        s_bin = np.zeros(P, dtype=np.int64)
        # 组合索引：e×24 + f×6 + n×2 + s
        context = e_bin * 24 + f_bin * 6 + n_bin * 2 + s_bin

        # 2) 事件收益
        delta_e = np.clip((energy_now - energy_before) / max_e, -1.0, 1.0)
        social = np.where(n_count > 0, pcfg.social_rpe, pcfg.alone_rpe)  # 有同伴→正，孤独→负
        # 信息增益：所在格有信号标记 → 获得信息（好奇心满足），权重提高
        info = np.where(self.signals._marks[flat] > 0, 0.5, 0.0)
        reward = pcfg.w_energy * delta_e + pcfg.w_info * info + pcfg.w_social * social

        # 3) RPE = 实际 − 预期
        expected = self._expectation[idx, context]
        rpe = reward - expected

        # 4) 更新四数组
        # valence：瞬态响应 + 衰减回中性
        self._valence[:P] += rpe * 0.3
        self._valence[:P] *= pcfg.valence_decay
        self._valence[:P] = np.clip(self._valence[:P], -1.0, 1.0)
        # arousal：意外事件（|RPE| 大）→ 高唤醒
        self._arousal[:P] += np.abs(rpe) * 0.2
        self._arousal[:P] *= pcfg.arousal_decay
        self._arousal[:P] = np.clip(self._arousal[:P], 0.0, 1.0)
        # expectation：EWMA 学习（慢半拍，那半拍就是愉悦来源）
        self._expectation[idx, context] += pcfg.alpha * rpe
        self._expectation[:P] = np.clip(self._expectation[:P], 0.0, pcfg.max_reward)
        # baseline：慢漂移（习惯化——长期愉悦/不愉悦会被适应）
        self._baseline[:P] = (
            self._baseline[:P] * (1.0 - pcfg.baseline_rate)
            + self._valence[:P] * pcfg.baseline_rate
        )

    def pleasure_summary(self) -> dict:
        """愉悦度系统的统计快照（给观察者/实验用）。"""
        if len(self._id) == 0:
            return {"valence_mean": 0.0, "arousal_mean": 0.0, "baseline_mean": 0.0}
        return {
            "valence_mean": float(self._valence.mean()),
            "valence_std": float(self._valence.std()),
            "arousal_mean": float(self._arousal.mean()),
            "baseline_mean": float(self._baseline.mean()),
            "expectation_mean": float(self._expectation.mean()),
        }