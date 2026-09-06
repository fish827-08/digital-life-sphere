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
from simulation.tick import TickStats
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
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
        "config", "rng", "world", "light", "resources",
        "_flat", "_energy", "_stomach", "_genes", "_age", "_generation",
        "_parent", "_id", "_next_id", "_max_generation",
        "_repro_cooldown",
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
        )
        # 初始填充覆盖：配置允许指定
        if config.resources.initial_fill != 0.5:
            self.resources.set_initial_fill(config.resources.initial_fill)

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
        if self._use_sim_core:
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
        else:
            self.resources.regrow(self._tick)
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
        P = len(self._id)
        if P == 0:
            self._extinct = True
            return 0, 0, Counter()

        # 温度相关量（一次算出全种群的那份，避免反复调用）
        activity = self.light.activity_factor(self._flat, self._tick)

        genes = self._genes[:P]
        energy = self._energy[:P]
        stomach = self._stomach[:P]

        # 0) 个体化温度响应（恒温基因 g9 + 温度偏好基因 g11）：
        #    冷血（g9=0）完全随环境：低温时消化慢、行动贵；
        #    恒温（g9=1）内部温度恒定：低温不再压制，但每 tick 多扣维持费。
        homeo = genes[:, 9]
        eff_activity = activity + homeo * (1.0 - activity)
        #    温度偏好（g11）：冷血个体只在自己偏好的温度附近才满速，
        #    偏离越远越慢（偏离 15° 活力掉到 ≈6 成）；恒温者不受影响。
        #    偏好温度 = -10 + g11×50，落在 [-10°C, 40°C] 覆盖全地图谱系。
        pref_temp = -10.0 + genes[:, 11] * 50.0
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
            #    赤道正午最多 ≈ photo_max(0.1)，寒侧/极地 ≈ 0 —— 与代谢 0.6 相比明显偏少。
            energy += (
                self.light.illumination(self._flat, self._tick)
                * genes[:, 8]
                * ocfg.photo_max
            )

            # 2) 代谢转化：胃 → 能量（总量不变，快慢受温度+基因影响）
            #    转化速率 = base_metabolism × metabolic_mult × eff_activity
            #    metabolic_mult = 0.5 + g1×1.5（基因放大代谢快慢）
            metab_mult = 0.5 + genes[:, 1] * 1.5
            digest_rate = ocfg.base_metabolism * metab_mult * eff_activity
            # 每 tick 最多转化这么多；不得超出胃里有的
            digest = np.minimum(stomach, digest_rate)
            energy += digest * ocfg.eat_efficiency  # 同样食物 → 同样能量
            stomach -= digest

            # 3) 基础维持消耗（体温/活动，必扣，与温度无关基础价）
            #    随年龄的"需求"阶段：幼体在长身体（growth_mult 倍）→
            #    成年（1 倍）→ 老年器官退化（senile_mult 倍）。
            #    成熟/老年年龄按各自寿命（g3）的比例划分，寿命长的物种成熟和老化都更晚。
            life_span = self._lifespan(genes[:, 3])
            age_f = self._age[:P].astype(np.float64)
            maturity_age = ocfg.maturity_fraction * life_span
            senile_age = ocfg.senile_fraction * life_span
            age_mult = np.ones(P, dtype=np.float64)
            age_mult[age_f < maturity_age] = ocfg.growth_mult
            age_mult[age_f >= senile_age] = ocfg.senile_mult
            energy -= ocfg.base_metabolism * metab_mult * age_mult
            #    + 恒温维持费（g9）：恒温个体每 tick 另付 homeo_upkeep
            energy -= ocfg.homeo_upkeep * homeo

        # 4) 进食：从格子里吃进胃（先吃后扣基础维持，保证当天能吃到）
        #    饱食度：胃容量上限（基础 = max_energy/eat_efficiency/2，g5 缩放 0.5~2 倍）
        #    进食量：每 tick 最多 eat_amount（g4 缩放 0.5~1.5 倍）
        eat_mult = 0.5 + genes[:, 4] * 1.0
        cap_mult = 0.5 + genes[:, 5] * 1.5
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
            hunter = self._genes[hung, 10] >= 0.5
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

        # 5) 移动：raw 抽样在 Python（RNG），能量门槛判定两路径一致（Rust 在 stage2 内）
        if self._use_sim_core:
            moved_raw = self.rng.random(P) < genes[:, 0]
            move_cost_ind = ocfg.move_cost * cold_penalty * (0.5 + genes[:, 6])
            out_moved = np.empty(P, dtype=bool)
            out_starved = np.empty(P, dtype=bool)
            out_expired = np.empty(P, dtype=bool)
            out_repro = np.empty(P, dtype=bool)
            self._sim_core.step_vectors_stage2(
                energy, self._age[:P], self._repro_cooldown[:P], genes,
                moved_raw, move_cost_ind,
                out_moved, out_starved, out_expired, out_repro,
                day_len, ocfg.maturity_fraction, ocfg.max_energy,
            )
            moved = out_moved
        else:
            moved = self.rng.random(P) < genes[:, 0]
            move_cost_ind = ocfg.move_cost * cold_penalty * (0.5 + genes[:, 6])
            moved &= energy >= move_cost_ind  # 付得起才走
        Nm = int(moved.sum())
        if Nm:
            mi = np.flatnonzero(moved)
            # 群居性（g13）：g>0.5 朝同伴多的邻格走（聚群），
            # g<0.5 专挑冷清邻格走（避群），g≈0.5 或单邻格则随机。
            densities = np.bincount(self._flat, minlength=self.world.n_cells)
            targets = np.empty(Nm, dtype=np.int64)
            for i, idx in enumerate(mi):
                nb = self.world.neighbors(int(self._flat[idx]))
                soc = genes[idx, 13]
                if len(nb) == 1 or abs(soc - 0.5) < 0.1:
                    targets[i] = nb[int(self.rng.integers(0, len(nb)))]
                elif soc > 0.5:
                    targets[i] = nb[int(np.argmax(densities[np.asarray(nb)]))]
                else:
                    targets[i] = nb[int(np.argmin(densities[np.asarray(nb)]))]
            self._flat[mi] = targets
            if not self._use_sim_core:
                energy[mi] -= move_cost_ind[mi]  # Rust 路径已在 stage2 内扣除

        # 6) 年龄推进（Rust 路径已由 stage2 就地 +1）
        if not self._use_sim_core:
            self._age[:P] += 1

        # 7) 死亡判定：饿死（energy<=0）→ 老死（age>=寿命）
        if self._use_sim_core:
            starved, expired = out_starved, out_expired
        else:
            starved = energy <= 0.0
            life_span = self._lifespan(genes[:, 3])
            expired = (
                (~starved) & (self._age[:P].astype(np.float64) >= life_span)
            )
        dead = starved | expired
        deaths: Counter = Counter()
        n_starved = int(starved.sum())
        n_expired = int(expired.sum())
        if n_starved:
            deaths[DeathCause.STARVATION] = n_starved
        if n_expired:
            deaths[DeathCause.OLD_AGE] = n_expired

        # 8) 繁殖：冷却期（g12）倒数；能量 ≥ 阈值（0.25+g2×0.65），且种群未满
        if self._use_sim_core:
            repro = out_repro  # 冷却倒数与成熟判定已在 Rust stage2 内完成
        else:
            self._repro_cooldown[:P] = np.maximum(
                0.0, self._repro_cooldown[:P] - 1.0
            )
            repro_thr = (0.25 + genes[:, 2] * 0.65) * ocfg.max_energy
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
            split = 0.3 + genes[ri, 7] * 0.4
            child_energy = energy[ri] * split
            child_stomach = stomach[ri] * split
            energy[ri] -= child_energy
            stomach[ri] -= child_stomach
            # 生完进入冷却（g12）：间隔 = g12 × 60 tick，冷却没到攒再多也不生
            self._repro_cooldown[ri] = genes[ri, 12] * 60.0

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
            born = K
            new_max = int(self._generation.max())
            if new_max > self._max_generation:
                self._max_generation = new_max

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

        if len(self._id) == 0:
            self._extinct = True
        return born, n_starved + n_expired, Counter(deaths)