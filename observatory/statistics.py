"""observatory.statistics：观察层纯函数统计聚合（模块四 · 适配版）。

职责
----
- 输入：一台球面引擎（SphereEngine，鸭子类型的最小表面）；
- 输出：一个"观测点"的全部统计量（trait 分布 / 基因组多样性 / 谱系）。
  无副作用、无内部状态 → 同一引擎状态必然同一输出，可复现、可对比。

与旧版（digital_life/observatory/statistics.py）的差异
-----------------------------------------------------
旧版从"Organism 对象列表"聚合（o.age / o.genome.genes ...）；
本版引擎是数组化（SoA）存储，改为**直接从引擎内部数组聚合**：
engine._age / _energy / _generation / _genes，零对象创建、O(N) numpy。

统计口径（列名显式声明，与旧版保持一致）：
- trait 分布：均值 / 标准差（TRAIT_ORDER 见 observatory/traits.py）
- 基因组多样性：平均每位点离散杂合度（Simpson 型，0=单态，1=最大）
- 谱系：世代号分布（存活个体）、世代号中位数/均值
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from observatory.traits import TRAIT_ORDER, decode_trait_matrix

# 基因组多样性：基因值离散化箱数（[gene_min, gene_max) 均分）
_HET_BINS = 16
# 唯一基因型判定：离散化精度（小数点后位数）
_UQ_DECIMALS = 3
# 世代直方图的固定对数桶上界（世代号 → "0-9","10-99",...）
_GEN_BIN_EDGES = (0, 1, 10, 100, 1000, 10_000, 100_000)


@dataclass(frozen=True)
class GenerationStats:
    """一个观测点的聚合统计（全部字段 JSON 安全）。"""

    population: int
    mean_age: float
    mean_energy: float
    max_generation: int
    median_generation: float
    mean_generation: float
    generation_histogram: dict[str, int]  # 固定对数桶 {"0-0": n, "1-9": n, ...}
    trait_means: dict[str, float]
    trait_stds: dict[str, float]
    genome_diversity: float  # 平均每位点杂合度
    unique_genotypes: int  # 量化后不同基因型个数
    unique_genotype_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_stats() -> GenerationStats:
    return GenerationStats(
        population=0,
        mean_age=0.0,
        mean_energy=0.0,
        max_generation=-1,
        median_generation=0.0,
        mean_generation=0.0,
        generation_histogram={},
        trait_means={t: 0.0 for t in TRAIT_ORDER},
        trait_stds={t: 0.0 for t in TRAIT_ORDER},
        genome_diversity=0.0,
        unique_genotypes=0,
        unique_genotype_ratio=0.0,
    )


def generation_statistics(engine) -> GenerationStats:
    """从引擎当前存活种群聚合一个观测点。空种群返回全零（合法观测点）。"""
    n = len(engine._id)
    if n == 0:
        return _empty_stats()

    ages = np.asarray(engine._age, dtype=np.float64)
    energies = np.asarray(engine._energy, dtype=np.float64)
    gens = np.asarray(engine._generation, dtype=np.int64)
    genes = np.asarray(engine._genes)  # (n, gene_count)

    # ---- trait 分布（表现型，来自基因解码） ----
    day = float(engine.config.light.rotation_period)
    traits = decode_trait_matrix(genes, day)  # (n, n_traits)

    # ---- 基因组多样性：每位点离散化 → Simpson 杂合度 ----
    binned = np.floor(genes * _HET_BINS).clip(0, _HET_BINS - 1).astype(np.int64)
    het_per_locus = []
    for locus in range(binned.shape[1]):
        counts = np.bincount(binned[:, locus], minlength=_HET_BINS).astype(np.float64)
        probs = counts / n
        het_per_locus.append(1.0 - float((probs**2).sum()))
    diversity = float(np.mean(het_per_locus)) if het_per_locus else 0.0

    # ---- 唯一基因型 ----
    unique_count = len(np.unique(np.round(genes, _UQ_DECIMALS), axis=0))

    # ---- 世代直方图（对数桶） ----
    edges = np.array(_GEN_BIN_EDGES, dtype=np.int64)
    indices = np.clip(
        np.searchsorted(edges, gens, side="right") - 1, 0, len(edges) - 2
    )
    hist: dict[str, int] = {}
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1]) - 1
        hist[f"{lo}-{hi}"] = int((indices == i).sum())

    return GenerationStats(
        population=n,
        mean_age=float(ages.mean()),
        mean_energy=float(energies.mean()),
        max_generation=int(gens.max()),
        median_generation=float(np.median(gens)),
        mean_generation=float(gens.mean()),
        generation_histogram=hist,
        trait_means={t: float(traits[:, i].mean()) for i, t in enumerate(TRAIT_ORDER)},
        trait_stds={t: float(traits[:, i].std()) for i, t in enumerate(TRAIT_ORDER)},
        genome_diversity=diversity,
        unique_genotypes=int(unique_count),
        unique_genotype_ratio=float(unique_count / n),
    )

# ---- D-16：R31③/R38③ 要求入 CSV 的两个判据列（单一实现，供所有 runner 调用） ----
# （D-15 合并说明：分支 D-4 也实现过 codebook_convergence；保留 main 版——
#   含 `arbitrary_codebook=False` 恒 1.0 无判别力语义（内评 N3），为准。）

def codebook_convergence(codebook) -> float:
    """群体码本趋同度（0~1，1=完全一致）。

    对每个 state(0..15)，取群体中最常见映射的占比，再对 16 个 state 求均值。
    含义：**码本是否已收敛成"公共词典"**——这是"任意性码本"这一条件的可观测量。
    ⚠️ `arbitrary_codebook=False` 时码本是恒等映射，本指标**恒为 1.0（常数）**，
       不具判别力（内评 N3），故必须开码本才有意义。
    """
    import numpy as np

    if codebook is None or len(codebook) == 0:
        return 0.0
    conv = []
    for state in range(16):
        mappings = np.asarray(codebook)[:, state].astype(int)
        most_common = int(np.bincount(mappings, minlength=16).max())
        conv.append(most_common / len(mappings))
    return float(np.mean(conv))


# ============================================================================
# D2 信息结构指标（C4 口径统一，D-15 自 feat/cloud-d1-d9-main 合并；唯一权威实现）
# ============================================================================

def gene_mean(engine, locus: int) -> float:
    """基因位点均值（g14=locus14, g15=locus15）。空种群返回 0.0。"""
    n = len(engine._id)
    if n == 0:
        return 0.0
    return float(np.asarray(engine._genes)[:, locus].mean())


def trust_mean(engine) -> float:
    """trust 均值。空种群返回 0.0。"""
    n = len(engine._id)
    if n == 0:
        return 0.0
    return float(np.asarray(engine._trust).mean())


def learning_count_max(engine) -> int:
    """最大学习计数（lc_max）。空种群返回 0。"""
    n = len(engine._id)
    if n == 0:
        return 0
    return int(np.asarray(engine._learning_count).max())


def predation_fraction(deaths) -> float:
    """捕食死亡占全部死亡的比例（0~1；无死亡时为 0.0）。

    R38③：用于**按捕食占死因分层**——饱和封顶区制与捕食主导区制不能直接合并比较。
    `deaths` 为 `engine.death_cause_totals()` 之类的 {原因: 计数} 映射。
    """
    if not deaths:
        return 0.0
    total = 0
    predation = 0
    for cause, n in deaths.items():
        n = int(n)
        total += n
        if str(cause).endswith("PREDATION"):
            predation += n
    return float(predation / total) if total else 0.0


# ---- D-17：⑤ 个体层选择梯度（R42 / V-7 §2.2 口径，单一实现） ----------------

def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），供 Spearman 用；纯 numpy，无 scipy 依赖。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _fit_selection(g15, age, energy, fitness) -> dict:
    """一个 cohort 的 ⑤ 拟合：OLS 斜率（控年龄+能量）+ Spearman ρ（未控）。"""
    n = int(len(g15))
    out = {
        "n": n,
        "slope_g15": None,
        "spearman_rho": None,
        "mean_g15": round(float(g15.mean()), 6) if n else None,
        "mean_fitness": round(float(fitness.mean()), 6) if n else None,
    }
    if n < 10 or float(np.ptp(g15)) < 1e-12 or float(np.ptp(fitness)) < 1e-12:
        return out  # 样本不足或零方差 ⇒ 不可估，返回 None（G-D 判"是否全 NaN"用）
    X = np.column_stack([np.ones(n), g15, age, energy])
    coef, *_ = np.linalg.lstsq(X, fitness, rcond=None)
    rg, rf = _rankdata(g15), _rankdata(fitness)
    rg = rg - rg.mean()
    rf = rf - rf.mean()
    denom = float(np.sqrt((rg**2).sum() * (rf**2).sum()))
    rho = float((rg * rf).sum() / denom) if denom > 0 else 0.0
    out["slope_g15"] = round(float(coef[1]), 6)
    out["spearman_rho"] = round(rho, 6)
    return out


def selection_gradient(engine) -> dict:
    """D-17 ⑤：个体层选择梯度 `g15 → 剩余终身繁殖数`（R42 / V-7 §2.2）。

    被估量与口径（预注册，不得事后改）
    --------------------------------
    - **观测单元**：每个 `_id` 在其**首次存活且种群处于某窗口**的 tick 记一行
      （g15 / 年龄 / 能量 / 当时已生子代数 cc0）。
    - **被解释变量**：剩余终身繁殖数 = `_rs_children[id] − cc0`；
      **死亡个体行永久保留**（`_id` 键控数组不随死亡压缩）⇒ 自动满足 R42
      "必须含死亡个体"；run 末仍存活者为右删失（均匀删失，仪器层可接受）。
    - **分层**：非饱和窗（`P < 0.9×max_count`）= 主口径 `non_sat`；
      饱和窗 = **诊断** `sat_diagnostic`（该窗内繁殖是抽签——内评 V-7 G-2——
      诊断预期给出假阴性斜率，两窗**不得合并**）。
    - **估计**：OLS `fitness ~ 1 + g15 + age + energy` 取 g15 斜率（控年龄+能量，
      V-7 §2.2）+ Spearman ρ（未控，辅助方向）。n<10 或零方差 ⇒ None。
    - ⚠️ ⑤ 是**个体层选择梯度**，不是群体均值会涨（V-7 附注）；G-D 只要求
      "非全 NaN 且方向可读"。
    """
    g15 = np.asarray(engine._rs_g15, dtype=np.float64)
    age = np.asarray(engine._rs_age, dtype=np.float64)
    energy = np.asarray(engine._rs_energy, dtype=np.float64)
    children = np.asarray(engine._rs_children, dtype=np.float64)
    cc0 = np.asarray(engine._rs_cc0, dtype=np.float64)
    observed = np.asarray(engine._rs_observed, dtype=bool)
    cohort = np.asarray(engine._rs_cohort)
    fit = children - cc0
    return {
        "non_sat": _fit_selection(
            g15[observed & (cohort == 0)],
            age[observed & (cohort == 0)],
            energy[observed & (cohort == 0)],
            fit[observed & (cohort == 0)],
        ),
        "sat_diagnostic": _fit_selection(
            g15[observed & (cohort == 1)],
            age[observed & (cohort == 1)],
            energy[observed & (cohort == 1)],
            fit[observed & (cohort == 1)],
        ),
    }
@dataclass(frozen=True)
class D2Metrics:
    """D2 实验的一个观测点（C4 统一口径，字段与 CSV 列名一致）。"""
    tick: int
    N: int
    g14: float
    g15: float
    trust: float
    max_gen: int
    lc_max: int
    codebook_conv: float

    def to_dict(self) -> dict:
        return asdict(self)


def d2_metrics(engine, tick: int) -> D2Metrics:
    """从引擎当前状态一次性计算全部 D2 指标（唯一权威入口）。

    所有实验脚本应调用本函数，不得在脚本中重复计算 g14/g15/trust 等。
    """
    n = len(engine._id)
    return D2Metrics(
        tick=int(tick),
        N=n,
        g14=gene_mean(engine, 14),
        g15=gene_mean(engine, 15),
        trust=trust_mean(engine),
        max_gen=max_generation_current(engine),
        lc_max=learning_count_max(engine),
        codebook_conv=codebook_convergence(np.asarray(engine._codebook)) if n > 0 else 0.0,
    )


# ============================================================================
# R77：`max_gen` 口径化 —— 两个口径必须分名，禁止再用裸 `max_gen`
# ============================================================================

def max_generation_current(engine) -> int:
    """**当刻最深**世代 = 当前存活个体的最大 `_generation`（空种群 ⇒ -1）。

    与 `max_generation_highwater` 的区别（R77 立规的原因）：本函数只看**活人**，
    谱系灭绝后它会**回落**；高水位不会。实验脚本若用同名 `max_gen` 混装两者，
    跨脚本比较必然出错（D-24 实测：`d2_metrics` 用当刻口径、`a4_verify_capacity`
    用高水位口径，却是同一个列名）。
    """
    n = len(engine._id)
    if n == 0:
        return -1
    return int(np.asarray(engine._generation)[:n].max())


def max_generation_highwater(engine) -> int:
    """**历史高水位**世代 = 引擎 `_max_generation`（含已死者，永不回落）。

    ⚠️ 它回答的是"**曾经**出现过第几代"，不是"现在最深第几代"。内评
    `_eval/D24判读预析` §4.4 引用的 `max_gen`=117 属本口径。
    """
    return int(engine._max_generation)
