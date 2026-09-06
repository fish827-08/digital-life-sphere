"""observatory.traits：观察台统计口径的 trait 表（模块四）。

通俗理解
--------
观察台要回答"这一代生物长什么样"：把每个个体的一串基因（g0~g13）
翻译成一排"性状列"，连同几个派生指标（寿命/代谢倍率/成熟年龄）一起
放进统计表，之后均值/标准差就按这些列算。

为什么不叫"基因表"而叫 trait 表
-------------------------------
- 基因位是 0~1 的连续原始值（变异单位）；
- trait 是"人看得懂的表现型语义"：移动概率、代谢快慢、寿命多长……
  同一个基因位在 SETA 引擎里可能被多个行为读取，trait 名按它最主要的
  语义起名，不新增任何机制（纯统计口径，不参与演化）。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 14 个基因位的语义名（与引擎 _genes 列一一对应；索引即基因序号）。
GENE_TRAIT_NAMES: tuple[str, ...] = (
    "move_prob",          # g0  移动概率
    "metabolic",          # g1  代谢倍率（消化快慢）
    "repro_threshold",    # g2  繁殖能量门槛
    "life_gene",          # g3  寿命基因（影响 lifespan 派生）
    "eat_amount",         # g4  进食量
    "stomach_cap",        # g5  胃容量
    "move_cost",          # g6  移动能耗
    "parental_invest",    # g7  传代投入（分给子代的能量比例）
    "photosynthesis",     # g8  光合产能
    "homeotherm",         # g9  恒温性
    "forage_neighbor",    # g10 邻格觅食倾向
    "temp_pref",          # g11 温度偏好
    "repro_cooldown",     # g12 繁殖冷却长度
    "sociability",        # g13 群居性
)

# 派生 trait：由基因 + 环境基准（一昼夜 tick 数）解码，非独立基因位。
# 每个派生 trait = (名字, 解码函数)。函数输入 (genes_2d, day_length)。
_DERIVED_TRAITS: tuple[tuple[str, ...], ...] = (
    # 寿命（tick）＝一昼夜 × (1 + g3×7)：与引擎 _lifespan() 同一公式。
    ("life_span", lambda g, day: day * (1.0 + g[:, 3] * 7.0)),
    # 代谢倍率 ＝ 0.5 + g1×1.5：消化/维持按它缩放（与引擎 stage1 同一公式）。
    ("metabolic_mult", lambda g, day: 0.5 + g[:, 1] * 1.5),
    # 成熟年龄 ＝ 寿命 × 成熟比例（0.15）：达到才能繁衍。
    ("maturity_age", lambda g, day: 0.15 * day * (1.0 + g[:, 3] * 7.0)),
)

TRAIT_ORDER: tuple[str, ...] = GENE_TRAIT_NAMES + tuple(
    name for name, _ in _DERIVED_TRAITS
)


def decode_trait_matrix(
    genes: NDArray[np.float64], day_length: float
) -> NDArray[np.float64]:
    """把一整群个体的基因解码为 (n, n_traits) 表现型矩阵（观察台专用）。

    参数
    ----
    genes : NDArray, 形状 (n, 16)
        引擎的基因链（每行一个个体，前 14 位参与 trait 解码）。
    day_length : float
        一昼夜的 tick 数（light.rotation_period），派生 trait 的基准。

    返回
    ----
    NDArray, 形状 (n, len(TRAIT_ORDER))
        按 TRAIT_ORDER 排列的 trait 值矩阵（基因位原样 + 派生列）。
    """
    n = genes.shape[0]
    cols: list[NDArray[np.float64]] = [
        genes[:, i].astype(np.float64) for i in range(len(GENE_TRAIT_NAMES))
    ]
    for _, fn in _DERIVED_TRAITS:
        cols.append(np.asarray(fn(genes, day_length), dtype=np.float64))
    return np.stack(cols, axis=1)