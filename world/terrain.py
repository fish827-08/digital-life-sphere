"""TerrainField：球面世界的地形系统（L8 简化版）。

模块职责
--------
在 SphereWorld 拓扑上叠加静态地形类型，提供空间异质性的第二个维度
（第一个是 L1 斑块资源）：
- 平原（默认）：正常移动、正常资源再生
- 山地：移动能耗 × mountain_move_mult，资源再生 × mountain_regrow_mult
- 水域：不可通行（移动目标过滤掉水域格）

简化设计（与 L8-L10 文档的完整版对比）：
- 不用值噪声，用"纬度带偏置 + 随机斑块"生成，实现简单且可复现
- 只实现 3 种地形（平原/山地/水域），不做森林/沙漠/温度偏移
- 地形是静态的（世界生成时固定，不随时间变化）
- 地形生成用独立 rng（patch_seed 派生），不消费引擎 self.rng

地形类型常量
------------
TERRAIN_PLAIN = 0
TERRAIN_MOUNTAIN = 1
TERRAIN_WATER = 2
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld

TERRAIN_PLAIN: int = 0
TERRAIN_MOUNTAIN: int = 1
TERRAIN_WATER: int = 2


class TerrainField:
    """静态地形场：每格一个地形类型，提供可通行性和倍率查询。

    实例属性
    --------
    _terrain : NDArray[uint8], 形状 (n_cells,)
        每格地形类型（0=平原, 1=山地, 2=水域）。
    _passable : NDArray[bool], 形状 (n_cells,)
        可通行格（水域=False，其他=True）。
    _move_mult : NDArray[float64], 形状 (n_cells,)
        每格移动能耗倍率（平原=1.0，山地=mountain_move_mult，水域=1.0占位但不可通行）。
    _regrow_mult : NDArray[float64], 形状 (n_cells,)
        每格资源再生倍率（平原=1.0，山地=mountain_regrow_mult，水域=0.0）。
    """

    __slots__ = (
        "world",
        "_terrain",
        "_passable",
        "_move_mult",
        "_regrow_mult",
    )

    def __init__(
        self,
        world: SphereWorld,
        water_ratio: float = 0.08,
        mountain_ratio: float = 0.12,
        mountain_move_mult: float = 2.0,
        mountain_regrow_mult: float = 0.3,
        water_pole_bias: float = 0.5,
        seed: int = 42,
    ) -> None:
        """生成地形场。

        参数
        ----
        world : SphereWorld
            所属网格。
        water_ratio : float
            水域目标占比。
        mountain_ratio : float
            山地目标占比。
        mountain_move_mult : float
            山地移动能耗倍率。
        mountain_regrow_mult : float
            山地资源再生倍率。
        water_pole_bias : float
            极地水域概率加成（0=均匀随机，1=极地水域概率×2）。
        seed : int
            地形生成随机种子（独立 rng，不消费引擎 rng）。
        """
        self.world = world
        n_cells = world.n_cells
        rng = np.random.default_rng(seed)

        # 第一步：生成水域（极地偏置）
        # 计算每格的纬度绝对值（0=赤道，1=极点），用于极地偏置
        rows = np.arange(n_cells) // world.cols
        lat_abs = np.abs(rows / (world.rows - 1) * 2.0 - 1.0)  # 0=赤道, 1=极点
        water_prob = water_ratio * (1.0 + water_pole_bias * lat_abs)
        water_prob = np.clip(water_prob, 0.0, 0.9)
        water_mask = rng.random(n_cells) < water_prob

        # 第二步：在非水域格中生成山地
        non_water = ~water_mask
        mountain_prob = np.full(n_cells, mountain_ratio)
        mountain_mask = non_water & (rng.random(n_cells) < mountain_prob)

        # 组装地形数组
        self._terrain = np.zeros(n_cells, dtype=np.uint8)
        self._terrain[mountain_mask] = TERRAIN_MOUNTAIN
        self._terrain[water_mask] = TERRAIN_WATER

        # 派生查询数组
        self._passable = self._terrain != TERRAIN_WATER
        self._move_mult = np.ones(n_cells, dtype=np.float64)
        self._move_mult[mountain_mask] = mountain_move_mult
        self._regrow_mult = np.ones(n_cells, dtype=np.float64)
        self._regrow_mult[mountain_mask] = mountain_regrow_mult
        self._regrow_mult[water_mask] = 0.0

    @property
    def terrain(self) -> NDArray[np.uint8]:
        """每格地形类型（只读）。"""
        return self._terrain

    @property
    def passable(self) -> NDArray[np.bool_]:
        """每格可通行性（水域=False）。"""
        return self._passable

    @property
    def move_mult(self) -> NDArray[np.float64]:
        """每格移动能耗倍率。"""
        return self._move_mult

    @property
    def regrow_mult(self) -> NDArray[np.float64]:
        """每格资源再生倍率（水域=0）。"""
        return self._regrow_mult

    def is_passable(self, flat: int | NDArray[np.int64]) -> bool | NDArray[np.bool_]:
        """查询指定格是否可通行。"""
        return self._passable[flat]

    def move_cost_at(self, flat: int | NDArray[np.int64]) -> float | NDArray[np.float64]:
        """查询指定格的移动能耗倍率。"""
        return self._move_mult[flat]

    def stats(self) -> dict[str, float]:
        """地形统计：各地形占比。"""
        n = self.world.n_cells
        return {
            "plain_ratio": float((self._terrain == TERRAIN_PLAIN).sum()) / n,
            "mountain_ratio": float((self._terrain == TERRAIN_MOUNTAIN).sum()) / n,
            "water_ratio": float((self._terrain == TERRAIN_WATER).sum()) / n,
            "passable_ratio": float(self._passable.sum()) / n,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"TerrainField(plain={s['plain_ratio']:.1%}, "
            f"mountain={s['mountain_ratio']:.1%}, water={s['water_ratio']:.1%})"
        )
