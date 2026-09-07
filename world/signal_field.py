"""田字格信号场：每格 4 子格 0/1 = 16 种标记模式。

语言涌现的信号载体。生物可在当前格写入标记（耗能，由引擎扣），标记持续
duration tick 后自动消失。其他生物经过时可读取标记，获取前者留下的信息。

16 种模式作为字母表绰绰有余（人类音素最少的 Rotokas 语仅 11 个），
田字格的 4 个位置提供词内结构（槽位语法），高阶语言需文化积累涌现组合规则。

存储：uint8 低 4 位（bit0~bit3 对应 4 个子格），0 = 无标记。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld


class SignalField:
    __slots__ = (
        "world",
        "duration",
        "_marks",
        "_age",
    )

    def __init__(self, world: SphereWorld, duration: int = 50) -> None:
        """构造信号场。

        参数
        ----
        world : SphereWorld
            球面世界（决定格子数）。
        duration : int
            标记持续多少 tick 后自动消失。默认 50。
        """
        self.world = world
        self.duration = int(duration)
        self._marks = np.zeros(world.n_cells, dtype=np.uint8)
        self._age = np.zeros(world.n_cells, dtype=np.int32)

    # ---- 写入 ----------------------------------------------------------------

    def write(self, cell: int, pattern: int) -> None:
        """在指定格写入标记模式（0~15），覆盖已有标记，重置寿命。

        pattern=0 等同于清除标记。
        """
        self._marks[cell] = np.uint8(pattern & 0x0F)
        self._age[cell] = self.duration if (pattern & 0x0F) else 0

    def write_many(
        self, cells: NDArray[np.int64], patterns: NDArray[np.uint8]
    ) -> None:
        """批量写入（向量化）。cells 和 patterns 等长。"""
        p = patterns & 0x0F
        self._marks[cells] = p
        self._age[cells] = np.where(p > 0, self.duration, 0).astype(np.int32)

    # ---- 读取 ----------------------------------------------------------------

    def read(self, cell: int) -> int:
        """读取指定格的当前标记模式（0=无标记）。"""
        return int(self._marks[cell])

    def read_many(self, cells: NDArray[np.int64]) -> NDArray[np.uint8]:
        """批量读取，返回副本。"""
        return self._marks[cells].copy()

    # ---- 时间推进 ------------------------------------------------------------

    def tick(self) -> None:
        """推进一个 tick：所有标记年龄-1，过期清零。"""
        active = self._age > 0
        self._age[active] -= 1
        expired = active & (self._age <= 0)
        self._marks[expired] = 0
        self._age[expired] = 0

    # ---- 工具 ----------------------------------------------------------------

    def clear(self, cell: int) -> None:
        """清除指定格的标记。"""
        self._marks[cell] = 0
        self._age[cell] = 0

    def snapshot(self) -> NDArray[np.uint8]:
        """返回二维快照 (rows, cols)，用于可视化。"""
        return self._marks.reshape(self.world.rows, self.world.cols).copy()

    def active_count(self) -> int:
        """当前有标记的格子数。"""
        return int(np.count_nonzero(self._marks))

    def total_marks(self) -> int:
        """所有标记的子格点亮总数（用于统计信号密度）。"""
        return int(np.unpackbits(self._marks).sum())
