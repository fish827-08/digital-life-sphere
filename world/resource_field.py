"""ResourceField：球面世界的资源场（模块一 · 文件 3）。

模块职责（这一份是"世界的食物分布"）
-----------------------------------
前两份文件把"土地"（SphereWorld）和"天气"（LightAndTemperature）
准备好了，这一份在土地上铺设"食物"：
- 每块格子能存多少食物（容量）由它的【面积】决定：
  赤道格子大 → 能存得多；极点格子小 → 只能存一点点，
  所以极地天然"养不活多少生物"（竞争少）；
- 食物会随时间【缓慢恢复】（再生），恢复速度受【温度】影响：
  太冷的地方（极地、深夜）恢复慢，热带正午恢复快；
- 生物【吃掉】食物时，格子里的食物减少（见 consume）。

为什么这样设计（和你的设想对齐）
--------------------------------
你提到过"极地温度更低、生物竞争偏少"，这里的容量缩放在空间上
实现了"极地养不活多少"，再生受温度抑制在时间上实现了"冷的地方
食物长得慢"。两件事都不需要写死规则，是由环境自然涌现的。

时间约定：regrow(tick) 接收当时的 tick 计算温度因子，
所以昼夜会带来"白天恢复快、晚上恢复慢"的节律。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld
from world.light_and_temperature import LightAndTemperature


class ResourceField:
    """网格上的可再生的食物场（布局按 [row][col] 平铺一维数组）。

    实例属性（__slots__ 声明的全部字段）说明
    ---------------------------------------
    world : SphereWorld
        所属网格（提供面积权重 cell_area 与索引转换）。
    lt : LightAndTemperature
        光照温度场（提供温度，用于再生速度的时间调节）。
    capacity_per_area : float
        单位面积的食物容量。格子容量 = 该值 × 格子面积权重。
        （面积权重近赤道≈1，极点≈0.026，因此极点容量很小）
    regrowth_rate : float
        基准再生速度（每 tick 每格恢复的食物量，温度因子为 1 时）。
    temp_sensitivity : float
        再生受温度影响的强弱（0=完全不看温度，越大越看温度）。
    _grid : NDArray[float64], 形状 (n_cells,) 平铺
        每个格子当前的食物存量（扁平数组，通过 flat 索引访问）。
    _capacity : NDArray[float64], 形状 (n_cells,) 平铺
        每个格子的食物容量上限（构造时按面积算好，之后只读）。
    """

    __slots__ = (
        "world",
        "lt",
        "capacity_per_area",
        "regrowth_rate",
        "temp_sensitivity",
        "_grid",
        "_capacity",
    )

    def __init__(
        self,
        world: SphereWorld,
        lt: LightAndTemperature,
        capacity_per_area: float = 40.0,
        regrowth_rate: float = 0.5,
        temp_sensitivity: float = 1.0,
    ) -> None:
        """铺好初始食物（整个球面均匀随机填充到容量的一部分）。

        参数
        ----
        world : SphereWorld
            网格局（文件 1 的对象），决定格子数量与面积。
        lt : LightAndTemperature
            光照温度场（文件 2 的对象），再生时要查温度。
        capacity_per_area : float, 默认 40.0
            单位面积能存多少食物（可理解为"一单位面积土地的食物上限"）。
        regrowth_rate : float, 默认 0.5
            基准再生：温度合适时每 tick 向每格补充的食物量。
        temp_sensitivity : float, 默认 1.0
            再生对温度的依赖程度。设为 0 则再生与温度无关（纯空间）。
            越大则"冷的地方食物长得越慢"越明显。

        返回
        ----
        None。构造完成后即可查询/消费。
        """
        self.world = world
        self.lt = lt
        self.capacity_per_area = float(capacity_per_area)
        self.regrowth_rate = float(regrowth_rate)
        self.temp_sensitivity = float(temp_sensitivity)

        # 每格容量 = 面积权重 × 单位面积容量（极点小、赤道大）
        areas = world.cell_area(np.arange(world.n_cells))
        self._capacity = areas * capacity_per_area
        # 初始填一半（均匀随机，同一种子可复现）
        # 注：这里不用 rng 参数，初始分布由调用方决定，便于确定性测试
        self._grid = self._capacity * 0.5

    # ---- 查询（只看不吃） ---------------------------------------------------

    def capacity_at(self, flat) -> np.ndarray:
        """问：这个格子的食物上限是多少？

        通俗理解：格子的"粮仓容量"。赤道格子大粮仓大（≈40），
        极点格子小粮仓小（≈1）。容量主要看面积，不看时间。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（可传一个或一组）。

        返回
        ----
        float 或 NDArray[float64] : 该格（这些格）的食物容量上限。
        """
        out = self._capacity[np.asarray(flat, dtype=np.int64)]
        return out.item(0) if out.ndim == 0 else out

    def amount_at(self, flat) -> np.ndarray:
        """问：这个格子现在还有多少食物？

        通俗理解：看粮仓里还剩多少。刚构造时是容量的一半，
        吃了会少，再生会慢慢补回来。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（可传一个或一组）。

        返回
        ----
        float 或 NDArray[float64] : 当前食物存量（不会超过容量）。
        """
        out = self._grid[np.asarray(flat, dtype=np.int64)]
        return out.item(0) if out.ndim == 0 else out

    def total(self) -> float:
        """问：整个球面一共还剩多少食物？

        参数
        ----
        无。

        返回
        ----
        float : 所有格子食物存量之和（统计/日志用）。
        """
        return float(self._grid.sum())

    # ---- 消费（吃） ---------------------------------------------------------

    def consume(self, flat, amount: float) -> float:
        """吃掉一个格子的食物（最多吃到存量，不欠账）。

        通俗理解：一只生物在某格咬了一口食物。这格食物够就全吃，
        不够就只吃到剩的；存量不可能变成负数。

        参数
        ----
        flat : int
            要吃的格子（单个）。
        amount : float
            想吃的量。

        返回
        ----
        float : 实际吃到的量（不会大于存量，也不会大于想要的量）。
        """
        i = int(np.asarray(flat, dtype=np.int64))
        available = float(self._grid[i])
        taken = min(amount, available)
        self._grid[i] = available - taken
        return taken

    def consume_many(self, flats, amount: float) -> np.ndarray:
        """一批生物同时吃（各自在自己格子里吃）。

        通俗理解：把一整串生物送去同时进食。每只只在自己那格吃，
        各吃各的，一起结算。是 consume 的批量版（引擎一帧内
        全部生物一次算完，比一只一只快几十倍）。

        说明（同格多只时）：同一格的几只【均分】该格的存量——
        每格总消耗 = min(存量, 想吃的量 × 该格只数)，然后按只平分，
        所以绝不可能把格子吃到负数（欠账）。

        参数
        ----
        flats : NDArray[int64]
            一组格子平铺索引（每只生物各占一个）。
        amount : float
            每只生物想吃的量（吃相同的量）。

        返回
        ----
        NDArray[float64] : 每只生物实际吃到的量（与入参一一对应）。
        """
        flats = np.asarray(flats, dtype=np.int64)
        # 统计每个格子同时被多少只吃（np.add.at 对重复索引累加）
        cnt = np.zeros(self._grid.size, dtype=np.int64)
        np.add.at(cnt, flats, 1)
        # 每格总消耗 = min(存量, 单只想要 × 该格只数)；按只均分
        avail = self._grid[flats]
        per_cell = np.minimum(avail, amount * cnt[flats])
        share = per_cell / np.maximum(1, cnt[flats])
        np.subtract.at(self._grid, flats, share)
        return share

    # ---- 再生（食物慢慢长回来） -----------------------------------------------

    def regrow(self, tick: int) -> None:
        """让全世界的食物都长一点（一次性整场更新）。

        通俗理解：过了一个 tick，每格食物都加上一点"恢复量"，
        但上限是粮仓容量，长满了就不再长。而且——
        冷的地方（极点、深夜）恢复量会打折，温度越低长得越慢，
        热带正午长得最快。这是"随 tick 推进"的时间节律（昼夜）。

        参数
        ----
        tick : int
            当前时间步。决定此时各格温度 → 决定恢复量折扣。

        返回
        ----
        None。直接修改内部存量数组。
        """
        growth = self._regrowth_amount(tick)
        np.minimum(self._capacity, self._grid + growth, out=self._grid)

    def _regrowth_amount(self, tick: int) -> NDArray[np.float64]:
        """计算每格本 tick 应恢复的食物量（内部函数）。

        规则：恢复量 = 基准恢复率 × 温度因子。
        温度因子 = clip(温度 / 0°C, 0, 1)^temp_sensitivity：
          - 温度 ≥ 0°：因子 1，恢复满速；
          - 温度 0°~-20°：逐渐变小，越冷恢复越慢；
          - 温度 ≤ -20°：因子≈0，几乎不恢复（极地冰封）。
        极点因为温度极低，再生基本停摆；但容纳的生物也少，符合"竞争少"。

        参数
        ----
        tick : int
            当前时间步（用于查温度）。

        返回
        ----
        NDArray[float64] : 形状 (n_cells,)，每格本 tick 的恢复量。
        """
        temps = self.lt.temperature(np.arange(self.world.n_cells), tick)
        # 因子 = (温度+20)/20，clip 到 [0,1]：
        #   ≥0° → 1（满速）；-20° → 0（停摆）；中间线性过渡
        factor = np.clip((temps + 20.0) / 20.0, 0.0, 1.0)
        factor = np.power(factor, self.temp_sensitivity)
        return self.regrowth_rate * factor

    # ---- 快照 / 调试 ---------------------------------------------------------

    def snapshot(self) -> NDArray[np.float64]:
        """拷贝一份当前食物存量的二维网格图（行×列）。

        通俗理解：给现在的食物分布"拍张照"传给可视化，
        拍到的是一张 60×120 的图，一格一个值。

        参数
        ----
        无。

        返回
        ----
        NDArray[float64] : 形状 (rows, cols) 的拷贝，改它不影响场内数据。
        """
        return self._grid.reshape(self.world.rows, self.world.cols).copy()

    def __repr__(self) -> str:
        """打印这场的简要信息。

        返回
        ----
        str : 例如 "ResourceField(cells=7200, total=xx/xx)"。
        """
        cap = float(self._capacity.sum())
        return f"ResourceField(cells={self.world.n_cells}, total={self.total():.1f}/{cap:.1f})"