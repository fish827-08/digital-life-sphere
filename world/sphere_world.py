"""SphereWorld：经纬网格构成的球面世界（核心拓扑层）。

模块职责
--------
把旧版"平面矩形网格（torus）"升级为"经纬网格球面"，本文件只负责
【拓扑】：网格索引转换、面积权重、极点坍缩、邻居关系。
光照/温度、资源场在后续文件实现。

关键设计（极点坍缩）
------------------
- 纬度 60 行（row 0 = 上极 → row 59 = 下极），经度 120 列（环绕）；
- 极点行在物理上是【一个格子】：row 0 与 row 59 的所有经度槽位都坍缩
  为各自行的 col 0（flat 不变，仍是 row*cols+0），
  即 rc_to_flat(0, 任意col) 一律返回 0；
- 极区格子面积 ∝ cos(纬度) → 极点格子面积最小（≈0.026，约为赤道 1/38），
  生态承载能力在极地显著更低；
- 普通格邻居恒 8 个；极点格邻居为相邻纬度带整行（120 个）。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SphereWorld:
    """经纬球面网格（等距圆柱投影，默认 60×120）。

    实例属性（__slots__ 声明的全部字段）说明
    ---------------------------------------
    rows : int
        纬度行数（网格高度），构造时确定，不可变。
    cols : int
        经度列数（网格宽度），构造时确定，不可变。
    _n : int
        总格子数 = rows × cols（含极点坍缩后的冗余槽位，物理格更少）。
    _lat : NDArray[float64], 形状 (rows,)
        每行中心纬度，单位弧度，范围 (-π/2, π/2)。
        _lat[r] = -π/2 + (r + 0.5) * (π / rows)。
    _area : NDArray[float64], 形状 (rows,)
        每行的面积权重，等于 cos(_lat[r]) / max(cos(_lat))。
        近赤道行 ≈ 1.0，极点行 ≈ 0.026。查询 cell_area() 使用。
    _pole_top : int
        上极点行索引，恒为 0。
    _pole_bottom : int
        下极点行索引，恒为 rows - 1。
    """

    __slots__ = ("rows", "cols", "_lat", "_area", "_n", "_pole_top", "_pole_bottom")

    # ---- 构造函数 ----------------------------------------------------------

    def __init__(self, rows: int = 60, cols: int = 120) -> None:
        """构造球面网格。

        参数
        ----
        rows : int, 默认 60
            纬度行数。每行纬度跨度 = 180°/rows，决定纬度分辨率。
        cols : int, 默认 120
            经度列数。每列经度跨度 = 360°/cols，决定经度分辨率。

        返回
        ----
        None。构造完成后立即调用查询方法。
        """
        self.rows = rows
        self.cols = cols
        self._n = rows * cols

        # 每行中心纬度（弧度）：-90°+Δ/2 … +90°-Δ/2
        dlat = np.pi / rows
        self._lat = -np.pi / 2 + dlat / 2 + np.arange(rows) * dlat

        # 面积权重 ∝ cos(纬度)，归一化使最大行（近赤道）= 1
        cos_lat = np.cos(self._lat)
        self._area = cos_lat / cos_lat.max()

        # 极点行索引（上极 / 下极）
        self._pole_top = 0
        self._pole_bottom = rows - 1

    # ---- 只读属性 ----------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """返回网格总格数。

        说明：这是内存布局意义上的总槽位数（rows × cols）。
        因极点坍缩，物理上独立位置比槽位数少：上/下极点各坍缩为 1 格，
        共减少 2×(cols−1) 个物理重复槽位。

        返回
        ----
        int : rows × cols，如 60×120 = 7200。
        """
        return self._n

    # ---- 索引转换 ----------------------------------------------------------

    def flat_to_rc(self, flat: NDArray[np.int64]) -> tuple[np.ndarray, np.ndarray]:
        """把平铺索引拆成 (行, 列)。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（可传单个标量，也可传一组，便于向量化批量转换）。
            有效范围 0..rows*cols-1。

        返回
        ----
        (row, col) : 元组 (NDArray[int64], NDArray[int64])
            与入参形状相同的两个数组；row = flat // cols，col = flat % cols。
            例：flat=125, cols=120 → row=1, col=5。 //读到解释代码，就是把经纬度的点转化为具体的格子（120，60）的点吗？，方法解释是做什么可以说的通俗一点
        """
        flat = np.asarray(flat, dtype=np.int64)
        return np.divmod(flat, self.cols)

    def rc_to_flat(self, row, col) -> NDArray[np.int64]:
        """把 (行, 列) 合并成平铺索引（含环绕与极点坍缩处理）。

        参数
        ----
        row : int 或 NDArray[int64]
            纬度行序号（0..rows-1）。
        col : int 或 NDArray[int64]
            经度列序号（任意整数也可，内部自动取模到 [0, cols)）。

        返回
        ----
        NDArray[int64] : 平铺索引 = row * cols + col。
            两个特殊处理：
            - 经度环绕：col=-1 → cols-1，col=cols → 0；
            - 极点坍缩：row 为极点行时，无论 col 传什么，一律返回该行 col0
              对应的 flat（物理上极点只有一格）。
        """
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        # 经度环绕 [0, cols)
        col = col % self.cols
        # 极点行：所有经度物理坍缩为同一格（col 0）
        col = np.where(
            (row == self._pole_top) | (row == self._pole_bottom), 0, col
        )
        return (row * self.cols + col).astype(np.int64)

    # ---- 几何属性 ----------------------------------------------------------

    def latitude_of(self, row):
        """查询某(些)行的中心纬度（弧度）。

        参数
        ----
        row : int 或 NDArray[int64]
            纬度行序号（0..rows-1），可传标量或数组。

        返回
        ----
        float 或 NDArray[float64] :
            标量入参 → 返回 float；数组入参 → 返回同形状数组。
            数值范围 (-π/2, π/2)；极点行约 -1.535 rad（≈ -88°）。
        """
        out = np.asarray(self._lat[np.asarray(row, dtype=np.int64)])
        return float(out) if out.ndim == 0 else out

    def cell_area(self, flat):
        """查询格子面积权重（相对赤道的比例）。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引，可传标量或数组。

        返回
        ----
        float 或 NDArray[float64] :
            标量入参 → 返回 float；数组入参 → 返回同形状数组。
            权重 = cos(纬度)/cos(最大纬度中心)：
            近赤道行 ≈ 1.0；60 行网格极点行 ≈ 0.026。

        用途：资源容量、温度扩散、可视化着色等都按它缩放。
        """
        flat = np.asarray(flat, dtype=np.int64)
        rows, _ = self.flat_to_rc(flat)
        out = np.asarray(self._area[rows])
        return float(out) if out.ndim == 0 else out

    def is_pole(self, flat) -> np.ndarray:
        """判定格子是否位于极点带（row 0 或 rows-1）。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引，可传标量或数组。

        返回
        ----
        bool 或 NDArray[bool] :
            True 表示该格在极点带（物理上坍缩为一格）。
            极点带内任一 col 都算 True（如 flat=0 和 flat=59 都是极点）。
        """
        flat = np.asarray(flat, dtype=np.int64)
        rows, _ = self.flat_to_rc(flat)
        return (rows == self._pole_top) | (rows == self._pole_bottom)

    # ---- 邻居查询 ----------------------------------------------------------

    def neighbors(self, flat: int) -> NDArray[np.int64]:
        """查询单个格子的邻居（含对角，纬度方向取舍见返回说明）。

        参数
        ----
        flat : int
            单格平铺索引（本方法按单格查询设计，传数组无意义）。

        返回
        ----
        NDArray[int64] : 邻居格平铺索引数组。
            普通格：恒 8 个（上/下/左/右 + 4 个对角；经度方向环绕取模，
                    纬度方向在极点处钳制回界内，即最外行列不越界）。
            极点格：恒 cols 个 —— 相邻纬度带整行（向赤道方向一行）的所有格。
                    含义：从极点可一步到达赤道方向任意经度。
        """
        row, col = self.flat_to_rc(np.asarray(flat, dtype=np.int64))
        row, col = int(row), int(col)
        if row in (self._pole_top, self._pole_bottom):
            # 极点：邻居 = 相邻纬度带整行（向赤道方向移一行）
            adj = self._pole_top + 1 if row == self._pole_top else self._pole_bottom - 1
            return (np.arange(self.cols) + adj * self.cols).astype(np.int64)

        drow = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
        dcol = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
        r = np.clip(row + drow, 0, self.rows - 1)  # 纬度不越界（两极已在上面处理）
        c = (col + dcol) % self.cols  # 经度环绕
        return self.rc_to_flat(r, c)

    # ---- 调试 / 展示 -------------------------------------------------------

    def __repr__(self) -> str:
        """生成对象的人类可读字符串（便于打印调试）。

        参数
        ----
        无。

        返回
        ----
        str : 例如 "SphereWorld(rows=60, cols=120, cells=7200)"。
        """
        return f"SphereWorld(rows={self.rows}, cols={self.cols}, cells={self._n})"