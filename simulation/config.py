"""SimConfig：整个模拟的配置文件（模块二 · 文件 1）。

模块职责
--------
集中管理所有数值参数（网格大小、光照、温度、资源、生物、基因、种群、
运行时长……），让"修改规则 = 改一个配置文件"，不用改代码。
与模块一的关系：SphereWorld / LightAndTemperature / ResourceField 的
构造参数都来自这里；它们自己不写死数值。

为什么用 dataclass？
--------------------
- 旧项目用 pydantic 校验，但本项目从零开始，为减少依赖（后面要接 Rust)
  改用 Python 内置 dataclass + 简单的 __post_init__ 断言；
- 配置即复现：same config + same seed → 完全相同的模拟。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class WorldConfig:
    """球面网格尺寸。"""

    rows: int = 60          # 纬度行数（row 0 = 上极 → row rows-1 = 下极）
    cols: int = 120         # 经度列数（经度环绕，col 0 == col cols-1 相邻）

    def __post_init__(self) -> None:
        assert self.rows >= 3, "至少要 3 行（上下极 + 至少一行赤道带）"
        assert self.cols >= 4, "经度至少 4 列"


@dataclass
class LightConfig:
    """光照温度参数（对应 LightAndTemperature 构造参数）。"""

    rotation_period: int = 2400   # 世界自转一圈的 tick 数（决定昼夜节律）
    t_equator: float = 30.0       # 赤道基温（抽象温度单位）
    t_pole: float = -20.0         # 极点基温（恒冷）
    day_boost: float = 6.0        # 昼夜温差幅度（正午比深夜高这么多）
    lat_base_ref: float = 1.0     # 纬度光照敏感度（越大极地越暗越冷）

    def __post_init__(self) -> None:
        assert self.rotation_period >= 10, "自转周期至少 10 tick"
        assert self.t_equator > self.t_pole, "赤道必须比极地暖"


@dataclass
class ResourceConfig:
    """资源（食物）参数（对应 ResourceField 构造参数）。"""

    capacity_per_area: float = 40.0   # 每单位面积的食物上限（容量）
    regrowth_rate: float = 0.5        # 基准再生：温度合适时每 tick 每格长多少
    temp_sensitivity: float = 1.0     # 再生对温度的依赖（0=不 care，越大越敏感）
    initial_fill: float = 0.5         # 初始填充比例（每格开始有多少食物，0~1）

    # ---- 斑块化（L1，守恒版）------------------------------------------
    # distribution="uniform" 时以下字段全部不生效，行为与旧版完全一致。
    # "patchy" 时：食物聚簇到斑块，背景压低；两条守恒保证总食物量不变：
    #   ① 容量守恒：Σ_capacity 与 uniform 版相等（种群承载上限不变）
    #   ② 再生守恒：周期平均总再生量与 uniform 版相等（时间上供给不变）
    distribution: str = "uniform"          # "uniform" | "patchy"
    patch_count: int = 30                  # 斑块中心数（默认保守，避免覆盖过大）
    patch_radius: int = 2                  # 斑块半径（格，邻居扩散层数）
    patch_capacity_mult: float = 3.0       # 斑块格容量倍率（>1；建议 1.5~4，过大会背景容量为负）
    patch_regrowth_mult: float = 2.0       # 斑块格再生倍率
    background_fill: float = 0.1            # 背景格初始食物占比（压低，否则协作无收益）

    def __post_init__(self) -> None:
        assert self.capacity_per_area > 0, "容量为正"
        assert self.regrowth_rate >= 0, "再生率非负"
        assert 0.0 <= self.initial_fill <= 1.0, "初始填充比例在 0~1"
        assert self.distribution in ("uniform", "patchy"), "distribution 只能是 uniform 或 patchy"
        if self.distribution == "patchy":
            assert self.patch_count >= 1, "斑块数至少 1"
            assert self.patch_radius >= 1, "斑块半径至少 1"
            assert self.patch_capacity_mult > 1.0, "斑块容量倍率必须 > 1（否则无富集）"
            assert self.patch_regrowth_mult > 0, "斑块再生倍率为正"
            assert 0.0 <= self.background_fill <= 1.0, "背景填充比例在 0~1"


@dataclass
class OrganismConfig:
    """个体能量收支与生命周期参数。"""

    initial_energy: float = 60.0      # 新生个体起始能量
    max_energy: float = 300.0         # 能量上限（多余溢出丢弃）
    base_metabolism: float = 0.6      # 每 tick 基础维持消耗（体温/活动）
    move_cost: float = 0.4            # 移动一格的基础能量消耗
    eat_amount: float = 0.5           # 每 tick 每格进食量上限
    eat_efficiency: float = 3.0       # 每单位食物转化为能量的倍率
    photo_max: float = 0.1            # 光合最大产能：光照=1（赤道正午）时每 tick 产这么多
    homeo_upkeep: float = 0.15        # 恒温个体每 tick 的额外维持费（换取低温不减速）
    maturity_fraction: float = 0.15   # 成熟年龄 = 寿命的几成 → 达到才能繁衍（防止一出生就生）
    senile_fraction: float = 0.75     # 老年年龄 = 寿命的几成 → 进入衰老期，维持费上升
    growth_mult: float = 1.6          # 未成年（幼体）每 tick 维持费倍率（在长身体，吃得多耗得多）
    senile_mult: float = 1.4          # 老年每 tick 维持费倍率（器官退化，维持费上升）
    # 活性温度门（隐式选择压，审计标注 [隐含] → A2 收编）：
    # 冷血个体有效活动 = 环境活动 × (niche_floor + niche_gain × 温度适配度)。
    # 0.4 保底=即使完全不适配温度仍有 40% 活动 → 弱化 g11 温度偏好的选择梯度。
    niche_floor: float = 0.4
    niche_gain: float = 0.6

    def __post_init__(self) -> None:
        assert self.initial_energy < self.max_energy, "初始能量要小于上限"
        assert self.eat_amount > 0, "进食量上限为正"
        assert self.photo_max >= 0, "光合产能非负"
        assert self.homeo_upkeep >= 0, "恒温维持费非负"
        assert 0.0 < self.maturity_fraction < self.senile_fraction < 1.0, (
            "成熟年龄要在老年年龄之前，且都要在寿限内"
        )
        assert self.growth_mult >= 1.0, "幼体代谢倍率至少 1"
        assert self.senile_mult >= 1.0, "老年代谢倍率至少 1"


@dataclass
class GenomeConfig:
    """基因底物参数：定长连续基因链 + 变异。"""

    gene_count: int = 24            # 基因数量：g0~g13 核心行为；g14 感知/g15 信号/g16 攻击/g19 扎根 已接线；
                                    # g17 食性/g18 防御/g20 享乐/g21 处理位/g22 信任阈值/g23 为声明未接线（预留位，变异无行为效果）
    gene_min: float = 0.0           # 基因取值下限
    gene_max: float = 1.0           # 基因取值上限
    mutation_rate: float = 0.05     # 每个基因发生变异的概率
    mutation_sigma: float = 0.05    # 变异震荡幅度（相对基因区间宽度）

    def __post_init__(self) -> None:
        assert self.gene_max > self.gene_min, "上限要大于下限"
        assert 0.0 <= self.mutation_rate <= 1.0, "变异率在 0~1"


@dataclass
class PopulationConfig:
    """种群规模参数。"""

    initial_count: int = 200        # 初始个体数量
    max_count: int = 5000           # 种群硬上限（防失控）

    def __post_init__(self) -> None:
        assert self.initial_count >= 1, "至少一个个体"
        assert self.max_count >= self.initial_count, "上限不小于初始"


@dataclass
class SimulationConfig:
    """引擎运行参数。"""

    ticks: int = 1000               # 计划运行的最大 tick 数
    stop_on_extinction: bool = True    # 种群归零时提前停
    history_limit: int = 0          # 统计历史保留上限（0=无限，长程实验用环形尾部）
    use_sim_core: bool = False      # True=种群数值管线走 Rust（sim_core.step_vectors）

    def __post_init__(self) -> None:
        assert self.ticks >= 1, "至少跑一个 tick"


@dataclass
class PleasureConfig:
    """愉悦度系统参数（L2）：预测误差驱动的内在动机系统。

    核心公式：愉悦度 = 实际获得 − 预期获得（RPE，对应多巴胺系统）。
    不是"做好事给分"，而是"比预期好就愉悦"。
    """

    enabled: bool = True               # 总开关（False 时愉悦度数组仍存在但不更新）
    expectation_size: int = 120        # 情境数（能量5×食物4×邻居3×信号2=120）
    alpha: float = 0.05                # EWMA 预期学习率（越小越慢、越稳定）
    valence_decay: float = 0.95        # valence 每 tick 衰减（回到中性 0）
    arousal_decay: float = 0.97        # arousal 衰减（意外事件→高唤醒）
    baseline_rate: float = 0.001       # baseline 慢漂移率（习惯化）
    optimism: float = 0.8               # 初始乐观系数（expectation = optimism × max_reward）
    max_reward: float = 2.0             # 单 tick 最大可能收益（用于乐观初始化归一化）
    w_energy: float = 0.5               # 事件收益：Δ能量权重
    w_info: float = 0.3                 # 事件收益：信息增益权重
    w_social: float = 0.2               # 事件收益：社会增益权重
    inheritance_noise: float = 0.02     # 繁殖时 expectation 继承噪声（文化传递载体）
    # 社会事件基础效价（审计标注 [隐含] → A2 收编）：
    # 有同伴 ≥1 → +social_rpe / 孤独 → alone_rpe（注意非对称：0.2 vs -0.1）。
    social_rpe: float = 0.2
    alone_rpe: float = -0.1

    def __post_init__(self) -> None:
        assert 0 < self.alpha <= 1, "alpha 应在 (0,1]"
        assert 0 < self.valence_decay <= 1, "valence_decay 应在 (0,1]"
        assert self.expectation_size == 120, "当前情境编码固定为 5×4×3×2=120"


@dataclass
class PredationConfig:
    """捕食参数（L4，影响 g16 攻击性的 fitness）。

    审计标注：能量转移率 0.4 是[隐含]最强"战斗红利"、成功率乘 g16 是[刻意]强选择、
    攻击概率系数/门槛为[隐含]（→ A2 收编，默认值保持旧行为逐位一致）。
    """

    attack_cost: float = 0.1           # 每次攻击的能耗（无论成败）
    attack_prob_coef: float = 0.2      # 攻击概率 ≈ g16 × 系数 × 饥饿度
    attack_gene_gate: float = 0.3      # g16 低于该值不发动攻击
    success_gene_gain: float = 0.5     # 成功率 = 能量比 × (0.5 + g16×gain)，clamp
    success_floor: float = 0.1         # 成功率下限
    success_ceil: float = 0.9          # 成功率上限
    transfer_ratio: float = 0.4        # 捕食成功：猎物能量转移比例
    stomach_transfer: float = 0.4      # 猎物胃粮转移比例

    def __post_init__(self) -> None:
        assert self.attack_cost > 0
        assert 0 <= self.success_floor < self.success_ceil <= 1.0
        assert 0.0 <= self.transfer_ratio <= 1.0
        assert 0.0 <= self.stomach_transfer <= 1.0
        assert 0.0 <= self.attack_gene_gate <= 1.0


@dataclass
class CultureConfig:
    """文化学习参数（L5，信任系统）。

    审计标注：信任更新非对称（+trust_true / −trust_false）是[隐含]反合作偏置，
    → A2 收编。默认值保持旧行为（真 +0.05 / 假 −0.1）。
    """

    food_threshold: float = 0.3        # "邻格有粮"判定阈值（信号验证用）
    trust_true: float = 0.05           # 信号验证为真 → 信任上升幅度
    trust_false: float = 0.1           # 信号验证为假 → 信任下降幅度（注意取负前传）

    def __post_init__(self) -> None:
        assert 0.0 <= self.food_threshold <= 1.0
        assert self.trust_true >= 0
        assert self.trust_false >= 0


@dataclass
class FruitConfig:
    """果实-种子传播参数（L10a，植物-动物协同进化）。

    核心链路：植物蓄力→结果→动物吃果实→摄入种子→排泄→萌发新植物。
    默认 enabled=False（不改变现有行为），开启后新增果实场/蓄力/种子携带状态。
    """

    enabled: bool = False              # 总开关（False 时完全不执行 L10 步骤）
    plant_threshold: float = 0.5       # g19 >= 此值判定为植物（会结果）
    charge_rate: float = 0.1           # 每 tick 蓄力速率（× g8 光合产能）
    fruit_threshold: float = 10.0      # 蓄力达此值→结果释放
    fruit_ratio: float = 0.8           # 释放到果实场的能量比例（其余损耗）
    eat_rate: float = 0.05             # 动物每 tick 吃果实比例（× 果实能量）
    digest_ratio: float = 0.7          # 吃果实的能量消化率
    seed_intake_prob: float = 0.3      # 吃果实时摄入种子的概率
    excretion_prob: float = 0.1         # 携带种子每 tick 排泄概率
    germination_prob: float = 0.5       # 排泄后种子萌发概率
    seed_energy: float = 5.0            # 萌发新植物的初始能量
    max_seed_carried: int = 5           # 单个体最大携带种子数

    def __post_init__(self) -> None:
        assert 0.0 <= self.plant_threshold <= 1.0
        assert self.charge_rate >= 0
        assert self.fruit_threshold > 0
        assert 0.0 <= self.fruit_ratio <= 1.0
        assert 0.0 <= self.eat_rate <= 1.0
        assert 0.0 <= self.digest_ratio <= 1.0
        assert 0.0 <= self.seed_intake_prob <= 1.0
        assert 0.0 <= self.excretion_prob <= 1.0
        assert 0.0 <= self.germination_prob <= 1.0
        assert self.seed_energy > 0
        assert self.max_seed_carried >= 0


@dataclass
class CarcassConfig:
    """尸体能量守恒参数（L9）：死亡生物量不凭空消失，形成尸体→食腐→分解循环。

    能量守恒方程：总生物量(生物+胃粮+食物+尸体+果实)变化 = 光合输入 - 代谢/维持/移动/攻击/繁殖/分解损耗。
    默认 enabled=False（不改变现有行为），开启后死亡个体能量按比例转化为尸体。
    """

    enabled: bool = False              # 总开关（False 时尸体数组存在但不转化/食用）
    carcass_ratio: float = 0.8         # 死亡时能量转化为尸体的比例（其余 20% 为不可食用部分/代谢损耗）
    scavenge_digestibility: float = 0.7  # 食腐消化率（尸体不如新鲜食物）
    decomposition_rate: float = 0.002  # 尸体每 tick 自然分解比例（约 500 tick 完全分解）
    resource_conversion: float = 0.5   # 分解的尸体转化为食物资源的比例（其余为分解损耗）
    scavenge_joy: float = 0.003        # 食腐的愉悦度权重（低于进食新鲜食物）

    def __post_init__(self) -> None:
        assert 0.0 <= self.carcass_ratio <= 1.0
        assert 0.0 <= self.scavenge_digestibility <= 1.0
        assert 0.0 <= self.decomposition_rate <= 1.0
        assert 0.0 <= self.resource_conversion <= 1.0


@dataclass
class TerrainConfig:
    """地形系统参数（L8 简化版）：静态空间异质性，水域不可通行，山地移动能耗高。

    简化实现：不用值噪声，用纬度带（极地附近水域概率高）+ 随机斑块生成。
    地形类型：0=平原（默认）、1=山地（移动能耗×2、资源再生×0.3）、2=水域（不可通行）。
    默认 enabled=False（不改变现有行为）。
    """

    enabled: bool = False              # 总开关
    water_ratio: float = 0.08          # 水域占比（不可通行格）
    mountain_ratio: float = 0.12       # 山地占比
    mountain_move_mult: float = 2.0    # 山地移动能耗倍率
    mountain_regrow_mult: float = 0.3  # 山地资源再生倍率
    water_pole_bias: float = 0.5       # 极地水域概率加成（0=均匀，1=极地水域概率×2）

    def __post_init__(self) -> None:
        assert 0.0 <= self.water_ratio <= 0.5
        assert 0.0 <= self.mountain_ratio <= 0.5
        assert self.water_ratio + self.mountain_ratio <= 0.8
        assert self.mountain_move_mult >= 1.0
        assert 0.0 <= self.mountain_regrow_mult <= 1.0


@dataclass
class SeasonConfig:
    """季节系统参数：黄赤交角导致太阳直射点随季节移动，冬季半球资源贫瘠。

    季节周期 = year_length × rotation_period（tick）。太阳直射纬度 = axial_tilt × sin(2π×tick/year_ticks)。
    冬季半球：光照降低→温度降低→资源再生和植物光合减缓，迫使生物迁移至夏季半球。
    默认 enabled=False（不改变现有行为，保持原固定太阳模型）。
    """

    enabled: bool = False              # 总开关
    year_length: int = 3               # 一年 = 多少个世界日（rotation_period）。
                                        # 数值平衡：生物寿命=1~8世界日（g3决定，平均≈4.5），
                                        # year_length=3 时平均寿命≈1.5年、最长≈2.7年，
                                        # 确保大部分生物能体验至少1个完整四季，长寿生物2~3个。
                                        # 调大→季节变化慢、生物体验少；调小→季节变化快、选择压强。
    axial_tilt: float = 0.4            # 黄赤交角（弧度，约23°），太阳直射点摆动幅度
    winter_regrow_mult: float = 0.3    # 冬季半球资源再生倍率（冬季资源贫瘠）
    winter_photo_mult: float = 0.3     # 冬季半球植物光合倍率

    def __post_init__(self) -> None:
        assert self.year_length >= 1
        assert 0.0 <= self.axial_tilt <= 1.0
        assert 0.0 <= self.winter_regrow_mult <= 1.0
        assert 0.0 <= self.winter_photo_mult <= 1.0


@dataclass
class SimConfig:
    """顶层配置：唯一事实来源，决定一次完整模拟。"""

    seed: int = 42                  # 随机数种子（同配置+同种子 → 同结果）
    world: WorldConfig = field(default_factory=WorldConfig)
    light: LightConfig = field(default_factory=LightConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    organisms: OrganismConfig = field(default_factory=OrganismConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pleasure: PleasureConfig = field(default_factory=PleasureConfig)
    predation: PredationConfig = field(default_factory=PredationConfig)
    culture: CultureConfig = field(default_factory=CultureConfig)
    fruit: FruitConfig = field(default_factory=FruitConfig)
    carcass: CarcassConfig = field(default_factory=CarcassConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    season: SeasonConfig = field(default_factory=SeasonConfig)

    # ---- 可复现性辅助：配置 ⇄ dict ------------------------------

    def to_dict(self) -> dict:
        """把整个配置导出成普通 dict（存档/对比用）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        """从 dict 重建配置（保证存档可得回同样的配置）。

        旧存档缺失 predation/culture 键时回退默认值（保持向前兼容）。
        """
        return cls(
            seed=data["seed"],
            world=WorldConfig(**data["world"]),
            light=LightConfig(**data["light"]),
            resources=ResourceConfig(**data["resources"]),
            organisms=OrganismConfig(**data["organisms"]),
            genome=GenomeConfig(**data["genome"]),
            population=PopulationConfig(**data["population"]),
            simulation=SimulationConfig(**data["simulation"]),
            # L2 新增的愉悦度配置此前遗漏；旧存档缺少该键时回退默认值。
            pleasure=(
                PleasureConfig(**data["pleasure"])
                if "pleasure" in data
                else PleasureConfig()
            ),
            # A2 新增捕食/文化配置（隐式选择压收编）；旧存档回退默认值。
            predation=(
                PredationConfig(**data["predation"])
                if "predation" in data
                else PredationConfig()
            ),
            culture=(
                CultureConfig(**data["culture"])
                if "culture" in data
                else CultureConfig()
            ),
            # L10a 新增果实-种子传播配置；旧存档回退默认值（enabled=False）。
            fruit=(
                FruitConfig(**data["fruit"])
                if "fruit" in data
                else FruitConfig()
            ),
            # L9 尸体能量守恒；旧存档回退默认值（enabled=False）。
            carcass=(
                CarcassConfig(**data["carcass"])
                if "carcass" in data
                else CarcassConfig()
            ),
            # L8 地形系统；旧存档回退默认值（enabled=False）。
            terrain=(
                TerrainConfig(**data["terrain"])
                if "terrain" in data
                else TerrainConfig()
            ),
            # 季节系统；旧存档回退默认值（enabled=False）。
            season=(
                SeasonConfig(**data["season"])
                if "season" in data
                else SeasonConfig()
            ),
        )

    def fingerprint(self) -> str:
        """配置的规范字符串；两份配置是否完全一致（排查复现用）。"""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)