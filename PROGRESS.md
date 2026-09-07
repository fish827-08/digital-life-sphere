# 项目进度文档

## 项目概况
- 项目名称：digital-life-sphere
- 启动日期：2026-09-06
- 当前阶段：模块一（球面世界）完成，准备进入模块二（引擎层）

## 总体进度
- [x] 拷贝 AI-CODE-DEVELOPMENT-RULES.md 到项目根目录
- [x] 规则文件加入 .gitignore
- [x] 项目方向评估（规则二）：输出评估报告
- [x] 确认项目总体方案
- [x] 设计整体架构，AI 提草案并与我逐项确认
- [x] 创建 PROJECT-DESCRIPTION.md（功能描述文档，含目录，v0.5 定稿）
- [x] 再创建 MODULES.md（模块实现说明：每个文件/类/方法的通俗解释）
- [x] 模块一开发（SphereWorld 世界模块）——3 个文件全部完成并验证
- [x] 模块二开发（球形引擎 Python 版）——config/tick/lifecycle/sphere_engine 完成；
      基因扩充 g0~g13 + 生命周期年龄（成熟/年龄能耗）；引擎 pytest 套件 7 例全通过；
      observatory 快速长程验证通过（experiments/quick_check_age.py）
- [ ] 模块三开发（Rust 加速核 Sim-core，✅ 全部完成：3.1~3.5 下沉 + 预计算邻居表优化，
      全量 pytest 34 例通过，代码已提交 3ed6fd0）
- [ ] 模块四开发（observatory 适配 + 快照桥，✅ 完成：traits/statistics/observer/experiment/
      __main__/persistence/io + broker 快照桥，统计口径=14 基因位+3 派生，快照推送=100 tick，
      全量 pytest 50 例通过，新增 16 例）
- [ ] 模块五开发（前端渲染层）

## 已完成记录
| 日期 | 完成内容 | 备注 |
|------|----------|------|
| 2026-09-06 | 初始化项目目录 | 名称 digital-life-sphere，位于 the-world 内部子目录 |
| 2026-09-06 | 复制 AI-CODE-DEVELOPMENT-RULES.md 并加入 .gitignore | 遵循记忆中的文档要求 |
| 2026-09-06 | 模块一：world/sphere_world.py | 土地拓扑：经纬网格 60×120、极点坍缩、面积梯度、8 邻 |
| 2026-09-06 | 模块一：world/light_and_temperature.py | 天气：固定太阳+自转（昼夜扫掠）、纬度基温+小温差、活性因子 |
| 2026-09-06 | 模块一：world/resource_field.py | 食物：容量随面积、再生随温度（同格多只均分不欠账） |
| 2026-09-06 | 创建 MODULES.md | 收录模块一全部类与方法的通俗解释、测试点 |
| 2026-09-06 | 模块二：simulation/config.py | 总参数本：8 个子配置 dataclass，配置即复现（to_dict/from_dict/fingerprint） |
| 2026-09-06 | 模块二：simulation/tick.py + core/lifecycle.py | TickStats 快照 + DeathCause 死因枚举（饿死/老死） |
| 2026-09-06 | 模块二：simulation/sphere_engine.py | 引擎核心：单 tick 流程 0~9、数组化推进、繁殖/死亡/清理 |
| 2026-09-06 | 基因扩充 g4~g13 | 进食量/饱食度/移动能耗/繁殖投入/光合/恒温/邻格觅食/温度偏好/繁殖冷却/群居性；机制单测 + 1000 tick 冒烟通过 |
| 2026-09-07 | 模块三 3.5：sim_core/src/consume.rs | 进食结算下沉 Rust：先数每格几只再算每只实吃量（同格均分、绝不欠账），按原序逐只扣减，与 numpy `consume_many` 逐位等价；lib.rs 绑定加长度+越界校验；引擎进食/邻格觅食双路径接入；函数级对拍 8 例 + 引擎级 4 例，全量 pytest 30 例通过 |
| 2026-09-07 | 预计算邻居表优化（world/sphere_world.py） | 构造时一次性算好全网格邻居表 `_nb_table`(7200,8) + `_pole_nb`(2,120)，`neighbors()` 改查表返回；引擎移动/觅食两处调用点零改动，行为零变化（新 tests/test_sphere_world.py 全网格 vs 旧算法逐位对拍）；基准 3.030/3.049s → 0.615/0.569s ≈ 5x，全量 pytest 34 例通过 |
| 2026-09-07 | 模块四：observatory 适配 + 快照桥（✅） | traits（14 基因位+3 派生，统计口径与引擎同公式）/ statistics（数组化聚合，直接读引擎 SoA 数组）/ observer（世代+节拍双触发器）/ experiment（嵌套 SimConfig + overrides 合并 + SphereEngine runner，run_single 每 tick 对齐终止条件置位）/ __main__ CLI（--rows/--cols/--sim-core）/ persistence/io（manifest+generations 落盘）；快照桥 broker：每 100 tick 采 JSON 快照（含个体明细广播），WebSocket 推流，环形只有标量防 OOM，新客户端连上补发最新快照；pytest 全量 50 例通过（新增 16 例）|
| 2026-09-07 | 提交并推送模块三、四到 GitHub（origin/main） | 提交：3ed6fd0（模块三）/ 8aefe06（模块四）；`git push origin main` 后本地与远端同步 | GitHub 远端 `fish827-08/digital-life-sphere`（SSH）为唯一备份，推送后 3 个提交（模块二收尾~模块四）全部上云 |
| 2026-09-07 | **L1 斑块资源（守恒版）** — feature/l1-patchy 分支 | ResourceConfig 加 7 个斑块参数 + 断言；ResourceField 守恒分布（容量+再生守恒）；引擎接入（patch_seed 独立 rng，patchy 时 Rust regrow 回退 Python）；修复 `set_initial_fill` 不存在的潜在 bug；tests/test_patchy_resource.py 18 例全过；全量 Python 路径 39 例通过；experiments/patchy_vs_uniform.py 对比实验：总食物守恒（差异<1%）、种群 0/3 灭绝、空间聚集未显现（无感知基因，预期内） | 语言涌现五要素第一步：资源空间异质化。守恒设计保证不破坏平衡。生态效应需等 L3 感知基因上线才能观测 |
| 2026-09-07 | **L2+L3 愉悦度+信号+感知** — feature/l1-patchy 分支 | L2 信号场 world/signal_field.py（田字格 4 子格=16 模式 uint8，duration=50tick，11 例测试）；L2 愉悦度（四数组 _valence/_arousal/_expectation(N,120)/_baseline，RPE 预测误差驱动，120 情境=能量5×食物4×邻居3×信号2，EWMA α=0.05，乐观初始化，繁殖继承，9 例测试）；L3 g14 感知（移动决策综合得分=感知×(食物×0.7+信号×0.3)+群居×密度）；L3 g15 信号（发射概率=g15，耗能 0.5，模式=状态哈希，信息增益接入愉悦度）；PleasureConfig 加入 SimConfig；tests/test_pleasure.py 9 例 + tests/test_perception_signal.py 10 例全过；全量 90 例通过；experiments/long_run_l2l3.py 长跑脚本；5000tick 实验：117tick/s@N=500，种群爆发到 5000，信号密度 6073 格，g14 上升(0.52→0.55)被保留，g15 下降(0.47→0.30)发射耗能>接收收益，愉悦度 RPE 动力学符合预期（乐观→失望-0.58→学习→回升-0.09） | 语言涌现硬件三件套：感知+信号+愉悦度。关键发现：g15 信号基因因发射耗能>接收收益被淘汰，说明信号系统必须配"接收收益"才能维持——后续需增大信号在移动决策中的权重或让信号直接指示食物。Rust 环境已搭建（rustup 1.98.1+maturin，sim_core 编译成功） |

## 进行中
- 模块四已完成（8aefe06）；L1 斑块资源（守恒版）已完成；L2 愉悦度+信号场、L3 感知(g14)+信号(g15)已完成（全量 90 例通过，5000tick 实验完成，100 万 tick 后台长跑中）；下一步：L4 行为决策（捕食 g16/植物化 g19）或 L5 学习与继承（工作记忆/文化传递/信誉表），或修复 g15 信号基因接收收益不足问题

## 待办事项
1. 模块一 SphereWorld 开发（✅ 已完成）
2. 模块二 引擎核心（✅ config/tick/lifecycle/engine 完成，基因扩充 g0~g13 完成）
3. 模块二 收尾：pytest 用例固化（✅ tests 7 例通过）+ observatory 快速长程验证（✅）
4. 模块三 Rust 热核 Sim-core（✅ 3.1~3.5 + 邻居表优化，已提交 3ed6fd0）
5. 模块四 observatory 适配 + 快照桥（✅ 已提交 8aefe06 并推送远端）
6. 模块五 前端渲染层（未开始）
7. **语言涌现扩展 L1 斑块资源（守恒版）**（✅ 已完成，feature/l1-patchy 分支）
8. 语言涌现扩展 L2 数组级扩展（愉悦度四数组/田字格信号场，未开始）
9. 语言涌现扩展 L3 基因解码扩展（g14 感知/g15 信号/g16 攻击等，未开始）
10. 语言涌现扩展 L4 行为决策逻辑（信号发射/捕食/植物化，未开始）
11. 语言涌现扩展 L5 学习与继承机制（工作记忆/文化传递/信誉表，未开始）
12. 语言涌现扩展 L6 Rust 下沉（热点循环 PyO3，未开始）

## 问题与决策记录
| 日期 | 问题 | 决策 | 原因 |
|------|------|------|------|
| 2026-09-06 | 项目语言与性能架构 | Python 主语言 + Rust 热点下沉（PyO3），方案 4，热核边界选 **B（数值+资源场）** | 保留旧版资产，性能瓶颈下沉 |
| 2026-09-06 | 世界拓扑 | 经纬网格（等距圆柱投影），极点经度坍缩为一格，8 邻 | 极区格子小于赤道，符合球面特性 |
| 2026-09-06 | 网格分辨率 | 纬度 60 × 经度 120（7,200 格） | 2° 格 |
| 2026-09-06 | 光照模型 | 固定太阳 + 世界自转 → 昼夜扫掠；温度由光照映射（极地冷、夜晚略冷） | 简化模型，生态涌现靠环境梯度 |
| 2026-09-06 | 资源初始分布 | 均匀分布 | 差异靠再生与个体行为涌现 |
| 2026-09-06 | 快照推送频率 | 每 100 tick | 后续按实际效果调整 |
| 2026-09-06 | 项目组织 | 新项目 digital-life-sphere，代码复用旧项目 | 一步步重构，全程可控可理解 |
| 2026-09-06 | 开发顺序 | 先世界模块 → 再放生物 | 世界先独立可运行/可视化 |
| 2026-09-06 | tick 语义 | **tick 是抽象时间单位**：1 tick 打个比方 = 现实世界 1 秒；"这一秒世界发生的变化"= 引擎跑一次 `step()` 的整套流程（资源再生→代谢→维持→进食→移动→衰老→死亡→繁殖）；不写成与现实的硬映射 | 让世界观对齐：时间不再"神秘"，就是一句"每秒世界推进一帧" |
| 2026-09-06 | 基因扩充（第 2 批） | 新增 g11 温度偏好 / g12 繁殖冷却 / g13 群居性，全部挂靠既有机制（温度响应/繁殖/移动） | 继续丰富多样性，但**不新增子系统**，热路径改动最小 |
| 2026-09-06 | 基因位评估（参照 The Bibites 28 基因） | **加入**：光合(冷血收益)、恒温（费能换低温不减速）、邻格觅食、温度偏好、繁殖冷却、群居性；**暂缓**：食性/捕食（要"尸体+伤害"机制）、脂肪储能（要存储池）、感知/交流（独立感知子系统，是"生物间能否交流"的评估结论——等引擎稳定后另立模块再评）、器官分配（器官系统） | 优先接入与现存机制正交的基因，避免为单个基因造一整块新系统 |
| 2026-09-06 | "一出生就能繁衍"问题 | 引入**生命周期年龄**：未到成熟年龄（=寿命×15%，寿命由 g3 定）一律不能繁衍；能量需求随年龄变：幼体×1.6（长身体）/成年×1/老年×1.4（器官退化） | 防止新生命前几代疯狂爆发；寿命长则成熟晚→自然涌现"速生速死"vs"晚熟长寿"两种策略 |
| 2026-09-06 | 生命周期年龄长程验证（模块二收尾） | 新增 experiments/quick_check_age.py 做 4000 tick 快速长程检查：结果未灭绝、世代达 12、寿命 g3 std=622 保留、速生(g3低 均2285)vs晚熟(g3高 均3762)两类策略并存 → 机制生效 | 验证成熟门槛确实抑制"出生即生"、能量需求随年龄分化，同时长寿基因多样性不被冲掉 |
| 2026-09-06 | **寿命数值平衡（寿命 vs 昼夜）** | 寿命公式 `200 + g3×3800`（[200,4000]，小于一昼夜 2400）改为 **一昼夜 × (1 + g3×7)**（[2400,19200]，最短整整一昼夜、最长八昼夜），并抽象为引擎 `_lifespan()` 统一出口（挂 `light.rotation_period`，改昼夜设置自动缩放） | 原寿命全部小于/接近一昼夜，生物来不及对昼夜做反应；新基准保证所有个体至少活满一昼夜，仍保留"速生(候1昼夜)vs晚熟(候8昼夜)"分化 |
| 2026-09-06 | 3.3 管线切分（为什么两段式） | `_step_population` 的数值运算拆成 **stage1（第 1~3 步：光合/代谢/维持）** 与 **stage2（第 5~8 步：移动扣费/年龄/死亡/冷却/繁殖候选）**，中间的"进食（步骤 4）+ 移动抽样/觅食目标选择（RNG）"留 Python | 步骤 4~5 依赖 RNG 且在 Python 侧要按原顺序消费随机数（同种子同结果）；Rust 只碰确定性数值，才能逐位对拍 |
| 2026-09-06 | 3.3 对拍保证策略 | ① 数值纪律：Rust 只用 f64 四则 + min/max，禁 exp/pow → 与 numpy 逐位一致；② 函数级：同一份随机种群分别走 py_stage1/2 参考实现与 Rust stage1/2，assert_array_equal 全部输出；③ 引擎级：同 seed 两个引擎（use_sim_core 开关），逐 tick TickStats + 最终内部数组逐位比较 | 三层递进：先证函数位级一致，再证接入引擎后 RNG 消费顺序不变、整体行为不变 |
| 2026-09-07 | 3.4 regrow 接入方式 | 不在 ResourceField 内部加开关，而是引擎 `_advance_one_tick` 里双路径：use_sim_core 时组好温度数组调 sim_core.regrow，否则调 ResourceField.regrow | 与 3.3 步进下沉同一模式，资源场类保持零改动；sens=1.0 严格逐位，≠1 允许 ≤1-ULP（函数级 1e-9 容差兜底） |
| 2026-09-07 | **性能基准结论（重要）** | 2000 个体 × 300 tick 基准：rust 3.043s vs python 3.078s = **1.01x**，行为位级一致；cProfile 定位瓶颈：`SphereWorld.neighbors()` 逐个体调用 190,722 次占 **~77%**（移动目标选择 + 邻格觅食两个 per-individual Python 循环，内部反复 flat_to_rc/clip/rc_to_flat），其次 numpy.clip 0.99s | 已下沉的向量化数值段本就便宜（numpy 已向量化），真正的热点是"逐个体循环里的邻居索引计算"；优化方向定为：**预计算全网格邻居表**（类成员缓存，一次算好 (n_cells, 8) int64 数组，循环内直接查行）——纯 Python 即可，不依赖 Rust |
| 2026-09-07 | 基准复跑确认（使用 venv 解释器） | 全量 pytest 22 例通过；基准复跑 python 3.030s vs rust 3.049s = **0.99x**（噪声内持平 = 1x），回归校验 PASS、行为位级一致 | 确认握手结论：Rust 下沉的 regrow/stage1/stage2 数值段无净收益，提速方向在预计算邻居表 |
| 2026-09-07 | 3.5 consume_many 下沉方式（进食/邻格觅食） | 引擎第 4 步进食 + 邻格补吃的两处批量消耗都走双路径：use_sim_core 时组好数组调 `sim_core.consume_many`（grid 就地改、实吃量写回 out_taken），否则走原来 `ResourceField.consume_many` | 与 regrow/stage1/stage2 同一接入模式；本轮不重跑基准（位置循环是瓶颈，不在进食段）——全量 pytest 30 例通过即为回归无破坏的证据 |
| 2026-09-07 | **预计算邻居表优化（已实施）** | 只在 `SphereWorld` 构造期动手：一次性算好 `_nb_table`(n_cells,8)（普通格 8 邻，含经度环绕/极点坍缩提前折算）与 `_pole_nb`(2,cols)（两极各一行相邻纬度带），`neighbors()` 从"每次 flat_to_rc+clip+rc_to_flat 现算"改为按行号查表返回（普通格直接取表行、极点格取 pole_nb 行）；引擎两处调用点零改动 | 构造成本一次性 ~毫秒级，却消掉引擎每 tick 数十万次的小 numpy 调用堆栈；查表与现算在全部 7200 格逐位一致（新 tests/test_sphere_world.py 用旧算法做参照对拍）；基准：python 3.030s→0.615s、rust 3.049s→0.569s ≈ **5x**，双引擎位级一致 REGRESSION PASS。当前 rust/python=1.08x，说明剩余瓶颈已在 Python 侧（RNG 消费/逐 tick 数组拼接），数值段与邻居段均已足够便宜，Rust 加速的收益到头了 |
| 2026-09-07 | 模块四统计口径（GenerationStats.trait_means 用什么） | 基因（14 位原值）+ 派生 trait（寿命/代谢倍率/成熟年龄，与引擎公式同源） | 直接观测表现型语义（人确认的推荐口径），派生列公式与引擎 `_lifespan`/代谢/成熟年龄同一来源 → 统计口径不会与行为脱节 |
| 2026-09-07 | 模块四范围（observatory 是否连带快照桥） | 连带快照桥一起（PROGRESS 待办口径） | 项目整体按"观察台 + 直播桥"一次交付，前端渲染再单独进模块五 |
| 2026-09-07 | 模块四 run_single 终止条件 | 手动逐 tick 循环里，每 tick 后执行 `engine._finished = engine._end_condition_met()`（与 `engine.run()` 语义对齐） | 适配时发现手动循环不置 `_finished` 会导致灭绝/跑满不停（浪费 CPU 且 ended_reason 误报 max_ticks）；对齐后灭绝走 stop_on_extinction 提前停 |
| 2026-09-07 | 球面版 resource_distribution 组实验裁剪 | 只保留 dist_uniform_sparse / dist_uniform_rich 两个均匀对照，删除旧版 patchy（斑块）实验 | 球面资源场（模块一）只支持均匀填充，斑块机制未实现；不预设机制，诚实标注（等资源场支持后补回） |
| 2026-09-07 | 快照桥消息设计 | tick/总能量/总资源/平均能量等标量 +（广播版）个体明细 {id, flat, energy, generation, age}；环形缓冲只存标量版 | 前端渲染直接消费明细；5000 个体 × 4096 份缓冲会 OOM，明细只在广播时带；新客户端连上先补发最新标量快照（落点晚也能看到画面） |
| 2026-09-07 | **L1 斑块资源守恒设计** | 两条守恒：①容量守恒（Σ_capacity 与 uniform 相等，bg_cap_mult=(total-patch_area×mult)/bg_area，必须>0）②再生守恒（面积加权 patch_mult×patch_frac+bg_mult×bg_frac=1）；初始食物总量**不守恒**（patch 格填满 capacity×1.0、背景格 capacity×background_fill，这是 patchy 的核心特征——斑块富集/背景贫瘠） | 容量/再生守恒保证种群承载上限和时间供给不变→不破坏平衡；初始食物不守恒是设计选择（两种世界的初始条件本就不同），在实验中如实标注。如需严格对比可调整 background_fill 使总量相等 |
| 2026-09-07 | **L1 patch 中心随机数隔离** | patch 中心选择用独立 rng（np.random.default_rng(patch_seed)），patch_seed=config.seed；不消费引擎 self.rng | 保证同 seed 下 uniform 与 patchy 引擎的个体初始基因/位置分布完全一致（RNG 消费顺序不变）→ 可复现、可对拍；测试 test_engine_patchy_does_not_consume_engine_rng 验证 |
| 2026-09-07 | **L1 patchy 模式 Rust regrow 回退** | patchy 时 `_advance_one_tick` 自动用 Python `ResourceField.regrow`（含空间倍率），uniform 保持双路径对拍 | Rust 侧 `sim_core.regrow` 暂未支持 patch_mask/空间倍率；patchy 模式回退 Python 保证行为正确，不影响默认 uniform 的 Rust 加速。后续 L6 可在 Rust 侧加 patch 支持 |
| 2026-09-07 | **L1 默认参数保守化** | patch_count=30, patch_radius=2, patch_capacity_mult=3.0, patch_regrowth_mult=2.0, background_fill=0.1 | 初版默认 60斑块/半径3/倍率6 导致背景容量为负（覆盖面积过大）；保守参数保证 bg_cap_mult>0、可直接跑通；用户可在 config 中调大 |
| 2026-09-07 | **L1 实验结果解读** | 2000tick 内空间聚集度 patchy(4.67) 低于 uniform(5.37)、基因多样性几乎相同——**不是失败，是预期** | L1 只加了资源斑块，没有感知基因（g14），生物无法感知斑块差异→不会主动聚集；空间 CV 高来自随机分布波动而非主动聚集。生态效应需等 L3 感知基因 + 更长时间（10000+tick）才能观测。L1 的价值是基础设施就绪+守恒验证通过 |