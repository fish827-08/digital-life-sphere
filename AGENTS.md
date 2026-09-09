# AGENTS.md — digital-life-sphere 项目代理指南

> 供 AI 代理/协作者快速了解项目约束、构建命令和关键流程。
> 详细模块说明见 `MODULES.md`。

## 项目概述

数字生命模拟系统：60×120 球面网格，Python 引擎 + Rust（PyO3）加速核。
目标：在有限死规则内让生命自发涌现智能与语言（不使用 LLM）。

## 技术栈

- Python 3.12 + NumPy
- Rust 1.81+ + PyO3（sim_core 扩展）
- pytest（测试）

## 构建命令

```bash
# 编译 Rust 扩展（release）
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/sim_core_target
export LIBRARY_PATH=/opt/python3.12/lib
cd sim_core && cargo build --release -j1
cp /tmp/sim_core_target/release/libsim_core.so ../sim_core.so

# 跑测试
cd .. && python3 -m pytest tests/ -q --ignore=tests/test_broker.py
```

## 关键约束

1. **双路径逐位对拍**：所有 Rust 下沉模块必须与 Python 参考实现逐位一致
   （函数级对拍 + 引擎级对拍）。RNG 消费顺序必须一致（预生成随机数传入 Rust）。
2. **基因索引不裸写**：用 `Gene.MOVE_PROB` 而非 `0`。Python↔Rust 双写由
   `validate_gene_wiring()` 校验，漂移则报错。
3. **独立分支**：新功能在 feature 分支开发，不直接碰 main。
4. **快照兼容**：`save_snapshot`/`load_snapshot` 保存 gene_count 和配置指纹，
   扩 gene_count 会破坏旧快照兼容性。

## 加基因五步曲（C3 G1）

新增基因位必须按顺序执行（详见 `MODULES.md` 模块五）：

1. Python `Gene` 枚举末尾追加 + `GENE_SEMANTICS` + `GENE_META` + `GENE_WIRED`
2. Rust `sim_core/src/genes.rs` 追加同名常量
3. `lib.rs` 的 `validate_gene_wiring` 中 `rust_all` 追加
4. 引擎用 `Gene.NEW_TRAIT` 消费（Rust 侧用 `crate::genes::G_NEW_TRAIT`）
5. 跑 `tests/test_genes_registry.py` + 全量测试 + 更新 PROGRESS.md

**禁止**：改已有基因索引、在枚举中间插入、只改 Python 不改 Rust。

## 预留位接管规则（C3 G4）

6 个预留位（g17/g18/g20/g21/g22/g23）优先复用，耗尽前不扩 gene_count。
扩位时触发存档格式升级提示。详见 `MODULES.md` 模块五 G4。

## D2 信息结构开发约定（2026-09-09）

D2 是语言涌现的核心重构，分支 `feat/info-structure`。四大机制：学习瓶颈 / 任意性码本 / 信息不对称 / Steels 对齐。

### 开发规则

1. **渐进式开启**：禁止一次性开启全部机制验证。必须按 学习瓶颈+Steels → +码本 → +信息不对称 顺序，每步确认生态稳定。
2. **D2 启用时走 Python 路径**：`use_sim_core=False`。Rust 侧暂未实现码本/学习瓶颈/softmax，后续逐步下沉，每步双路径对拍。
3. **配置总开关**：`InfoStructureConfig.enabled=False`（默认），关闭时行为与旧版完全一致。新增参数必须有默认值且 `from_dict` 旧存档回退。
4. **快照版本**：D2 后 `SNAPSHOT_VERSION=3`，新增 `codebook`/`learning_count`。旧快照加载时回退默认值。
5. **码本是 uint8 (N,16)**：初始恒等映射 `codebook[state]=state`。Steels 对齐用离散概率替换（不能直接浮点运算）。
6. **唯一验收门**：g15 无手调补贴从 0.5 上升（3 seed 一致）。其他指标（cult_div/trust/signal_density）为辅助观察。

### 实验工具

- `experiments/run_d2_experiment.py`：D2 参数调优专用脚本，支持四大机制独立开关 + 全部参数 CLI 透传 + 自动 manifest + CSV。
- 详细调优方案见 `评估-EVAL-D2参数调优实验指南-20260909.md`。

### 禁止事项

- 禁止在 D2 启用时调用 Rust 路径的 `step_movement`/`signal_emit`/`reproduce_batch`（会忽略码本/学习瓶颈）。
- 禁止把码本当浮点数组做 EWMA（uint8 离散映射，用概率替换）。
- 禁止修改 `perception_radius` 为非 4/6/8 的值（邻居索引逻辑只支持这三档）。

## 测试状态

- 全量：192 passed, 1 skipped（含 17 例 D2 单元测试 + 双路径逐位对拍 + 快照 + 信号 + 愉悦度 + 繁殖 + L10a + 基因注册表）
- D2 信息结构：17/17（配置/数据结构/学习瓶颈/码本/信息不对称/Steels对齐/快照v3/向后兼容）
- 快照：10/10（save/load + RNG 可复现 + 多次循环 + v3 兼容）
- 信号发射：6/6（函数级对拍）
- 愉悦度：5/5（函数级对拍）
- 繁殖下沉：3/3（函数级对拍，T4）
- 果实-种子传播：6/6（函数级对拍，L10a）
- 基因注册表：18/18（双写漂移检测 + 元数据完整性）

## 当前阶段（2026-09-08，详见 后续发展路线.md）

1. **性能线**：T1~T4 下沉全部并入；T5 slots 预研完成（评估结论：暂不实现）；A3 敏感性扫描实验由云端运行中，产出 results/sensitivity/
2. **扩展特性优先级**：L9 尸体能量守恒 → L10 种子传播（先行版已并入）→ L8 地形
3. **分支结构**：远程单一 main 主干；云端后续任务开新 feature 分支开发，经审查合并后并入 main
4. **L7d Rayon**：暂缓；触发条件 = 单线程 60 万 tick > 2h 且 T1~T4 合入（当前 30 万 tick=33 min，未到）
5. 评估材料 = 评估材料打包清单.md（12 项，9 项完成，A3 产出后补齐 #9）
6. 实验记录：30 万 tick L4+L5 长实验已完成（见 PROGRESS.md），数据 experiments/long_run_l4l5_result.csv
