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

## G5 工程契约（D5 复现性基建，2026-09-09 固化）

### G5.1 Python 参考实现降级为黄金测试预言机（不删除）

`simulation/sphere_engine.py` 中的纯 Python 实现是**黄金参考（oracle）**，永远不删除、不简化。

- Rust 下沉的正确性唯一验证手段是与 Python 参考实现逐位对拍。
- 性能优化只能在 Rust 侧做，Python 侧保持可读、可审计、作为规范。
- 对拍不一致时 **Rust 侧必须改到与 Python 一致**，除非有明确数学理由并经审查。
- `use_sim_core=False` 路径必须始终可运行、可测试。

### G5.2 RNG 用预生成数组按索引消费（禁用每线程独立 rng）

所有随机数消费必须通过**预生成数组 + 索引**方式，禁止在循环/线程内创建独立 rng。

- 每 tick 开始时预生成该 tick 所需全部随机数数组（`rng.random(P)` 等）。
- 个体级随机消费通过数组索引（`rand_arr[i]`），禁止循环内调用 `rng.random()`。
- Rust 侧接收 Python 预生成数组，禁止在 Rust 内创建 `rand::thread_rng()`。
- 新增随机消费点必须在预生成阶段新增对应数组，并验证双路径 RNG 消费顺序一致。
- "独立随机"需求（如 `signal_mode=random`）用固定种子偏移的独立 rng 预生成，不消费主 rng。

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

## 测试状态

- 全量：152 passed, 1 skipped（双路径逐位对拍 + 快照 + 信号 + 愉悦度 + 繁殖 + L10a 果实-种子 + 基因注册表）
- 快照：10/10（save/load + RNG 可复现 + 多次循环）
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
