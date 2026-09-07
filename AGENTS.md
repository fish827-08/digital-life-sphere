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

## 测试状态

- 全量：~109 passed, 2 failed（已知 Rust release 1ULP 浮点误差：regrow sens=1.0 的 [37][200]）
- 快照：10/10（save/load + RNG 可复现 + 多次循环）
- 信号发射：6/6（函数级对拍）
- 愉悦度：5/5（函数级对拍）
- 基因注册表：18/18（双写漂移检测 + 元数据完整性）
