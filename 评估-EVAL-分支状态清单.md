# digital-life-sphere 分支状态清单（供外部评估与上云）

> 生成时间：2026-09-08 | 说明：各功能分支**暂不合并**，按现状上云；每分支标注功能、状态、测试与差异。

---

## 分支总览

| 分支 | 基线 | 状态 | 功能 | 测试 |
|------|------|------|------|------|
| `main` | 946b5ff | ✅ 主干，164 passed 1 skipped | 全部合入功能（L1~L7e、T1~T4、C1/C2/C3/C5/C6、快照 v2、大小世界） | 164 passed 1 skipped |
| `feature/l7d-rayon` | 与 main 同端（ahead=0，已合入） | ⚪ 已完成并合入 main | L7d：4 个数值函数 LightTempRust/pleasure/regrow/stage1 的 Rayon 并行 + 大世界支持（200×400） | 已含在 main |
| `feature/l9-terrain-season` | ahead of main ×1 | 🟢 新功能（未合） | **L8 地形**（水域不可通行/山地耗能高）+ **季节系统**（year_length 世界年 + axial_tilt 黄赤交角太阳直射摆动，SeasonConfig）+ **L9 尸体能量守恒** | tests/test_l9_terrain_season.py 278 行；启用季节/地形时 Rust 自动回退 Python |
| `feature/language-signal` | ahead of main ×1 | 🟢 新功能（未合） | 语言/信号机制增强（config + engine 增量：信号相关选择压参数化） | 未单独建测试（增量 37 行） |
| `feature/snapshot-cron` | 与 main 同端（ahead=0，已合入） | ⚪ 已完成并合入 main | 快照机制 v2（save/load + RNG 复现 + 续跑 + cron 分段） | 已含在 main |

## 各分支差异明细

### feature/l9-terrain-season（+244 行文档，+278 行测试，config+93，engine+152，light_and_temperature+88）
- `SeasonConfig`：`year_length=3`（一年=3 个世界日，平均寿命≈1.5年，最长≈2.7年）；`axial_tilt=0.4`（黄赤交角，太阳直射纬度 = axial_tilt×sin(2π×tick/年)）
- `TerrainConfig`：地形静态空间异质，水域不可通行、山地移动能耗高
- `CarcassConfig`：L9 尸体能量守恒（死亡→肥料→能量回流）
- **兼容保障**：旧存档配置缺省回退（enabled=False）；启用 L8/L9/季节时引擎自动回退 Python 路径（Rust 暂未下沉）

### feature/language-signal（config+13，engine+27）
- 信号系统选择压参数化（配合隐式选择压审计 A3 敏感性扫描结果的中性化建议）
- 增量未建独立测试（与 main 的 164 例共存）

## 划分子说明（各功能目前状态）

| 功能 | 状态 |
|------|------|
| 球面世界 + 光照昼夜温度 | ✅ main |
| 24 基因（18 接线） | ✅ main（g17/g18/g20~23 预留） |
| L1 斑块资源 | ✅ main |
| L2 信号场+愉悦度 RPE | ✅ main |
| L3 感知 g14 + 信号 g15 | ✅ main |
| L4 捕食 g16 + 植物化 g19 | ✅ main |
| L5 信任+工作记忆+文化传递 | ✅ main |
| L6 系列 Rust 下沉（T1~T4 + C1/C2/C5） | ✅ main |
| L7d Rayon 并行（4 函数） | ✅ main |
| 大世界 200×400 支持 | ✅ main |
| 快照 v2 + 续跑 | ✅ main |
| L7f LightTemp Rust 并行 | ✅ main |
| L8 地形 | 🟢 l9-terrain-season（未合） |
| L9 尸体能量守恒 | 🟢 l9-terrain-season（未合） |
| 季节系统（四季） | 🟢 l9-terrain-season（未合） |
| 语言/信号增强参数化 | 🟢 language-signal（未合） |
| 前端渲染（React+globe.gl） | ⏳ 未启动（模块五） |
| 有性繁殖/社会结构/工具使用 | ⏳ 远期（L11+，未排期） |

---
*配合 评估-EVAL-综合报告-20260908.md 使用*