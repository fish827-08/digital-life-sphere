# T5 Slots 预分配模式评估报告

> 任务来源：TASKS-DEV.md §一 T5 / TASKS-CLOUD.md 第二轮 C6
> 评估日期：2026-09-08
> 结论：**暂不实现**（收益 <10%，RNG 顺序风险高，建议优先优化其他瓶颈）

---

## 一、当前数组管理方式

### 1.1 现状

`SphereEngine` 使用**动态数组 + np.concatenate/np.delete** 管理种群：

- 所有个体数组（`_id`, `_flat`, `_energy`, `_genes`, `_age`, `_generation`,
  `_valence`, `_arousal`, `_expectation`, `_trust`, `_work_memory`, `_interpret`,
  `_fruit_charge`, `_seed_carried` 等共 **20+ 个数组**）长度 = 当前存活个体数 P。
- **繁殖**：`np.concatenate([原数组, 子代数组])`，子代追加到末尾。
- **死亡**：`keep = ~dead; arr = np.concatenate([arr[:P][keep], arr[P:]])`，
  先筛选前 P 个亲代的存活者，再拼接子代（子代不参与本 tick 死亡判定）。

### 1.2 性能开销实测

环境：N=1467（N=5000 时线性外推），20 个数组，每 tick 繁殖 K≈10 / 死亡 D≈5。

| 操作 | N=1467 | N=5000（外推） | 占总 tick 时间比 |
|------|--------|----------------|-----------------|
| np.concatenate（繁殖，20数组） | 0.04 ms | 0.13 ms | ~2% |
| 布尔索引删除（死亡，20数组） | 0.04 ms | 0.14 ms | ~2% |
| **合计** | **0.08 ms** | **0.27 ms** | **~3-4%** |

> 总 tick 时间基准：N=5000, use_sim_core=True 时约 4-8 ms/tick（125-250 tick/s）。

**关键发现**：数组管理开销占比仅 3-4%，即使完全消除也只能提升 ~4% 性能。

---

## 二、Slots 预分配方案设计

### 2.1 核心思路

预分配 `N_max`（=种群上限 5000）大小的数组，用 `alive: np.ndarray[bool]` 标记存活个体：

```
数组布局：[slot 0][slot 1] ... [slot N_max-1]
alive:    [ True ][ False ] ... [ True  ]
                    ↑ 死亡槽，可被新个体复用
```

- **死亡**：`alive[slot] = False`，将 slot 加入空闲链表。
- **繁殖**：从空闲链表取 slot，写入子代数据，`alive[slot] = True`。
- **无 np.concatenate/np.delete**，数组大小固定为 N_max。

### 2.2 存活性管理算法

**方案 A：空闲栈（LIFO）**
```python
self._free_slots = []  # 空闲槽索引栈
# 死亡
self._free_slots.append(dead_slot)
self._alive[dead_slot] = False
# 繁殖
if self._free_slots:
    new_slot = self._free_slots.pop()
else:
    new_slot = self._next_slot  # 顺序分配（种群增长期）
    self._next_slot += 1
```
- 优点：O(1) 分配/释放，实现简单。
- 缺点：空闲槽顺序 = 死亡逆序，可能影响 RNG 消费顺序。

**方案 B：空闲位图 + 顺序扫描**
```python
self._free_mask = np.ones(N_max, dtype=bool)  # True=空闲
# 繁殖：找第一个空闲槽
new_slot = np.argmax(self._free_mask)
self._free_mask[new_slot] = False
```
- 优点：槽位顺序分配（与当前追加到末尾的行为更接近）。
- 缺点：np.argmax 是 O(N_max)，每次繁殖扫描 5000 元素，开销可能超过 np.concatenate。

**推荐方案 A**（空闲栈），O(1) 开销。

### 2.3 死亡槽复用策略

- 每 tick 死亡判定后，将死亡槽加入空闲栈。
- 繁殖时从空闲栈取槽。如果空闲栈为空（种群增长期），用顺序分配器 `_next_slot`。
- 子代写入新槽的所有数组字段（基因/能量/位置/年龄/愉悦度/信任等）。
- `alive[new_slot] = True`。

---

## 三、与 RNG consume 顺序的交互风险（关键风险）

### 3.1 当前 RNG 消费顺序

引擎使用单一 `self.rng`（np.random.Generator），所有随机数按代码执行顺序消费：

```
tick 开始
  → 步骤 1-3: 光合/代谢/维持（无随机）
  → 步骤 4: 进食（无随机）
  → 步骤 4.5: 信号发射（rand_emit = self.rng.random(P)）
  → 步骤 5: 移动（moved_raw = self.rng.random(P)）
  → 步骤 6: 捕食（rand_attack = self.rng.random(P)）
  → 步骤 7: 死亡判定（无随机）
  → 步骤 8: 繁殖（rand_repro = self.rng.random(P)，基因突变 self.rng.normal(...)）
  → 步骤 9: 清理尸体（无随机）
```

**关键**：随机数消费顺序与**个体在数组中的顺序**直接相关。例如 `rand_emit = self.rng.random(P)` 生成 P 个随机数，第 i 个随机数对应数组第 i 个个体。

### 3.2 Slots 方案的 RNG 风险

**风险 1：子代槽位不连续 → 后续 tick 的随机数映射改变**

当前方案：子代追加到数组末尾，数组顺序 = 出生顺序。下 tick 的 `self.rng.random(P)` 中，前 P_old 个随机数对应亲代，后 K 个对应子代。

Slots 方案：子代可能写入中间的空闲槽（死亡槽复用），数组顺序不再 = 出生顺序。下 tick 的 `self.rng.random(P)` 中，随机数与个体的对应关系完全改变。

**后果**：即使使用相同 seed，slots 方案与当前方案的模拟轨迹会从第一次繁殖复用死亡槽起完全分叉。**无法做双路径对拍**（当前方案 vs slots 方案的结果不可能一致）。

**风险 2：繁殖时的基因突变随机数消费顺序**

当前繁殖：`child_genes = parent_genes + self.rng.normal(0, mutation_std, (K, gene_count))`，K 个子代连续消费 K×gene_count 个随机数。

Slots 方案：如果子代写入不同槽位，基因突变的随机数消费顺序不变（仍然是连续 K 个），但子代在数组中的位置改变，影响后续 tick 的随机数映射。

**风险 3：空闲栈的 LIFO 顺序不可预测**

死亡槽的复用顺序 = 死亡逆序（LIFO），这取决于每 tick 的死亡模式，难以预测和控制。

### 3.3 风险缓解方案

**方案 1：保持子代追加到末尾，死亡槽仅用于"逻辑删除"**
- 数组仍然动态增长（np.concatenate），但死亡个体不删除，只标记 alive=False。
- 统计计算用 alive 掩码筛选。
- 问题：数组只增不减，长时间运行后内存膨胀（死亡个体占槽但不释放）。
- 需要周期性"压缩"（移除死亡个体），但压缩时又会改变数组顺序 → RNG 分叉。

**方案 2：每 tick 结束后重排数组，保持出生顺序**
- 繁殖时子代写入空闲槽，tick 结束后将所有存活个体按出生顺序重排到数组前部。
- 重排 = 一次 O(N) 拷贝，开销与当前 np.concatenate 相当。
- 问题：重排改变了个体在数组中的位置，但如果重排规则确定（按出生时间排序），RNG 消费顺序可以保持一致。
- 实现复杂度高，且重排开销抵消了 slots 的收益。

**方案 3：接受 RNG 分叉，slots 作为独立模式（不做对拍）**
- slots 模式下用相同 seed 跑，结果与当前模式不同，但 slots 模式自身可复现。
- 需要单独验证 slots 模式的正确性（单元测试 + 生态等价性测试，而非逐位对拍）。
- 问题：维护两套数组管理逻辑，代码复杂度翻倍。

---

## 四、性能预期

| 指标 | 当前（np.concatenate） | Slots 方案（预期） | 提升 |
|------|----------------------|-------------------|------|
| 数组管理开销/tick | 0.27 ms（N=5000） | ~0.01 ms（空闲栈 O(1)） | -0.26 ms |
| 总 tick 时间 | 4-8 ms | 3.7-7.7 ms | ~3-7% |
| tick/s（N=5000） | 125-250 | 129-270 | ~3-7% |
| 内存分配频率 | 每 tick 20+ 次数组分配 | 0（预分配） | 显著降低 GC 压力 |

**额外收益**：
- 消除频繁内存分配，降低 GC 压力和内存碎片。
- 数组大小固定，缓存命中率可能提升（但 numpy 数组本身已是连续内存，提升有限）。

**额外开销**：
- 所有统计计算需要 `alive` 掩码筛选（如 `_energy[:P][alive[:P]]`），增加一次布尔索引。
- 空闲栈的 Python 层操作（list append/pop），每 tick 几十次，开销约 0.01ms。
- `_expectation` 是 (N_max, 120) 二维数组，预分配 5000×120 = 60 万元素 = 4.8 MB，可接受。

---

## 五、结论与建议

### 5.1 结论

**暂不实现 slots 预分配模式**，理由：

1. **收益有限**：数组管理开销仅占总 tick 时间的 3-4%，完全消除也只能提升 ~4% 性能。
2. **RNG 顺序风险高**：slots 方案改变个体在数组中的顺序，导致与当前方案无法逐位对拍，且所有依赖数组顺序的随机数消费都会分叉。
3. **实现复杂度高**：需要管理空闲栈、alive 掩码、所有数组的槽位写入、统计计算的掩码筛选，代码改动面大（涉及 sphere_engine.py 几乎所有步骤）。
4. **维护成本高**：两套数组管理逻辑（当前动态 + slots）需要并行维护，测试覆盖翻倍。

### 5.2 建议优先优化的瓶颈

按收益排序，建议优先优化以下方向（均比 slots 收益高）：

| 优先级 | 方向 | 预期收益 | 难度 |
|--------|------|---------|------|
| 1 | **邻居表查找向量化**：当前移动/捕食中邻居遍历可能是 Python 循环，下沉 Rust | 10-30% | 中 |
| 2 | **资源场再生下沉 Rust**（L7e 已部分完成） | 5-15% | 低 |
| 3 | **文化学习/信任更新向量化**：减少 Python 层循环 | 5-10% | 中 |
| 4 | **繁殖步骤下沉 Rust**（T4 已完成核心，待 Python 集成） | 5-10% | 中 |
| 5 | **Slots 预分配**（本评估） | 3-7% | 高 |

### 5.3 未来重新评估的条件

当出现以下情况时，可重新评估 slots 方案：

1. **N_max 大幅提升**（如从 5000 提升到 20000）：数组管理开销线性增长，slots 收益比例提升。
2. **繁殖/死亡率大幅提升**（如高捕食压力环境）：每 tick np.concatenate/delete 次数增加，slots 收益提升。
3. **其他瓶颈全部优化完毕**：当数组管理成为最大剩余瓶颈时，slots 的相对收益提升。
4. **接受 RNG 分叉**：如果项目决定 slots 作为独立模式（不要求与当前模式对拍），可以实现。

---

## 六、附录：实现 slots 的最小改动清单（如未来决定实现）

如果未来决定实现 slots 方案，最小改动包括：

1. **`__init__`**：所有数组预分配 N_max，新增 `_alive: np.ndarray[bool]`、`_free_slots: list[int]`、`_next_slot: int`。
2. **`_step_population`**：
   - 死亡判定后：`_alive[dead_idx] = False; _free_slots.extend(dead_idx)`。
   - 繁殖时：从 `_free_slots` 取槽，写入子代所有数组字段，`_alive[new_slot] = True`。
   - 移除所有 `np.concatenate` 和死亡清理代码。
3. **所有步骤**：用 `alive[:P]` 掩码筛选存活个体（如 `energy[alive[:P]]`）。
4. **统计计算**：所有 `mean/sum/max` 等统计用 `alive` 掩码。
5. **RNG**：接受与当前方案的分叉，slots 模式自身可复现（相同 seed → 相同结果）。
6. **测试**：新增 slots 模式的单元测试（正确性）+ 生态等价性测试（与当前模式的终局统计分布一致，而非逐位对拍）。

预估工作量：3-5 天（含测试）。
