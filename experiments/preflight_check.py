"""Pre-Flight 检查工具（R61 纪律 + R78-3 扩展 + C5 规格自洽）—— 实验启动前的机械化自查。

为什么需要它
------------
R61 立规「实验前置检查」后，检查单靠人肉执行；本项目已发生多起**静默配置事故**
（F-R9：`--arm control` 未关码本 ⇒ control 与 main 同轨迹整批作废；D2 年龄永不推进；
`radius=6` 静默 no-op）。它们的共同特征是**配置看似生效、实则没生效**，且
**跑完才知道**。本工具把可机械化的检查项固化成代码，跑前 1–3 分钟即可拦截。

**C4+ / C5 是两次"仪器化之后才暴露"的追加**：
- **C4+（R78-3）**：源自外部 issue 的跨线佐证——分析器硬编码世界种子 ⇒ `std ≡ 0` 假稳定；
- **C5（2026-09-15）**：源自 D-24 的 G-A 不过——**V-1 规格自身把 `donation` 设成 0.05 而 `SIGNAL_COST` 是 0.1**，
  于是 `return_ratio` 的结构性上限 = `donation/SIGNAL_COST` = **0.5 < 1.0**，**"保本"语义在数学上不可达**。
  即：**配置全都正确生效了（C4 全过），但它们之间自相矛盾（C5 才管）**。

覆盖的检查项
------------
| 项 | 内容 | 对应纪律 |
|----|------|---------|
| **C3** | **臂间轨迹差异断言**：同 seed 下不同臂的轨迹若**逐字段完全一致** ⇒ 立即停（F-R9 症状） | R61-C3 |
| **C4** | **开关读回核对**：从 summary.switches 读回真实开关状态，与臂语义逐条核对 | R61-C4 |
| **C4+** | **世界种子敏感性断言**（R78-3）：同模型 ≥3 世界种子，关键指标 std ≡ 0 ⇒ **拒判** | R78-3 |
| **C5** | **规格自洽检查**：参数**之间**的关系能否表达所声明的语义（如"保本"要求 `donation ≥ SIGNAL_COST`）⇒ 不能 ⇒ `OracleConfig.__post_init__` 直接报错 | **C5（2026-09-15）** |

设计原则
--------
1. **驱动真实 CLI**（`a4_verify_capacity.py`），不复制 arm→开关 映射 —— 该映射
   出过一次事故（F-R9），必须只存在一处（runner 的 `main()`）。本工具只解释输出。
2. **纯观测**：不写 `simulation/`、不改任何既有产物；输出落在 `--workdir`。
3. **可单测**：比较/断言逻辑抽为纯函数（`tests/test_preflight_check.py`）。

用法
----
    # 默认：main vs control（F-R9 的那一对）× 3 seed × 1000 tick
    python experiments/preflight_check.py

    # 跑全 5 臂 / 自定义
    python experiments/preflight_check.py --arms main,control,zero,sigoff,oracle \\
        --seeds 42,43,44 --ticks 1000 --max-count 3240

    # 只做开关读回（不跑模拟，读已有 summary）
    python experiments/preflight_check.py --readback-only --workdir _rerun_logs/d24

退出码：0 = 全部检查通过（可启动实验）；1 = 有断言失败（**不得启动**）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 臂语义（期望的开关状态）—— 仅用于**读回核对**，不作为运行参数（参数由 CLI 决定）
EXPECTED_SWITCHES = {
    "main":    {"arbitrary_codebook": True,  "neutral_genes": False, "signal_disabled": False, "oracle_enabled": False},
    "control": {"arbitrary_codebook": False, "neutral_genes": False, "signal_disabled": False, "oracle_enabled": False},
    "zero":    {"arbitrary_codebook": True,  "neutral_genes": True,  "signal_disabled": False, "oracle_enabled": False},
    "sigoff":  {"arbitrary_codebook": True,  "neutral_genes": False, "signal_disabled": True,  "oracle_enabled": False},
    "oracle":  {"arbitrary_codebook": True,  "neutral_genes": False, "signal_disabled": False, "oracle_enabled": True},
}

COMPARE_FIELDS = ("N", "g14", "g15", "trust", "max_gen", "max_gen_cur")

# 信号发射成本（单次发射扣费）——**从单一真源取，不再本地抄一份**（C5）。
# 此前本文件抄了一份字面量；连同引擎两处局部字面量与 oracle.EMISSION_COST，同源值共 **4 处声明**。
# 现统一：`simulation/config.py` 的 `SIGNAL_COST` 是唯一真源，此处只引用。
from simulation.config import SIGNAL_COST  # noqa: E402

# 容差：C5 用浮点比较（与常见档位 0.02/0.05/0.1/0.5/0.7 相比，1e-9 足够严、又不误报）
_C5_TOL = 1e-9

# 正向选择压的建议裕度（联网线「保本中性边界」预警，2026-09-15）：
# `ratio = donation/SIGNAL_COST = 1.0` 时 sender 期望净收益 = 0 ⇒ 选择差 = 0 ⇒ 漂变主导
# ⇒ 阳性对照需要 **严格为正** 的选择差。1.2 是建议下限（不硬拦，见 spec_consistency_problems）。
POSITIVE_CONTROL_MIN_RATIO = 1.2


# ---------------------------------------------------------------- 纯函数（可单测）

def compare_trajectories(rows_a: list[dict], rows_b: list[dict],
                         fields: tuple[str, ...] = COMPARE_FIELDS) -> tuple[bool, int | None]:
    """两条 CSV 轨迹是否**逐字段完全一致**（F-R9 症状）。

    返回 (是否完全一致, 首个分歧 tick)。只在两文件共有的 tick 上比较；
    行数不同也视为"不一致"（分歧 tick = 较短者的末尾后一 tick）。
    """
    ta = {int(r["tick"]): r for r in rows_a}
    tb = {int(r["tick"]): r for r in rows_b}
    common = sorted(set(ta) & set(tb))
    for t in common:
        for f in fields:
            va, vb = ta[t].get(f), tb[t].get(f)
            if va != vb:
                return False, t
    if len(ta) != len(tb):
        return False, None  # 行数不同（一方提前结束）⇒ 不一致
    return True, None


def seed_sensitivity(values: list[float]) -> tuple[bool, float]:
    """世界种子敏感性（R78-3）：std ≡ 0 ⇒ 拒判。

    返回 (是否通过, std)。values = 同一模型在不同世界种子下的同一指标。
    """
    import numpy as np

    if len(values) < 2:
        return False, 0.0          # 样本不足本身就不能断言敏感
    sd = float(np.std(np.asarray(values, dtype=np.float64)))
    return sd > 1e-12, sd


def readback_problems(arm: str, switches: dict) -> list[str]:
    """开关读回核对（R61-C4）：summary.switches 是否与臂语义一致。"""
    exp = EXPECTED_SWITCHES.get(arm)
    if exp is None:
        return [f"未知臂名 {arm}（无法核对）"]
    return [f"{k}: 期望 {v}，实测 {switches.get(k)!r}"
            for k, v in exp.items() if switches.get(k) != v]


def signal_cost_ceiling(donation: float, signal_cost: float = SIGNAL_COST) -> float:
    """单次发射可获得的**结构性上限**回馈比 = `donation / signal_cost`。

    C-9 把"单次发射的累计回馈"封顶在 `SIGNAL_COST`（保本语义）⇒ 因此
    `oracle_return_ratio` 永远不可能超过这个比值。它**与生态/演化无关，是纯代数上界**。
    """
    if signal_cost <= 0:
        return float("inf")
    return float(donation) / float(signal_cost)


def spec_consistency_problems(switches: dict, *, tolerance: float = _C5_TOL) -> list[str]:
    """**C5 规格自洽检查**（2026-09-15）：参数**之间**的关系能否表达所声明的语义。

    与 C4 的分工（关键区别）
    ------------------------
    - **C4** 问：*每个开关是否真的生效了？*（"声明 = 实际"）
    - **C5** 问：*生效了的这些值，彼此之间是否自洽？*（"参数关系 = 所声明的语义"）

    事故原型（D-24 / G-A 不过）：V-1 规格把 `donation` 定为 0.05，而 `SIGNAL_COST` 是 0.1
    ⇒ `return_ratio` 的结构性上限 = 0.5 < 1.0 ⇒ **"保本"语义在数学上不可达**。
    **C4 全过（每个开关都确实生效了），但规格自相矛盾** —— 这正是 C5 存在的理由。

    检查项
    ------
    1. `oracle_enabled` 且 `donation < SIGNAL_COST` ⇒ 违规（保本不可达）
       - 违规时 `simulation/config.py` 的 `OracleConfig.__post_init__` **直接报错、拒绝运行**
       - 本函数是**独立复算**：即使有人放宽了那个断言，这里仍会拦下（双保险）
       - 若某批**有意**测"补贴不足"档，可显式放宽该断言（届时开关会被读回并如实记录）
    2. `donation < 0` ⇒ 违规
    3. `oracle_enabled` 为假但 `donation > 0` ⇒ **仅提示**（关闭时行为应逐位一致，不受影响）
    4. 🔴 **保本中性边界（advisory，2026-09-15）**：`ratio = donation/SIGNAL_COST == 1.0` 时
       sender **期望净收益 = 0 ⇒ 选择差 = 0 ⇒ 漂变主导** ⇒ 跨 seed CI 结构性易含 0。
       ⇒ 若该臂要当**阳性对照**，`ratio` 应 **> 1**（建议 ≥ `POSITIVE_CONTROL_MIN_RATIO`）。
       **不硬拦**——因为 R86 的剂量点 `1.0` 本身要跑（它正是用来定位真实保本点的）。
    """
    probs: list[str] = []
    d = switches.get("donation")
    if d is None:
        return probs                      # 非 oracle 批不带此开关 ⇒ 无从检查
    try:
        d = float(d)
    except (TypeError, ValueError):
        return [f"donation 不是数值：{switches.get('donation')!r}"]

    enabled = bool(switches.get("oracle_enabled"))
    waived = bool(switches.get("allow_non_breakeven"))
    if d < 0:
        probs.append(f"donation={d} 为负（C-3 要求非负）")
    if enabled and d < SIGNAL_COST - tolerance and waived:
        # 显式逃生阀：须在报告中留痕，但放行（"有意研究补偿不足区制"）
        print(f"  ℹ️ allow_non_breakeven=True ⇒ 放行 donation={d} < SIGNAL_COST"
              "（本批声明要研究补偿不足区制；结果不得表述为保本/补偿）")
        return probs
    if enabled and d < SIGNAL_COST - tolerance:
        ceil = signal_cost_ceiling(d)
        probs.append(
            f"**【保本语义不可达】** oracle 已启用，但 donation({d}) < SIGNAL_COST({SIGNAL_COST}) "
            f"⇒ return_ratio 结构性上限 = {ceil:.4f} < 1.0 ⇒ 无论生态如何演化都不可能保本 "
            f"（D-24 G-A 不过的根因；修复=反解 donation ≥ SIGNAL_COST/转化率）"
        )
    elif enabled:
        # 保本已达标，但保本 ≠ 阳性：ratio==1 ⇒ sender 期望净收益 0 ⇒ 选择差 0 ⇒ 漂变主导
        # （联网线 2026-09-15「保本中性边界」预警；Lewis 原型成功时 sender 净得 +1，严格为正）
        ceil = signal_cost_ceiling(d)
        if ceil <= 1.0 + tolerance:
            print(f"  ⚠️ 保本中性边界：ratio = {ceil:.4f} == 1.0 ⇒ sender 期望净收益 = 0 "
                  f"⇒ 选择差 = 0 ⇒ 漂变主导，跨 seed CI **结构性易含 0**。"
                  f"若该臂要作阳性对照，建议 ratio ≥ {POSITIVE_CONTROL_MIN_RATIO}"
                  f"（R86 的 1.0 剂量点保留：它正是用来定位真实保本点的）")
    return probs


# ---------------------------------------------------------------- 运行与检查

def _load_rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def _load_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_matrix(arms: list[str], seeds: list[int], ticks: int, max_count: int,
               workdir: Path, python: str) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    # ⚠️ 本工具自身的静默事故教训（2026-09-14 实跑发现）：
    #   默认快照目录 `_rerun_logs/snap/` 里已有同名 run 的快照（如 D-24 的 main_s42.snapshot.npz），
    #   若不强制 --fresh，短跑会"从这个 30000+ tick 的快照续跑" ⇒ `--ticks 1000` 时循环体为空、
    #   CSV 只剩表头 ⇒ 检查静默失效（且结果看起来"正常"）。
    #   ⇒ 双保险：**强制 --fresh** + **独立快照目录**（绝不触碰他批快照）。
    snap_dir = workdir / "_snap"
    for arm in arms:
        for seed in seeds:
            out = workdir / f"{arm}_s{seed}.csv"
            if out.exists() and out.with_suffix(".summary.json").exists():
                continue
            cmd = [python, str(ROOT / "experiments" / "a4_verify_capacity.py"),
                   "--mode", "on", "--arm", arm, "--seed", str(seed),
                   "--ticks", str(ticks), "--max-count", str(max_count),
                   "--snapshot-every", "0", "--fresh",
                   "--snapshot-dir", str(snap_dir), "--out", str(out)]
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=str(ROOT),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  {'✅' if rc == 0 else '❌'} {arm}_s{seed}  rc={rc}  "
                  f"{time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-Flight 检查（R61 + R78-3）")
    ap.add_argument("--arms", default="main,control")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ticks", type=int, default=3000,
                    help="短跑 tick 数；≥3000 才有 ≥3 个采样点，C3 差异断言才有判别力")
    ap.add_argument("--max-count", type=int, default=3240)
    ap.add_argument("--workdir", default="_rerun_logs/preflight")
    ap.add_argument("--python", default=None)
    ap.add_argument("--readback-only", action="store_true",
                    help="不跑模拟，只对既有 summary 做开关读回核对（如对 D-24 批次）")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wd = Path(args.workdir)
    if not wd.is_absolute():
        wd = ROOT / wd
    py = args.python or str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not Path(py).exists():
        py = sys.executable

    print(f"Pre-Flight 检查 | 臂={arms} | 种子={seeds} | tick={args.ticks} "
          f"| max_count={args.max_count} | 目录={wd}")
    if not args.readback_only:
        print("① 跑矩阵：")
        run_matrix(arms, seeds, args.ticks, args.max_count, wd, py)

    failures: list[str] = []
    report: list[str] = ["# Pre-Flight 检查报告", "",
                         f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                         f"- 臂：{arms} / 种子：{seeds} / tick：{args.ticks}", ""]

    # ---- C4：开关读回核对 ----
    print("\n② C4 开关读回核对：")
    report += ["## C4 开关读回核对", "", "| run | 结果 | 问题 |", "|---|---|---|"]
    for arm in arms:
        for seed in seeds:
            sw = (_load_summary(wd / f"{arm}_s{seed}.summary.json") or {}).get("switches")
            if sw is None:
                msg = "缺 summary"
                failures.append(f"C4 {arm}_s{seed}: {msg}")
                report.append(f"| {arm}_s{seed} | ❌ | {msg} |")
                continue
            probs = readback_problems(arm, sw)
            ok = not probs
            if not ok:
                failures.append(f"C4 {arm}_s{seed}: {probs}")
            print(f"  {'✅' if ok else '❌'} {arm}_s{seed} {probs if probs else ''}")
            report.append(f"| {arm}_s{seed} | {'✅' if ok else '❌'} | {'; '.join(probs)} |")

    # ---- C5：规格自洽（参数之间的关系能否表达所声明的语义）----
    print("\n②b C5 规格自洽检查（参数关系是否自相矛盾）：")
    report += ["", "## C5 规格自洽检查", "",
               "> C4 问「开关有没有生效」；C5 问「生效的这些值彼此自洽吗」。", "",
               "| run | 结果 | 问题 |", "|---|---|---|"]
    for arm in arms:
        for seed in seeds:
            sw = (_load_summary(wd / f"{arm}_s{seed}.summary.json") or {}).get("switches")
            if sw is None:
                continue                       # 缺 summary 已在 C4 段登记，此处不重复计
            probs = spec_consistency_problems(sw)
            ok = not probs
            if not ok:
                failures.append(f"C5 {arm}_s{seed}: {probs}")
            print(f"  {'✅' if ok else '❌'} {arm}_s{seed} {probs if probs else ''}")
            report.append(f"| {arm}_s{seed} | {'✅' if ok else '❌'} | {'; '.join(probs)} |")
    # 顺带报一次结构性上限（供判读直接用，即使本批开关全自洽）
    _d = (_load_summary(wd / f"oracle_s{seeds[0]}.summary.json") or {}).get("switches", {}).get("donation")
    if _d is not None:
        _ceil = signal_cost_ceiling(float(_d))
        print(f"  ℹ️ 结构性上限 donation/SIGNAL_COST = {float(_d)}/{SIGNAL_COST} = {_ceil:.4f}"
              f"{'  ⚠️ <1.0 ⇒ 保本不可达' if _ceil < 1.0 - _C5_TOL else '  ✅ ≥1.0'}")
        report.append(f"| （结构性上限） | {'✅' if _ceil >= 1.0 - _C5_TOL else '⚠️'} | "
                      f"donation/SIGNAL_COST = {_ceil:.4f} |")

    # ---- C3：臂间轨迹差异 ----
    print("\n③ C3 臂间轨迹差异断言（同 seed，逐字段一致 ⇒ 失败）：")
    report += ["", "## C3 臂间轨迹差异", "",
               "| seed | 臂对 | 结果 | 首个分歧 tick |", "|---|---|---|---|"]
    degenerate = args.ticks < 2000
    if degenerate:
        # 采样点 = ticks // log_interval(1000)：只 1 个点时"差异断言"近乎无判别力
        print("  ⚠️ 退化守卫：ticks < 2000 ⇒ 采样点 <2，本项判别力不足（建议 ≥3000）")
    for seed in seeds:
        rows = {a: _load_rows(wd / f"{a}_s{seed}.csv") for a in arms}
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                if not rows[a] or not rows[b]:
                    print(f"  ⚠️ s{seed} {a} vs {b}: CSV 缺行，无法比较")
                    continue
                identical, first_diff = compare_trajectories(rows[a], rows[b])
                if identical:
                    failures.append(f"C3 s{seed}: {a} 与 {b} 轨迹逐字段一致（疑配置未生效）")
                print(f"  {'❌' if identical else '✅'} s{seed} {a} vs {b}"
                      f"{' 轨迹完全一致!' if identical else f' 分歧于 tick {first_diff}'}")
                report.append(f"| {seed} | {a} vs {b} | {'❌ 完全一致' if identical else '✅ 有差异'} "
                              f"| {first_diff if first_diff else '—'} |")

    # ---- C4+：世界种子敏感性（R78-3） ----
    print("\n④ R78-3 世界种子敏感性（std≡0 ⇒ 拒判）：")
    report += ["", "## R78-3 世界种子敏感性", "", "| 臂 | 指标 | 各 seed 值 | std | 结果 |",
               "|---|---|---|---|---|"]
    for arm in arms:
        g15_vals: list[float] = []
        n_vals: list[float] = []
        for seed in seeds:
            rows = _load_rows(wd / f"{arm}_s{seed}.csv")
            if rows and rows[-1].get("g15") not in (None, ""):
                g15_vals.append(float(rows[-1]["g15"]))
            res = (_load_summary(wd / f"{arm}_s{seed}.summary.json") or {}).get("result", {})
            if res.get("final_N") is not None:
                n_vals.append(float(res["final_N"]))
        for metric, vals in (("g15_final", g15_vals), ("N_final", n_vals)):
            if len(vals) < 2:
                continue
            ok, sd = seed_sensitivity(vals)
            if not ok:
                failures.append(f"R78-3 {arm}/{metric}: std={sd:.3e}（≡0 ⇒ 拒判）")
            print(f"  {'✅' if ok else '❌'} {arm} {metric}: {[round(v, 4) for v in vals]} std={sd:.4g}")
            report.append(f"| {arm} | {metric} | {[round(v, 4) for v in vals]} | {sd:.4g} "
                          f"| {'✅' if ok else '❌ 拒判'} |")

    # ---- C4'：臂漂移守卫（runner 新增臂而期望表未同步 ⇒ 失败；F-R9 同型防线）----
    runner_src = (ROOT / "experiments" / "batch_runner.py")
    if runner_src.exists():
        txt = runner_src.read_text(encoding="utf-8")
        m = re.search(r"choices\s*=\s*\[([^\]]*)\]", txt)
        if m:
            declared = {x.strip().strip("\"'\"") for x in m.group(1).split(",") if x.strip()}
            missing = sorted(declared - set(EXPECTED_SWITCHES))
            if missing:
                failures.append(f"臂漂移：runner 声明了 {missing}，但 EXPECTED_SWITCHES 未同步")
                print(f"  ❌ 臂漂移守卫：runner 声明 {missing} 未进预期表")
            else:
                print(f"  ✅ 臂漂移守卫：runner 声明 {sorted(declared)} 与预期表一致")

    # ---- 结论 ----
    ok_all = not failures
    print("\n" + "=" * 70)
    print("✅ Pre-Flight 全部通过 ⇒ 可启动实验" if ok_all
          else f"❌ Pre-Flight 未通过（{len(failures)} 项）⇒ **不得启动实验**")
    for f in failures:
        print(f"   - {f}")
    report += ["", "## 结论", "",
               "✅ 全部通过 ⇒ 可启动实验" if ok_all
               else f"❌ 未通过（{len(failures)} 项）⇒ **不得启动实验**", ""]
    for f in failures:
        report.append(f"- {f}")
    (wd / "_preflight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n报告已写：{wd / '_preflight_report.md'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
