"""Pre-Flight 检查工具（R61 纪律 + R78-3 扩展）—— 实验启动前的机械化自查。

为什么需要它
------------
R61 立规「实验前置检查」后，检查单靠人肉执行；本项目已发生多起**静默配置事故**
（F-R9：`--arm control` 未关码本 ⇒ control 与 main 同轨迹整批作废；D2 年龄永不推进；
`radius=6` 静默 no-op）。它们的共同特征是**配置看似生效、实则没生效**，且
**跑完才知道**。本工具把可机械化的检查项固化成代码，跑前 1–3 分钟即可拦截。

覆盖的检查项
------------
| 项 | 内容 | 对应纪律 |
|----|------|---------|
| **C3** | **臂间轨迹差异断言**：同 seed 下不同臂的轨迹若**逐字段完全一致** ⇒ 立即停（F-R9 症状） | R61-C3 |
| **C4** | **开关读回核对**：从 summary.switches 读回真实开关状态，与臂语义逐条核对 | R61-C4 |
| **C4+** | **世界种子敏感性断言**（R78-3）：同模型 ≥3 世界种子，关键指标 std ≡ 0 ⇒ **拒判** | R78-3 |

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

COMPARE_FIELDS = ("N", "g14", "g15", "trust", "max_gen")


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
