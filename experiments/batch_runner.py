"""D-25 并行跑批包装（R48 新纪律：单机实验一律走并行包装）。

为什么需要它
------------
R48：并行化是**唯一必要手段**——152 h → 约 4 h，零语义风险，成本极低。
R48 新纪律：**单机实验一律走并行包装**，并发上限**按内存定**（须留 ≥1 GB 余量）。

相比旧 `run_d2x_parallel.ps1` 的改进
------------------------------------
1. **跨平台**（纯 Python / 无 psutil 依赖；Windows 用 GlobalMemoryStatusEx，Linux 读 /proc/meminfo）
2. **并发按内存动态定**，不是写死 20（R48）
3. **内存守卫**：启动前 + 运行中持续检查，低于阈值就不再拉起新 run（只排队，不杀已有进程）
4. **失败重试**：退出码≠0 或 summary 缺失 ⇒ 自动重试 N 次
5. **`--skip-existing`：已完成且 summary 合法的 run 直接跳过** ⇒ 断点续批，
   避免"8 臂已跑完却重跑 10 臂"这类浪费（R19 教训）
6. **汇总**：批结束后读所有 summary → 分层表（R38③：`pred_frac` 分层，不得合并均值）
   → 落 `_summary.csv` / `_summary.md`，并记录墙钟（供 R50 速率校准）

用法
----
    # 预设：R19 验收批（cb0/cb1 × seed 42-46 × 60k）
    python experiments/batch_runner.py --preset r19 --skip-existing

    # 自定义网格
    python experiments/batch_runner.py \\
        --script experiments/a4_verify_capacity.py \\
        --grid codebook=0,1 seed=42,43,44,45,46 \\
        --fixed mode=on ticks=60000 snapshot-every=5000 fresh \\
        --out-template "_rerun_logs/a4_fix/asym_on_cb{codebook}_s{seed}.csv"

    # 只看看会跑什么（不执行）
    python experiments/batch_runner.py --preset r19 --dry-run

退出码：0 = 全部成功；1 = 有 run 失败（已耗尽重试）；2 = 用法/环境错误。
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 内存（零依赖）

if sys.platform == "win32":

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def memory_gb() -> tuple[float, float, int]:
        """返回 (可用GB, 总GB, 负载%)。"""
        m = _MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.ullAvailPhys / 1024**3, m.ullTotalPhys / 1024**3, int(m.dwMemoryLoad))

else:

    def memory_gb() -> tuple[float, float, int]:
        info = {}
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0]) * 1024
        except Exception:
            return (0.0, 0.0, 0)
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        load = int(100 * (total - avail) / total) if total else 0
        return (avail / 1024**3, total / 1024**3, load)


def cpu_count() -> int:
    return os.cpu_count() or 1


def pick_concurrency(per_run_gb: float, reserve_gb: float, cap: int | None) -> int:
    """并发上限 = min(cap, cpu-1, 按内存算)。R48：必须留 ≥ reserve_gb 余量。"""
    avail, _total, _load = memory_gb()
    by_mem = int(max(1, (avail - reserve_gb) // per_run_gb)) if per_run_gb > 0 else 1
    n = min(by_mem, max(1, cpu_count() - 1))   # 留 1 核给系统/汇总
    if cap:
        n = min(n, cap)
    return max(1, n)


# ---------------------------------------------------------------- run 规格

@dataclass
class Run:
    name: str
    cmd: list[str]
    out: Path
    summary: Path
    attempt: int = 0
    status: str = "pending"      # pending | running | done | failed | skipped
    wall: float = 0.0
    note: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def log(self) -> Path:
        return self.out.with_suffix(self.out.suffix + f".attempt{self.attempt}.log")


def summary_ok(p: Path) -> bool:
    """summary 合法 = 存在 + 可解析 + 有 final_N + **跑满目标 tick**。

    ⚠️ 判据必须是"跑满目标 tick"而不是"跑满 60k / 通过生态门"：
    小规格批（R47 ≈15 run）与 D-25 自测都跑不满 60k，若用 eco_gate 判据会把
    它们一律误判为失败并无限重试。**灭绝提前结束（final_tick < 目标）才算失败**。
    """
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return False
    r = d.get("result", {})
    if "final_N" not in r:
        return False
    target = d.get("switches", {}).get("ticks_target")
    ft = r.get("final_tick")
    if target is None or ft is None:
        return bool(r.get("eco_gate_pass"))
    try:
        return int(ft) >= int(target)
    except Exception:
        return False


def expand(grid: list[str], fixed: list[str], template: str, script: str,
           workdir: Path) -> list[Run]:
    """笛卡尔积展开网格 → Run 列表。"""
    g: dict[str, list[str]] = {}
    for item in grid:
        k, _, vs = item.partition("=")
        g[k.strip()] = [v for v in vs.split(",") if v != ""]

    keys = list(g)
    combos: list[dict[str, str]] = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in g[k]]

    fixed_pairs: list[tuple[str, str]] = []
    for item in fixed:
        k, sep, v = item.partition("=")
        fixed_pairs.append((k.strip(), v.strip() if sep else ""))

    runs: list[Run] = []
    for c in combos:
        out_rel = template.format(**c)
        out = workdir / out_rel
        args = [script, "--out", out_rel]
        for k in keys:                       # 网格参数在 out 模板里用到，也传给脚本
            args += [f"--{k}", c[k]]
        for k, v in fixed_pairs:
            args.append(f"--{k}")
            if v:
                args.append(v)
        runs.append(Run(
            name=Path(out_rel).stem,
            cmd=args,
            out=out,
            summary=out.with_suffix(".summary.json"),
        ))
    return runs


PRESETS = {
    # R19 验收批：V12 口径 4 机制主臂(cb1) + 3 机制对照(cb0)，seed 42-46，60k
    "r19": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["codebook=0,1", "seed=42,43,44,45,46"],
        fixed=["mode=on", "ticks=60000", "snapshot-every=5000"],
        template="_rerun_logs/a4_fix/asym_on_cb{codebook}_s{seed}.csv",
    ),
    # D-24 小规格验证批（R47/R53）：5 臂 × 3 seed，max_count=3240（R41，⑤ 不饱和前提），
    # 60k tick，--arm 隐含开 ⑤观测+⑥探针（oracle 臂另开 oracle）
    "d24": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["arm=main,control,zero,sigoff,oracle", "seed=42,43,44"],
        fixed=["mode=on", "ticks=60000", "max-count=3240", "snapshot-every=5000"],
        template="_rerun_logs/d24/{arm}_s{seed}.csv",
    ),
}


# ---------------------------------------------------------------- 调度

def poll(running: list[Run]) -> None:
    for r in list(running):
        rc = r.proc.poll()
        if rc is None:
            continue
        running.remove(r)
        r.wall = time.time() - r._t0
        # ---- F-R10：判据放宽（2026-09-15）----
        # 旧判据 `rc == 0 and summary_ok` 会把"rc=1 假失败"误判为 failed 并触发重试。
        # 假失败成因：a4 收尾删 progress.json 被环境 safe-delete 守卫 fail-closed 拒绝
        # ⇒ 子进程退出码 1，而 **summary 早已先写、数据无损**。
        # 新判据：summary 合法即可判 done。为防"读到上一轮的旧 summary"，
        # 额外要求 summary 的 mtime **不早于本次启动时刻**（`_t0`）。
        _fresh = True
        try:
            _fresh = r.summary.stat().st_mtime >= float(getattr(r, "_t0", 0.0)) - 1.0
        except OSError:
            _fresh = False
        if summary_ok(r.summary) and _fresh:
            r.status = "done"
            if rc == 0:
                print(f"  ✅ {r.name}  done  {r.wall/60:.1f} min")
            else:
                r.note = f"⚠️ rc={rc}（summary 完整且为本轮产出 ⇒ 判 done；F-R10 假失败）"
                print(f"  ✅ {r.name}  done（⚠️ rc={rc}，summary 完整）  {r.wall/60:.1f} min")
        else:
            r.status = "failed"
            _why = "summary缺失/不合法"
            if summary_ok(r.summary) and not _fresh:
                _why = "summary 为旧文件（mtime 早于本次启动）"
            r.note = f"rc={rc} {_why}"
            print(f"  ❌ {r.name}  FAILED ({r.note})  {r.wall/60:.1f} min")


def run_batch(runs: list[Run], conc: int, retries: int, python: str, workdir: Path,
              per_run_gb: float, reserve_gb: float, dry: bool) -> int:
    if dry:
        print(f"[dry-run] 并发上限 {conc}（按内存 {per_run_gb:.2f} GB/run，留 {reserve_gb:.1f} GB）")
        for r in runs:
            print(f"  {'⏭️ skip' if r.status=='skipped' else '▶ run '} {r.name}: "
                  f"{python} {' '.join(r.cmd)}")
        return 0

    pending = [r for r in runs if r.status == "pending"]
    running: list[Run] = []
    t_start = time.time()
    guard_hits = 0

    while pending or running:
        # 内存守卫：不足则不再拉起新 run（只排队，绝不杀已有进程）
        while pending and len(running) < conc:
            avail, total, load = memory_gb()
            need = reserve_gb + per_run_gb
            if avail < need:
                guard_hits += 1
                print(f"  🛡️ 内存守卫：可用 {avail:.2f} GB < 需要 {need:.2f} GB"
                      f"（负载 {load}%）⇒ 暂不拉起新 run（运行中 {len(running)}）")
                break
            r = pending.pop(0)
            r.attempt += 1
            with open(r.log, "w", encoding="utf-8") as lf:
                r.proc = subprocess.Popen([python] + r.cmd, cwd=str(workdir),
                                          stdout=lf, stderr=subprocess.STDOUT)
            r._t0 = time.time()
            r.status = "running"
            running.append(r)
            print(f"  ▶ {r.name} (attempt {r.attempt}, 运行中 {len(running)}/{conc}, "
                  f"可用内存 {avail:.2f}/{total:.2f} GB)")

        poll(running)

        # 重试回收
        for r in runs:
            if r.status == "failed" and r.attempt <= retries:
                r.status = "pending"
                pending.append(r)
                print(f"  🔁 {r.name} 重试 {r.attempt}/{retries}")

        if pending or running:
            time.sleep(2.0)

    total_wall = time.time() - t_start
    done = [r for r in runs if r.status == "done"]
    skipped = [r for r in runs if r.status == "skipped"]
    failed = [r for r in runs if r.status == "failed"]
    print(f"\n批次结束：done={len(done)} skipped={len(skipped)} failed={len(failed)} "
          f"墙钟={total_wall/60:.1f} min（守卫触发 {guard_hits} 次）")
    return 1 if failed else 0


# ---------------------------------------------------------------- 汇总

def summarize(runs: list[Run], outdir: Path) -> None:
    from observatory.statistics import predation_fraction  # D-16 单一口径

    rows = []
    for r in runs:
        if not r.summary.exists():
            continue
        try:
            with open(r.summary, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        res, sw = d.get("result", {}), d.get("switches", {})
        causes = res.get("deaths_by_cause") or {}
        pred = res.get("final_pred_frac")
        if pred is None:                      # 回算（D-16 回算口径）
            pred = round(predation_fraction(causes), 4)
        rows.append({
            "arm": r.name,
            "codebook": sw.get("arbitrary_codebook"),
            "arm_type": sw.get("arm"),
            "seed": sw.get("seed"),
            "final_N": res.get("final_N"),
            "eco_gate": res.get("eco_gate_pass"),
            "pred_frac": pred,
            "regime": ("predation_dominant" if pred is not None and pred >= 0.9
                       else "non_predation"),
            "codebook_conv": res.get("final_codebook_conv"),
            # D-17⑤ / D-18⑥ / D-8 oracle（R53 六门判读的原始量）
            "slope_g15": (res.get("selection_gradient") or {}).get("non_sat", {}).get("slope_g15"),
            "resp_a": (res.get("signal_response") or {}).get("resp_a_exposure"),
            "resp_b": (res.get("signal_response") or {}).get("resp_b_delta"),
            "resp_triple": (res.get("signal_response") or {}).get("resp_triple"),
            "oracle_ratio": (res.get("oracle") or {}).get("oracle_return_ratio"),
            "wall_min": round(r.wall / 60, 1),
        })
    if not rows:
        print("（无 summary 可汇总）")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / "_summary.csv"
    with open(csvp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== 汇总（{len(rows)} run）===")
    print(f"{'arm':<22}{'cb':>4}{'seed':>6}{'final_N':>9}{'gate':>7}{'pred_frac':>11}{'区制':>22}{'wall_min':>10}")
    print("-" * 92)
    for x in sorted(rows, key=lambda z: (str(z["codebook"]), z["seed"] or 0)):
        print(f"{x['arm']:<22}{str(x['codebook']):>4}{str(x['seed']):>6}{str(x['final_N']):>9}"
              f"{str(x['eco_gate']):>7}{x['pred_frac']:>11.4f}{x['regime']:>22}"
              f"{str(x['wall_min']):>10}")
    print("-" * 92)

    hi = [x for x in rows if x["pred_frac"] is not None and x["pred_frac"] >= 0.9]
    lo = [x for x in rows if x["pred_frac"] is not None and x["pred_frac"] < 0.9]
    print(f"分层（阈值 0.9）：捕食主导 {len(hi)} / 非捕食主导 {len(lo)}")
    if hi and lo:
        med = lambda g: sorted(x["final_N"] for x in g if isinstance(x["final_N"], int))[len(g) // 2]
        print(f"  ⚠️ R38③：两层不得合并均值。final_N 中位："
              f"捕食主导 {med(hi)} vs 非捕食主导 {med(lo)}")

    (outdir / "_summary.csv").write_text(csvp.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\n汇总已写：{csvp}")


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="D-25 并行跑批包装（R48）")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--script", default="experiments/a4_verify_capacity.py")
    ap.add_argument("--grid", nargs="*", default=[], help="如 codebook=0,1 seed=42,43")
    ap.add_argument("--fixed", nargs="*", default=[], help="如 mode=on ticks=60000 fresh")
    ap.add_argument("--out-template", default="_rerun_logs/a4_fix/run_{seed}.csv")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--python", default=None, help="默认用主工作区 .venv")
    ap.add_argument("--concurrency", type=int, default=None, help="默认按内存自动")
    ap.add_argument("--per-run-gb", type=float, default=0.15,
                    help="单 run 内存估计（实测 68-78 MB，取 2× 余量）")
    ap.add_argument("--reserve-gb", type=float, default=1.0,
                    help="R48：必须留出的内存余量（GB）")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="已有合法 summary 的 run 直接跳过（断点续批）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    args = ap.parse_args()

    if args.preset:
        p = PRESETS[args.preset]
        args.script = p["script"]
        args.grid = p["grid"]
        args.fixed = p["fixed"]
        args.out_template = p["template"]
    if not args.grid:
        print("错误：需要 --grid 或 --preset")
        return 2

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    py = args.python or str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    runs = expand(args.grid, args.fixed, args.out_template, args.script, workdir)
    for r in runs:                      # 产出目录可能与 workdir 不同（如 workdir=_wt_r19）
        r.out.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        for r in runs:
            if summary_ok(r.summary):
                r.status = "skipped"
                r.note = "已有合法 summary"

    avail, total, load = memory_gb()
    conc = args.concurrency or pick_concurrency(args.per_run_gb, args.reserve_gb, None)
    print(f"任务数 {len(runs)}（跳过 {sum(1 for r in runs if r.status=='skipped')}）"
          f" | 并发上限 {conc}（CPU {cpu_count()} 核）"
          f" | 内存 {avail:.2f}/{total:.2f} GB 可用（负载 {load}%）"
          f" | 守卫阈值 留 {args.reserve_gb:.1f} GB + {args.per_run_gb:.2f} GB/run")

    rc = run_batch(runs, conc, args.retries, py, workdir,
                   args.per_run_gb, args.reserve_gb, args.dry_run)

    if not args.no_summarize and not args.dry_run:
        outdir = runs[0].out.parent
        summarize(runs, outdir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
