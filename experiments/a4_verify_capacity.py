"""A4/R19 生态门验证（本地开发）— 修复后代码重跑，4 机制为主 + 3 机制对照。

裁定依据：
- **R19**：R4 批运行于修复提交前 ⇒ 整体降级；生态门须用**修复后**代码重跑
  **5 seed × 60k、`arbitrary_codebook=true`**。
- **V12 口径**：4 机制为主 + 3 机制对照臂；两档同 seed 集、同 tick、同 log 间隔。
- **R22 (F-D15)**：manifest 必须记录 git commit + sim_core 指纹 + 配置指纹 + 全部开关真实状态。

用法：
  python experiments/a4_verify_capacity.py --mode on --seed 42 --ticks 60000 --codebook 1 \
      --out _rerun_logs/a4_fix/asym_on_cb1_s42.csv --snapshot-every 5000

断点续跑（快照）：
  - 每 --snapshot-every tick 写一次 <out>.snapshot.npz + <out>.rngstate.pkl
  - 中断后原命令重跑即自动续跑（读取已存在的快照，从 e._tick+1 继续）
  - --fresh 强制从 tick 0 重来
  ⚠️ D2 的感知噪声走【全局 np.random】（已知缺陷 F-D2），引擎快照不含它；
     故本脚本额外存取 np.random 状态，保证续跑与"不中断连续跑"**逐位一致**。
生态门判定：跑满目标 tick 且 tick>=10000 起 N 全程 >0（= N 稳定 >0 持续 50k）。
进度可见：CSV 每采样点 flush；同目录 <name>.progress.json 每 5000 tick 更新。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


# D-19：provenance 统一走 simulation.provenance（硬校验，不再本地静默 None/"unknown"）
from simulation.provenance import collect as prov_collect, validate as prov_validate


def build(mode: str, codebook: bool, seed: int, ticks: int) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = False          # D2 须走 Python 路径（AGENTS.md）
    c.simulation.history_limit = 100           # 环形缓冲，限内存（不改变语义）
    c.population.initial_count = 200           # R4 manifest 真实口径
    c.population.max_count = 5000
    c.resources.distribution = "uniform"       # R4 manifest 真实口径
    d2 = InfoStructureConfig(enabled=True)
    d2.learning_bottleneck = True
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = codebook           # R19/V12：4 机制=True，3 机制对照=False
    d2.steels_alignment = True
    if mode == "off":                          # 对称对照：全感知/无噪声/argmax
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    return SphereEngine(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("on", "off"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=60000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--codebook", type=int, default=1,
                    help="1=四机制(arbitrary_codebook=True, R19 主) / 0=三机制对照")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=5000,
                    help="每 N tick 写一次快照（0=不写）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已有快照，从 tick 0 重跑")
    args = ap.parse_args()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog = out.with_suffix(".progress.json")
    snap = out.with_suffix(".snapshot.npz")
    rngp = out.with_suffix(".rngstate.pkl")

    # ---- 断点续跑：快照存在且非 --fresh 则从快照恢复 ----
    resumed = False
    if not args.fresh and snap.exists():
        e = SphereEngine.load_snapshot(str(snap))       # 配置由快照自带，指纹天然一致
        if rngp.exists():                               # F-D2：补回全局 np.random 状态
            with open(rngp, "rb") as fh:
                np.random.set_state(pickle.load(fh))
        start_tick = int(e._tick)
        resumed = start_tick > 0
    else:
        e = build(args.mode, bool(args.codebook), args.seed, args.ticks)
        start_tick = 0
    if resumed:
        print(f"  ↻ 从快照续跑：tick {start_tick} → {args.ticks}")

    fields = ["tick", "N", "g14", "g15", "trust", "max_gen", "mean_row", "polar_frac"]
    fh = out.open("a" if resumed else "w", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if not resumed:
        w.writeheader()
    fh.flush()

    last = start_tick
    for t in range(start_tick + 1, args.ticks + 1):
        e.step()
        if t % args.log_interval == 0 or e.extinct:
            P = len(e._id)
            r = (e._flat[:P] // 120) if P else np.zeros(0)
            w.writerow({
                "tick": t, "N": P,
                "g14": round(float(e._genes[:P, 14].mean()), 4) if P else "",
                "g15": round(float(e._genes[:P, 15].mean()), 4) if P else "",
                "trust": round(float(e._trust[:P].mean()), 4) if P else "",
                "max_gen": int(e._max_generation),
                "mean_row": round(float(r.mean()), 3) if P else "",
                "polar_frac": round(float(((r <= 5) | (r >= 54)).mean()), 4) if P else "",
            })
            fh.flush()
            last = t
            if e.extinct:
                break
            if t % 5000 == 0:
                prog.write_text(json.dumps({"seed": args.seed, "mode": args.mode,
                                            "codebook": args.codebook, "tick": t,
                                            "N": P}, ensure_ascii=False), encoding="utf-8")
        # 快照独立节拍（不依赖 log_interval），保证任意时刻中断最多丢 snapshot-every 个 tick
        if args.snapshot_every and t % args.snapshot_every == 0:
            e.save_snapshot(str(snap))
            with open(rngp, "wb") as fh2:
                pickle.dump(np.random.get_state(), fh2)
    fh.close()

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    tail = [r for r in rows if int(r["tick"]) >= 10000]
    reached = last >= args.ticks
    stable50k = bool(tail) and all(int(r["N"]) > 0 for r in tail) and last >= 10000
    dc = {str(k): int(v) for k, v in e.death_cause_totals().items()}
    prov = prov_collect(e.config, rng_draws=int(e.rng_draws))
    prov_validate(prov, require_sim_core=bool(e.config.simulation.use_sim_core))
    summary = {
        "manifest": {
            **prov,
            "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "experiments/a4_verify_capacity.py",
        },
        "switches": {                              # R22：全部开关真实状态
            "mode": args.mode, "seed": args.seed, "ticks_target": args.ticks,
            "resumed_from_snapshot": bool(resumed), "start_tick": start_tick,
            "arbitrary_codebook": bool(args.codebook),
            "perception_radius": int(e.config.info_structure.perception_radius),
            "perception_noise": float(e.config.info_structure.perception_noise),
            "softmax_tau": float(e.config.info_structure.softmax_tau),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "steels_alignment": bool(e.config.info_structure.steels_alignment),
            "reputation_weight": float(e.config.info_structure.reputation_weight),
            "use_sim_core": bool(e.config.simulation.use_sim_core),
            "distribution": e.config.resources.distribution,
            "initial_count": int(e.config.population.initial_count),
            "max_count": int(e.config.population.max_count),
            "neutral_genes": bool(e.config.neutral_genes),
            "signal_disabled": bool(e.config.signal_disabled),
        },
        "result": {
            "final_tick": last, "final_N": len(e._id), "extinct": bool(e.extinct),
            "reached_60k": reached, "stable_N_gt0_last50k": stable50k,
            "eco_gate_pass": bool(reached and stable50k),
            "born_total": int(e.total_born), "died_total": int(e.total_died),
            "deaths_by_cause": dc,
        },
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if prog.exists():
        prog.unlink()
    s = summary["result"]
    print(f"[{args.mode} cb={args.codebook} s{args.seed}] tick={s['final_tick']} "
          f"N={s['final_N']} eco_gate={s['eco_gate_pass']} "
          f"born={s['born_total']} died={s['died_total']}")


if __name__ == "__main__":
    main()
