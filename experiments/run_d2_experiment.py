#!/usr/bin/env python3
"""D2 信息结构参数调优实验脚本。

支持渐进式开启四大机制，单变量参数扫描，自动保存 manifest + CSV。

用法示例：
  # 阶段1：只开学习瓶颈+Steels对齐
  python experiments/run_d2_experiment.py --ticks 50000 --tag d2_s1 \
      --learning-bottleneck --steels-alignment --seed 42

  # 阶段2：加入任意性码本
  python experiments/run_d2_experiment.py --ticks 50000 --tag d2_s2 \
      --learning-bottleneck --steels-alignment --arbitrary-codebook --seed 42

  # 阶段3：加入信息不对称（全机制）
  python experiments/run_d2_experiment.py --ticks 50000 --tag d2_s3 \
      --learning-bottleneck --steels-alignment --arbitrary-codebook \
      --info-asymmetry --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig, InfoStructureConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D2 信息结构参数调优实验")
    # 世界配置
    p.add_argument("--rows", type=int, default=60)
    p.add_argument("--cols", type=int, default=120)
    p.add_argument("--ticks", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="d2")
    p.add_argument("--log-interval", type=int, default=1000,
                   help="每隔多少tick输出一次统计")
    # D2 四大机制开关
    p.add_argument("--learning-bottleneck", action="store_true",
                   help="开启学习瓶颈（解读表不遗传，幼体从观察学习）")
    p.add_argument("--arbitrary-codebook", action="store_true",
                   help="开启任意性码本（pattern=码本[状态]）")
    p.add_argument("--info-asymmetry", action="store_true",
                   help="开启信息不对称（感知半径4+噪声+softmax）")
    p.add_argument("--steels-alignment", action="store_true",
                   help="开启Steels对齐（同格相遇解读表+码本对齐）")
    # D2 参数：学习瓶颈
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--learning-samples-max", type=int, default=60)
    p.add_argument("--learning-maturity-ticks", type=int, default=1200)
    # D2 参数：任意性码本
    p.add_argument("--codebook-mutation-rate", type=float, default=0.02)
    # D2 参数：信息不对称
    p.add_argument("--perception-radius", type=int, default=4,
                   choices=[4, 6, 8])
    p.add_argument("--perception-noise", type=float, default=0.05)
    p.add_argument("--softmax-tau", type=float, default=0.15)
    # D2 参数：Steels对齐
    p.add_argument("--alignment-rate", type=float, default=0.1)
    p.add_argument("--alignment-step", type=float, default=0.15)
    p.add_argument("--alignment-noise", type=float, default=0.02)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SimConfig:
    cfg = SimConfig()
    cfg.world.rows = args.rows
    cfg.world.cols = args.cols
    cfg.simulation.seed = args.seed
    cfg.simulation.use_sim_core = False  # D2 暂未下沉 Rust

    any_enabled = any([
        args.learning_bottleneck, args.arbitrary_codebook,
        args.info_asymmetry, args.steels_alignment,
    ])
    d2 = InfoStructureConfig(enabled=any_enabled)
    d2.learning_bottleneck = args.learning_bottleneck
    d2.arbitrary_codebook = args.arbitrary_codebook
    d2.steels_alignment = args.steels_alignment
    d2.learning_rate = args.learning_rate
    d2.learning_samples_max = args.learning_samples_max
    d2.learning_maturity_ticks = args.learning_maturity_ticks
    d2.codebook_mutation_rate = args.codebook_mutation_rate
    d2.alignment_rate = args.alignment_rate
    d2.alignment_step = args.alignment_step
    d2.alignment_noise = args.alignment_noise
    if args.info_asymmetry:
        d2.perception_radius = args.perception_radius
        d2.perception_noise = args.perception_noise
        d2.softmax_tau = args.softmax_tau
    else:
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    cfg.info_structure = d2
    return cfg


def codebook_convergence(codebook: np.ndarray) -> float:
    """计算群体码本趋同度（0~1，1=完全一致）。"""
    if len(codebook) == 0:
        return 0.0
    conv = []
    for state in range(16):
        mappings = codebook[:, state].astype(int)
        most_common = int(np.bincount(mappings, minlength=16).max())
        conv.append(most_common / len(mappings))
    return float(np.mean(conv))


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # 保存 manifest
    exp_dir = ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    manifest_path = exp_dir / f"manifest_{args.tag}_s{args.seed}.json"
    manifest = {
        "args": vars(args),
        "config": cfg.to_dict(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"=== D2 实验: tag={args.tag}, seed={args.seed} ===")
    print(f"D2 enabled={cfg.info_structure.enabled}")
    print(f"  learning_bottleneck={cfg.info_structure.learning_bottleneck}")
    print(f"  arbitrary_codebook={cfg.info_structure.arbitrary_codebook}")
    print(f"  info_asymmetry={args.info_asymmetry}")
    print(f"  steels_alignment={cfg.info_structure.steels_alignment}")
    print(f"世界: {args.rows}x{args.cols}, ticks={args.ticks}")
    print()

    engine = SphereEngine(cfg)
    stats = []
    t0 = time.time()

    for tick in range(1, args.ticks + 1):
        engine.step()
        if engine._extinct:
            print(f"*** 种群灭绝 at tick={tick} ***")
            break

        if tick % args.log_interval == 0 or tick == args.ticks:
            P = len(engine._id)
            g15 = float(engine._genes[:, 15].mean()) if P > 0 else 0.0
            g14 = float(engine._genes[:, 14].mean()) if P > 0 else 0.0
            trust = float(engine._trust.mean()) if P > 0 else 0.0
            max_gen = int(engine._generation.max()) if P > 0 else 0
            lc_max = int(engine._learning_count.max()) if P > 0 else 0
            cb_conv = codebook_convergence(engine._codebook) if P > 0 else 0.0
            elapsed = time.time() - t0
            rate = tick / elapsed if elapsed > 0 else 0

            row = {
                "tick": tick, "N": P, "g14": round(g14, 4),
                "g15": round(g15, 4), "trust": round(trust, 4),
                "max_gen": max_gen, "lc_max": lc_max,
                "codebook_conv": round(cb_conv, 4),
            }
            stats.append(row)
            print(
                f"tick={tick:>6} N={P:>4} g15={g15:.3f} g14={g14:.3f} "
                f"trust={trust:.3f} max_gen={max_gen:>3} lc_max={lc_max:>3} "
                f"cb_conv={cb_conv:.3f} rate={rate:.1f}t/s"
            )

    # 保存 CSV
    csv_path = exp_dir / f"long_{args.tag}_s{args.seed}.csv"
    if stats:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)
        print(f"\n结果已保存: {csv_path}")

    # 更新 manifest
    manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    manifest["final_tick"] = tick
    manifest["extinct"] = engine._extinct
    manifest["final_N"] = len(engine._id)
    if stats:
        manifest["final_g15"] = stats[-1]["g15"]
        manifest["final_g14"] = stats[-1]["g14"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"Manifest: {manifest_path}")
    print("=== 实验结束 ===")


if __name__ == "__main__":
    main()
