#!/usr/bin/env python3
"""
语言涌现分析脚本（language_analysis.py）

从快照文件中读取信号场、解读表和种群数据，计算语言涌现的量化指标。
7 个维度：
  1. 信号词汇统计（频率/熵/Zipf 拟合度）
  2. 文化多样性（解读表标准差，分模式细化）
  3. 代际文化稳定性（相邻世代解读表相关性）
  4. 空间聚类（信号模式的空间自相关 Moran's I）
  5. 信号-语境互信息（信号模式与局部环境的 MI）
  6. 解读一致性（种群内对相同信号的解读收敛度）
  7. 信号基因-表型相关性（g15 与信号发射行为的相关）

用法：
  python3 experiments/language_analysis.py snapshot.npz
  python3 experiments/language_analysis.py snapshot.npz --output lang_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_snapshot(path: str) -> dict:
    """加载快照，只保留存活个体的数据。"""
    d = np.load(path, allow_pickle=True)
    count = int(d["count"])
    return {
        "tick": int(d["tick"]),
        "count": count,
        "max_generation": int(d["max_generation"]),
        "signal_marks": d["signal_marks"],       # (n_cells,) uint8, 低4位=pattern
        "signal_age": d["signal_age"],             # (n_cells,) int32
        "interpret": d["interpret"][:count],       # (P, 16)
        "genes": d["genes"][:count],               # (P, 24)
        "flat": d["flat"][:count],                 # (P,)
        "generation": d["generation"][:count],     # (P,)
        "energy": d["energy"][:count],             # (P,)
        "n_cells": d["signal_marks"].shape[0],
    }


# ─── 维度 1：信号词汇统计 ───────────────────────────────────────────

def signal_vocabulary(signal_marks: np.ndarray, signal_age: np.ndarray) -> dict:
    """16 种信号模式的使用频率、熵、Zipf 拟合度。"""
    # 只统计活跃信号（age >= 0，0 表示无信号）
    active = signal_age >= 0
    patterns = signal_marks[active] & 0x0F  # 低4位
    total = len(patterns)

    if total == 0:
        return {"total_signals": 0, "entropy": 0.0, "zipf_r2": 0.0,
                "freq_by_pattern": [0.0] * 16, "unique_patterns": 0}

    counts = np.bincount(patterns, minlength=16).astype(float)
    freq = counts / total
    # 熵（自然对数）
    nonzero = freq[freq > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = np.log(16)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Zipf 拟合：频率排序后 log(freq) ~ log(rank) 的 R²
    sorted_freq = np.sort(counts)[::-1]
    sorted_freq = sorted_freq[sorted_freq > 0]
    if len(sorted_freq) >= 2:
        ranks = np.arange(1, len(sorted_freq) + 1, dtype=float)
        log_rank = np.log(ranks)
        log_freq = np.log(sorted_freq)
        # 线性回归
        slope, intercept = np.polyfit(log_rank, log_freq, 1)
        predicted = slope * log_rank + intercept
        ss_res = np.sum((log_freq - predicted) ** 2)
        ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
        zipf_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        zipf_slope = float(slope)
    else:
        zipf_r2 = 0.0
        zipf_slope = 0.0

    return {
        "total_signals": int(total),
        "unique_patterns": int(np.sum(counts > 0)),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "zipf_r2": round(zipf_r2, 6),
        "zipf_slope": round(zipf_slope, 4),
        "freq_by_pattern": [round(float(f), 6) for f in freq],
    }


# ─── 维度 2：文化多样性（细化） ─────────────────────────────────────

def cultural_diversity(interpret: np.ndarray) -> dict:
    """解读表的文化多样性，分模式细化 + 总体指标。"""
    if len(interpret) == 0:
        return {"overall_std": 0.0, "per_pattern_std": [0.0] * 16,
                "interpretation_range": 0.0}

    # 每个模式的解读值标准差
    per_pattern_std = interpret.std(axis=0)  # (16,)
    overall_std = float(per_pattern_std.mean())

    # 解读值范围（最大-最小，衡量分化程度）
    interpretation_range = float(interpret.max() - interpret.min())

    # 解读表的协方差矩阵迹（衡量总方差）
    centered = interpret - interpret.mean(axis=0, keepdims=True)
    total_variance = float(np.trace(centered.T @ centered) / max(len(interpret) - 1, 1))

    return {
        "overall_std": round(overall_std, 6),
        "per_pattern_std": [round(float(s), 6) for s in per_pattern_std],
        "interpretation_range": round(interpretation_range, 6),
        "total_variance": round(total_variance, 6),
        "converged_patterns": int(np.sum(per_pattern_std < 0.01)),
    }


# ─── 维度 3：代际文化稳定性 ─────────────────────────────────────────

def generational_stability(interpret: np.ndarray, generation: np.ndarray) -> dict:
    """相邻世代个体解读表的平均相关性，衡量文化传承的稳定性。"""
    if len(interpret) < 4:
        return {"avg_correlation": 0.0, "generation_pairs": 0, "per_gen_corr": {}}

    unique_gens = np.unique(generation)
    if len(unique_gens) < 2:
        return {"avg_correlation": 0.0, "generation_pairs": 0,
                "per_gen_corr": {str(int(g)): 0.0 for g in unique_gens}}

    # 每个世代的平均解读表
    gen_means = {}
    for g in unique_gens:
        mask = generation == g
        if np.sum(mask) >= 2:
            gen_means[int(g)] = interpret[mask].mean(axis=0)

    if len(gen_means) < 2:
        return {"avg_correlation": 0.0, "generation_pairs": 0, "per_gen_corr": {}}

    sorted_gens = sorted(gen_means.keys())
    correlations = []
    per_gen_corr = {}

    for i in range(len(sorted_gens) - 1):
        g1, g2 = sorted_gens[i], sorted_gens[i + 1]
        v1, v2 = gen_means[g1], gen_means[g2]
        # Pearson 相关
        if np.std(v1) > 1e-12 and np.std(v2) > 1e-12:
            corr = float(np.corrcoef(v1, v2)[0, 1])
        else:
            corr = 0.0
        correlations.append(corr)
        per_gen_corr[f"{g1}->{g2}"] = round(corr, 6)

    avg_corr = float(np.mean(correlations)) if correlations else 0.0

    return {
        "avg_correlation": round(avg_corr, 6),
        "generation_pairs": len(correlations),
        "per_gen_corr": per_gen_corr,
        "stable_generations": int(np.sum(np.array(correlations) > 0.8)),
    }


# ─── 维度 4：空间聚类（简化 Moran's I） ─────────────────────────────

def spatial_clustering(signal_marks: np.ndarray, signal_age: np.ndarray,
                        n_cells: int, rows: int = 60, cols: int = 120) -> dict:
    """信号模式的空间自相关：相邻格子是否倾向于有相同信号。"""
    active = (signal_age >= 0) & (signal_marks > 0)
    if np.sum(active) < 4:
        return {"morans_i": 0.0, "same_neighbor_ratio": 0.0, "active_cells": 0}

    patterns = (signal_marks & 0x0F).astype(int)

    # 构建邻居对（球面网格：左右 + 上下，上下行偏移）
    same_count = 0
    total_pairs = 0

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if not active[idx]:
                continue
            # 右邻居
            right = r * cols + ((c + 1) % cols)
            if active[right]:
                total_pairs += 1
                if patterns[idx] == patterns[right]:
                    same_count += 1
            # 下邻居
            below = ((r + 1) % rows) * cols + c
            if active[below]:
                total_pairs += 1
                if patterns[idx] == patterns[below]:
                    same_count += 1

    same_ratio = same_count / total_pairs if total_pairs > 0 else 0.0

    # 简化 Moran's I：(observed_same - expected_same) / (1 - expected_same)
    # expected_same = sum(p_i^2) 随机匹配概率
    active_patterns = patterns[active]
    _, counts = np.unique(active_patterns, return_counts=True)
    freq = counts / counts.sum()
    expected_same = float(np.sum(freq ** 2))
    morans_i = (same_ratio - expected_same) / (1 - expected_same) if (1 - expected_same) > 0 else 0.0

    return {
        "morans_i": round(float(morans_i), 6),
        "same_neighbor_ratio": round(same_ratio, 6),
        "expected_same_random": round(expected_same, 6),
        "active_cells": int(np.sum(active)),
        "neighbor_pairs": total_pairs,
    }


# ─── 维度 5：信号-语境互信息 ─────────────────────────────────────────

def signal_context_mi(signal_marks: np.ndarray, signal_age: np.ndarray,
                       resource_grid: np.ndarray, flat: np.ndarray,
                       n_cells: int) -> dict:
    """信号模式与局部语境（食物丰富度）的互信息。"""
    active = (signal_age >= 0) & (signal_marks > 0)
    if np.sum(active) < 10:
        return {"mutual_information": 0.0, "context_bins": 0}

    patterns = (signal_marks[active] & 0x0F).astype(int)
    food = resource_grid[active]

    # 食物分箱：低/中/高（3 档）
    food_bins = np.clip((food / (food.max() + 1e-9) * 3).astype(int), 0, 2)

    # 联合分布
    n_patterns = 16
    n_context = 3
    joint = np.zeros((n_patterns, n_context))
    for p, c in zip(patterns, food_bins):
        joint[p, c] += 1
    joint /= joint.sum()

    p_pattern = joint.sum(axis=1, keepdims=True)  # (16,1)
    p_context = joint.sum(axis=0, keepdims=True)  # (1,3)

    # 互信息 MI = sum joint * log(joint / (p_pattern * p_context))
    mask = joint > 0
    # 广播 p_pattern (16,1) 和 p_context (1,3) 到 (16,3)
    p_product = p_pattern * p_context  # (16,3)
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / p_product[mask])))

    # 归一化 MI（除以联合熵，0~1）
    joint_entropy = float(-np.sum(joint[mask] * np.log(joint[mask])))
    normalized_mi = mi / joint_entropy if joint_entropy > 0 else 0.0

    return {
        "mutual_information": round(mi, 6),
        "normalized_mi": round(normalized_mi, 6),
        "context_bins": n_context,
        "active_signals": int(len(patterns)),
    }


# ─── 维度 6：解读一致性 ──────────────────────────────────────────────

def interpretation_consistency(interpret: np.ndarray, genes: np.ndarray) -> dict:
    """种群内对相同信号的解读收敛度 + 基因 g14（感知）与解读的关系。"""
    if len(interpret) < 4:
        return {"consistency_score": 0.0, "dominant_interpretation_ratio": 0.0}

    # 对每个模式，计算解读值的"集中度"：1 - std/max_std
    per_pattern_std = interpret.std(axis=0)
    # 初始化解读表是 N(0, 0.3)，所以 max_std ≈ 0.3
    max_std = 0.3
    consistency_per_pattern = 1.0 - np.clip(per_pattern_std / max_std, 0, 1)
    consistency_score = float(consistency_per_pattern.mean())

    # 主导解读比例：对每个模式，有多少比例的个体解读值接近种群均值（±0.1）
    means = interpret.mean(axis=0)
    dominant_ratios = []
    for p in range(16):
        close = np.abs(interpret[:, p] - means[p]) < 0.1
        dominant_ratios.append(float(np.mean(close)))
    avg_dominant_ratio = float(np.mean(dominant_ratios))

    # g14（感知基因，索引14）与解读多样性的相关
    g14 = genes[:, 14] if genes.shape[1] > 14 else np.zeros(len(interpret))
    individual_interpret_std = interpret.std(axis=1)
    if np.std(g14) > 1e-12 and np.std(individual_interpret_std) > 1e-12:
        g14_corr = float(np.corrcoef(g14, individual_interpret_std)[0, 1])
    else:
        g14_corr = 0.0

    return {
        "consistency_score": round(consistency_score, 6),
        "dominant_interpretation_ratio": round(avg_dominant_ratio, 6),
        "per_pattern_consistency": [round(float(c), 6) for c in consistency_per_pattern],
        "g14_interpret_diversity_corr": round(g14_corr, 6),
    }


# ─── 维度 7：信号基因-表型相关性 ─────────────────────────────────────

def signal_genotype_phenotype(genes: np.ndarray, signal_marks: np.ndarray,
                               signal_age: np.ndarray, flat: np.ndarray) -> dict:
    """g15（信号基因）与个体所在格信号密度的相关性。"""
    if len(genes) < 4 or genes.shape[1] < 16:
        return {"g15_mean": 0.0, "g15_signal_corr": 0.0}

    g15 = genes[:, 15]

    # 每个个体所在格是否有活跃信号
    active = (signal_age >= 0) & (signal_marks > 0)
    has_signal = active[flat].astype(float)

    if np.std(g15) > 1e-12 and np.std(has_signal) > 1e-12:
        corr = float(np.corrcoef(g15, has_signal)[0, 1])
    else:
        corr = 0.0

    # g15 高的个体（>0.5）所在格有信号的比例 vs g15 低的个体
    high_g15 = g15 > 0.5
    low_g15 = g15 <= 0.5
    high_signal_rate = float(np.mean(has_signal[high_g15])) if np.sum(high_g15) > 0 else 0.0
    low_signal_rate = float(np.mean(has_signal[low_g15])) if np.sum(low_g15) > 0 else 0.0

    return {
        "g15_mean": round(float(np.mean(g15)), 6),
        "g15_signal_corr": round(corr, 6),
        "high_g15_fraction": round(float(np.mean(high_g15)), 6),
        "high_g15_signal_rate": round(high_signal_rate, 6),
        "low_g15_signal_rate": round(low_signal_rate, 6),
        "signal_rate_diff": round(high_signal_rate - low_signal_rate, 6),
    }


# ─── 综合语言涌现评分 ────────────────────────────────────────────────

def language_emergence_score(metrics: dict) -> dict:
    """
    综合语言涌现评分（0~100），基于 7 个维度的加权。
    评分逻辑：
      - 信号熵适中（不是全用一种，也不是完全随机）→ 词汇丰富度
      - Zipf R² 高 → 词汇有等级结构
      - 文化多样性适中（有差异但不混乱）→ 文化演化空间
      - 代际稳定性高 → 文化可传承
      - 空间聚类高 → 信号有地域方言
      - 信号-语境 MI 高 → 信号有指代意义
      - 解读一致性高 → 种群有共享语义
    """
    vocab = metrics["signal_vocabulary"]
    culture = metrics["cultural_diversity"]
    stability = metrics["generational_stability"]
    spatial = metrics["spatial_clustering"]
    mi = metrics["signal_context_mi"]
    consistency = metrics["interpretation_consistency"]
    genotype = metrics["signal_genotype_phenotype"]

    # 各维度子分（0~1）
    # 1. 词汇丰富度：归一化熵在 0.5~0.9 之间最佳
    ne = vocab["normalized_entropy"]
    vocab_score = max(0, 1 - abs(ne - 0.7) / 0.7)

    # 2. Zipf 结构：R² > 0.8 且 slope 在 -1.5~-0.5 之间
    zipf_ok = vocab["zipf_r2"] > 0.7 and -2.0 < vocab["zipf_slope"] < -0.3
    zipf_score = vocab["zipf_r2"] if zipf_ok else vocab["zipf_r2"] * 0.3

    # 3. 文化多样性：overall_std 在 0.05~0.2 之间最佳（有差异但不混乱）
    cd = culture["overall_std"]
    culture_score = max(0, 1 - abs(cd - 0.1) / 0.1)

    # 4. 代际稳定性：avg_correlation > 0.5
    stability_score = max(0, stability["avg_correlation"])

    # 5. 空间聚类：Moran's I > 0.2
    spatial_score = max(0, min(1, spatial["morans_i"] / 0.5))

    # 6. 信号-语境 MI：normalized_mi > 0.1
    mi_score = max(0, min(1, mi["normalized_mi"] / 0.3))

    # 7. 解读一致性：consistency_score > 0.5
    consistency_score = max(0, consistency["consistency_score"])

    # 加权综合
    weights = {
        "vocabulary": 0.15,
        "zipf_structure": 0.10,
        "cultural_diversity": 0.10,
        "generational_stability": 0.20,
        "spatial_clustering": 0.10,
        "signal_context_mi": 0.20,
        "interpretation_consistency": 0.15,
    }
    total = (
        weights["vocabulary"] * vocab_score +
        weights["zipf_structure"] * zipf_score +
        weights["cultural_diversity"] * culture_score +
        weights["generational_stability"] * stability_score +
        weights["spatial_clustering"] * spatial_score +
        weights["signal_context_mi"] * mi_score +
        weights["interpretation_consistency"] * consistency_score
    )

    return {
        "total_score": round(total * 100, 2),
        "subscores": {
            "vocabulary_richness": round(vocab_score, 4),
            "zipf_structure": round(zipf_score, 4),
            "cultural_diversity": round(culture_score, 4),
            "generational_stability": round(stability_score, 4),
            "spatial_clustering": round(spatial_score, 4),
            "signal_context_mi": round(mi_score, 4),
            "interpretation_consistency": round(consistency_score, 4),
        },
        "weights": weights,
        "interpretation": _interpret_score(total),
    }


def _interpret_score(score: float) -> str:
    if score < 10:
        return "无语言迹象：信号基本是随机或单一的，无共享语义"
    elif score < 25:
        return "初级信号系统：有简单信号使用，但缺乏结构和共享语义"
    elif score < 40:
        return "原语言阶段：信号有一定词汇结构和语境关联，但文化传承不稳定"
    elif score < 60:
        return "语言涌现中：信号有词汇等级、语境指代和文化传承，接近真正语言"
    elif score < 80:
        return "显著语言涌现：信号系统具备语言的多数核心特征"
    else:
        return "成熟语言系统：信号系统高度结构化、可传承、有丰富语义"


# ─── 主函数 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="语言涌现分析")
    parser.add_argument("snapshot", help="快照文件路径 (.npz)")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument("--rows", type=int, default=60, help="网格行数")
    parser.add_argument("--cols", type=int, default=120, help="网格列数")
    args = parser.parse_args()

    if not Path(args.snapshot).exists():
        print(f"错误：快照文件不存在: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    print(f"加载快照: {args.snapshot}")
    snap = load_snapshot(args.snapshot)
    print(f"  tick={snap['tick']}, 存活={snap['count']}, 最大世代={snap['max_generation']}")

    metrics = {}

    print("\n[1/7] 信号词汇统计...")
    metrics["signal_vocabulary"] = signal_vocabulary(snap["signal_marks"], snap["signal_age"])

    print("[2/7] 文化多样性...")
    metrics["cultural_diversity"] = cultural_diversity(snap["interpret"])

    print("[3/7] 代际文化稳定性...")
    metrics["generational_stability"] = generational_stability(snap["interpret"], snap["generation"])

    print("[4/7] 空间聚类...")
    metrics["spatial_clustering"] = spatial_clustering(
        snap["signal_marks"], snap["signal_age"], snap["n_cells"], args.rows, args.cols)

    print("[5/7] 信号-语境互信息...")
    metrics["signal_context_mi"] = signal_context_mi(
        snap["signal_marks"], snap["signal_age"],
        snap.get("resource_grid", np.zeros(snap["n_cells"])),
        snap["flat"], snap["n_cells"])

    print("[6/7] 解读一致性...")
    metrics["interpretation_consistency"] = interpretation_consistency(snap["interpret"], snap["genes"])

    print("[7/7] 信号基因-表型相关性...")
    metrics["signal_genotype_phenotype"] = signal_genotype_phenotype(
        snap["genes"], snap["signal_marks"], snap["signal_age"], snap["flat"])

    print("\n综合语言涌现评分...")
    metrics["language_emergence_score"] = language_emergence_score(metrics)

    # 元信息
    metrics["meta"] = {
        "snapshot": str(args.snapshot),
        "tick": snap["tick"],
        "alive_count": snap["count"],
        "max_generation": snap["max_generation"],
        "n_cells": snap["n_cells"],
        "grid": f"{args.rows}x{args.cols}",
    }

    # 输出
    output_json = json.dumps(metrics, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\n结果已保存: {args.output}")

    # 打印摘要
    score = metrics["language_emergence_score"]
    print("\n" + "=" * 60)
    print("语言涌现分析摘要")
    print("=" * 60)
    print(f"快照: {args.snapshot} (tick={snap['tick']}, N={snap['count']})")
    print(f"活跃信号数: {metrics['signal_vocabulary']['total_signals']}")
    print(f"信号熵(归一化): {metrics['signal_vocabulary']['normalized_entropy']:.4f}")
    print(f"Zipf R²: {metrics['signal_vocabulary']['zipf_r2']:.4f}")
    print(f"文化多样性(std): {metrics['cultural_diversity']['overall_std']:.6f}")
    print(f"代际稳定性(相关): {metrics['generational_stability']['avg_correlation']:.4f}")
    print(f"空间聚类(Moran's I): {metrics['spatial_clustering']['morans_i']:.4f}")
    print(f"信号-语境MI(归一化): {metrics['signal_context_mi']['normalized_mi']:.4f}")
    print(f"解读一致性: {metrics['interpretation_consistency']['consistency_score']:.4f}")
    print(f"g15均值: {metrics['signal_genotype_phenotype']['g15_mean']:.4f}")
    print(f"g15-信号相关: {metrics['signal_genotype_phenotype']['g15_signal_corr']:.4f}")
    print("-" * 60)
    print(f"综合语言涌现评分: {score['total_score']}/100")
    print(f"解读: {score['interpretation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
