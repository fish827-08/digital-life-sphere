"""函数级对拍：sim_core.reproduce_batch vs Python 参考实现（T4 / L7b 繁殖下沉）。

直接驱动 Rust reproduce_batch，与 sphere_engine 内 Python 参考路径逐字段比较
（基因继承/变异、能量/胃粮分割、冷却写入、culture 继承）。RNG 序列：
mut_mask(random) → gene_noise(若有变异) → exp_noise → interp_noise，与引擎内一致。
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.genes import Gene, GENE_COUNT
from simulation.sphere_engine import SphereEngine
import sim_core


def build_case(seed, n=40, k=6, gene_count=GENE_COUNT):
    rng = np.random.default_rng(seed)
    cfg = SimConfig(seed=seed)
    genes = rng.uniform(cfg.genome.gene_min, cfg.genome.gene_max, size=(n, gene_count))
    energy = rng.uniform(0.0, 1.0, size=n)
    stomach = rng.uniform(0.0, 0.5, size=n)
    cooldown = rng.uniform(0.0, 5.0, size=n)
    exp = rng.uniform(0.0, cfg.pleasure.max_reward, size=(n, 120))
    interpret = rng.uniform(0.0, 1.0, size=(n, 16))
    trust = rng.uniform(0.0, 1.0, size=n)
    baseline = rng.uniform(0.0, 1.0, size=n)
    ri = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
    return cfg, genes, energy, stomach, cooldown, exp, interpret, trust, baseline, ri


def python_reference(cfg, genes, energy, stomach, cooldown, exp, interpret,
                     trust, baseline, ri, mut_mask, gene_noise, exp_noise,
                     interp_noise):
    """与 sphere_engine 步骤 8 的 Python 参考路径逐行等价（书签：engine 804~874 行）。"""
    gc = cfg.genome.gene_count
    k = len(ri)
    child_genes = genes[ri].copy()
    sig = cfg.genome.mutation_sigma * (cfg.genome.gene_max - cfg.genome.gene_min)
    child_genes = np.clip(
        child_genes + np.where(mut_mask != 0, gene_noise, 0.0),  # mut 判定已预生成
        cfg.genome.gene_min, cfg.genome.gene_max,
    )
    split = 0.3 + genes[ri, Gene.PARENTAL_INVEST] * 0.4
    child_energy = energy[ri] * split
    child_stomach = stomach[ri] * split
    p_energy = energy.copy()
    p_stomach = stomach.copy()
    p_cooldown = cooldown.copy()
    p_energy[ri] -= child_energy
    p_stomach[ri] -= child_stomach
    p_cooldown[ri] = genes[ri, Gene.REPRO_COOLDOWN] * 60.0
    child_exp = np.clip(exp[ri] + exp_noise, 0.0, cfg.pleasure.max_reward)
    child_interp = interpret[ri] + interp_noise
    child_trust = trust[ri] * 0.8 + 0.5 * 0.2
    child_baseline = baseline[ri] * 0.5
    return {
        "child_genes": child_genes, "child_energy": child_energy,
        "child_stomach": child_stomach, "child_exp": child_exp,
        "child_interp": child_interp, "child_trust": child_trust,
        "child_baseline": child_baseline,
        "p_energy": p_energy, "p_stomach": p_stomach, "p_cooldown": p_cooldown,
    }, sig


def rust_impl(cfg, genes, energy, stomach, cooldown, exp, interpret, trust,
              baseline, ri, mut_mask, gene_noise, exp_noise, interp_noise):
    gc = cfg.genome.gene_count
    k = len(ri)
    out = dict(
        child_genes=np.empty((k, gc)), child_energy=np.empty(k),
        child_stomach=np.empty(k), child_exp=np.empty((k, 120)),
        child_interp=np.empty((k, 16)), child_trust=np.empty(k),
        child_baseline=np.empty(k),
        p_energy=energy.copy(), p_stomach=stomach.copy(),
        p_cooldown=cooldown.copy(),
    )
    sim_core.reproduce_batch(
        ri, genes.copy(), out["p_energy"], out["p_stomach"], out["p_cooldown"],
        exp, interpret, trust, baseline,
        mut_mask, gene_noise, exp_noise, interp_noise,
        out["child_genes"], out["child_energy"], out["child_stomach"],
        out["child_exp"], out["child_interp"], out["child_trust"],
        out["child_baseline"],
        cfg.genome.gene_min, cfg.genome.gene_max, cfg.pleasure.max_reward, 60.0,
    )
    return out


@pytest.mark.parametrize("seed", [11, 23, 99])
def test_reproduce_batch_bitwise(seed):
    cfg, genes, energy, stomach, cooldown, exp, interpret, trust, baseline, ri = \
        build_case(seed)

    rng = np.random.default_rng(seed + 10_000)
    k = len(ri)
    gc = cfg.genome.gene_count
    mut_mask = (rng.random((k, gc)) < cfg.genome.mutation_rate).astype(np.uint8)
    gene_noise = np.zeros((k, gc))
    if mut_mask.any():
        sig = cfg.genome.mutation_sigma * (cfg.genome.gene_max - cfg.genome.gene_min)
        gene_noise = rng.normal(0.0, sig, size=(k, gc))
    exp_noise = rng.normal(0.0, cfg.pleasure.inheritance_noise, size=(k, 120))
    interp_noise = rng.normal(0.0, 0.1, size=(k, 16))

    py_exp, _ = python_reference(
        cfg, genes, energy, stomach, cooldown, exp, interpret,
        trust, baseline, ri, mut_mask, gene_noise, exp_noise, interp_noise,
    )
    rs_out = rust_impl(
        cfg, genes, energy, stomach, cooldown, exp, interpret,
        trust, baseline, ri, mut_mask, gene_noise, exp_noise, interp_noise,
    )
    for key in py_exp:
        np.testing.assert_array_equal(
            rs_out[key], py_exp[key], err_msg=f"{key} 逐位不相等"
        )