//! 捕食（L4）Rust 下沉：高攻击基因个体攻击邻格猎物。
//!
//! 语义与 sphere_engine._step_population 步骤 5.5 逐位等价。
//! 提供两个入口：
//!   - `predation_attack`：自包含（自己构建 CSR），供独立调用/对拍
//!   - `predation_attack_with_csr`：接收预构建 CSR，供 l4_l5 合并调用

/// 捕食核心（接收预构建 CSR）。
#[allow(clippy::too_many_arguments)]
pub fn predation_attack_with_csr(
    energy: &mut [f64],
    stomach: &mut [f64],
    predation_mask: &mut [bool],
    flat: &[i64],
    genes: &[f64],
    attackers: &[i64],
    rand_prey: &[i64],
    rand_success: &[f64],
    neighbors: &[i64],
    n_cells: usize,
    nb_stride: usize,
    gene_count: usize,
    max_energy: f64,
    eat_efficiency: f64,
    cell_indptr: &[usize],
    cell_indices: &[usize],
) {
    let n = flat.len();
    if n == 0 || attackers.is_empty() {
        return;
    }
    let stomach_cap = max_energy / eat_efficiency.max(1e-9);

    for (k, &idx_i64) in attackers.iter().enumerate() {
        let idx = idx_i64 as usize;
        if idx >= n {
            continue;
        }
        if energy[idx] <= 0.1 {
            continue;
        }
        let c = flat[idx] as usize;
        if c >= n_cells {
            continue;
        }

        let mut prey_list: Vec<usize> = Vec::with_capacity(16);
        let nb_base = c * nb_stride;
        for nb_off in 0..nb_stride {
            let nbc = neighbors[nb_base + nb_off];
            if nbc < 0 {
                continue;
            }
            let nbc = nbc as usize;
            if nbc >= n_cells {
                continue;
            }
            let start = cell_indptr[nbc];
            let end = cell_indptr[nbc + 1];
            for pos in start..end {
                let j = cell_indices[pos];
                if j != idx {
                    prey_list.push(j);
                }
            }
        }
        prey_list.sort_unstable();
        prey_list.dedup();

        if prey_list.is_empty() {
            continue;
        }

        let prey_pos = (rand_prey[k].rem_euclid(prey_list.len() as i64)) as usize;
        let prey = prey_list[prey_pos];

        if predation_mask[prey] {
            continue;
        }

        energy[idx] -= 0.1;

        let total = energy[idx] + energy[prey];
        let energy_ratio = energy[idx] / total.max(1e-9);
        let g16 = genes[idx * gene_count + 16];
        let success_rate = (energy_ratio * (0.5 + g16 * 0.5)).clamp(0.1, 0.9);

        if rand_success[k] < success_rate {
            predation_mask[prey] = true;
            energy[idx] += energy[prey] * 0.4;
            stomach[idx] = (stomach[idx] + stomach[prey] * 0.4).min(stomach_cap);
            stomach[prey] = 0.0;
        }
    }
}

/// 捕食（自包含，自己构建 CSR）。
#[allow(clippy::too_many_arguments)]
pub fn predation_attack(
    energy: &mut [f64],
    stomach: &mut [f64],
    predation_mask: &mut [bool],
    flat: &[i64],
    genes: &[f64],
    attackers: &[i64],
    rand_prey: &[i64],
    rand_success: &[f64],
    neighbors: &[i64],
    n_cells: usize,
    nb_stride: usize,
    gene_count: usize,
    max_energy: f64,
    eat_efficiency: f64,
) {
    let n = flat.len();
    if n == 0 || attackers.is_empty() {
        return;
    }

    // 构建 CSR
    let mut cell_counts = vec![0usize; n_cells];
    for i in 0..n {
        let c = flat[i] as usize;
        if c < n_cells {
            cell_counts[c] += 1;
        }
    }
    let mut cell_indptr = vec![0usize; n_cells + 1];
    for c in 0..n_cells {
        cell_indptr[c + 1] = cell_indptr[c] + cell_counts[c];
    }
    let mut cell_indices = vec![0usize; n];
    let mut cell_pos = cell_indptr.clone();
    for i in 0..n {
        let c = flat[i] as usize;
        if c < n_cells {
            cell_indices[cell_pos[c]] = i;
            cell_pos[c] += 1;
        }
    }

    predation_attack_with_csr(
        energy, stomach, predation_mask, flat, genes,
        attackers, rand_prey, rand_success,
        neighbors, n_cells, nb_stride, gene_count,
        max_energy, eat_efficiency,
        &cell_indptr, &cell_indices,
    );
}
