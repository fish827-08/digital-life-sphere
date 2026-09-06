//! sim_core：digital-life-sphere 的 Rust 加速核（模块三）。
//!
//! 模块职责（对齐 PROJECT-DESCRIPTION.md 3.3）
//! -------------------------------------------
//! Python 引擎把"真正发热的批量数值运算"下沉到这里：
//!   - `regrow`：资源场再生（3.2 已完成，与 ResourceField 逐位等价）；
//!   - `step_vectors`：种群数值管线（3.3 移植中）。
//!
//! 边界约定
//! ---------
//! Rust 只算"数值"，不做"决策"：出生判定、世代/ID 维护、清理尸体都在
//! Python 侧。函数输入输出均为借用数组（与 numpy 零拷贝、就地更新）。
//!
//! 组织方式：`src/regrow.rs` / `src/step_vectors.rs` 是纯 Rust 数值核心
//! （不依赖 pyo3，可独立单测）；本文件只做 pyo3 绑定层（切片 ↔ numpy 数组）。
//! 温度/光照等环境量目前仍由 Python 侧计算后传入。
mod consume;
mod regrow;
mod step_vectors;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// regrow：资源场再生（3.2）。
///
/// 语义与 `world/resource_field.py::ResourceField.regrow` 逐位等价：
/// 就地更新 `grid`（第一参数，numpy 数组），返回 None。
/// `temperature` 由 Python 侧光照温度场计算后传入（模块三暂不搬温度场）。
#[pyfunction]
#[pyo3(name = "regrow")]  // 导出名 regrow；内部函数名避免与 mod regrow 冲突
fn regrow_rs(
    grid: Bound<'_, PyArray1<f64>>,
    capacity: PyReadonlyArray1<'_, f64>,
    temperature: PyReadonlyArray1<'_, f64>,
    regrow_rate: f64,
    temp_sensitivity: f64,
) -> PyResult<()> {
    let mut g = unsafe { grid.as_slice_mut()? };
    let cap = capacity.as_slice()?;
    let temp = temperature.as_slice()?;
    regrow::regrow(&mut g, cap, temp, regrow_rate, temp_sensitivity);
    Ok(())
}

/// 缺失长度校验：把 `name` 数组的期望长度核对到 n，否则抛 ValueError。
fn require_len(name: &str, got: usize, want: usize) -> PyResult<()> {
    if got != want {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name}: 长度 {got} 与个体数 {want} 不一致"
        )));
    }
    Ok(())
}

/// consume_many：资源场批量消耗（3.5）。
///
/// 语义与 `ResourceField.consume_many` 逐位等价：按 `flats` 逐只把
/// `amounts[i]`（想吃的量）从 `grid[flats[i]]` 扣掉（同格均分、绝不欠账），
/// 实吃量写回 `out_taken`；`grid` 就地更新、返回 None。
#[pyfunction]
fn consume_many(
    grid: Bound<'_, PyArray1<f64>>,
    flats: PyReadonlyArray1<'_, i64>,
    amounts: PyReadonlyArray1<'_, f64>,
    out_taken: Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    let n_cells = unsafe { grid.as_array().len() };
    let n = flats.as_array().len();
    require_len("amounts", amounts.as_array().len(), n)?;
    require_len("out_taken", unsafe { out_taken.as_array().len() }, n)?;

    let f = flats.as_slice()?;
    // 越界校验：任何 flat 不在 [0, n_cells) 内都直接报错，绝不进入
    // 索引运算（防 Rust slice 越界 panic / UB）。
    let idx_max = n_cells as i64 - 1;
    for &x in f {
        if x < 0 || x > idx_max {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "flats: 索引 {x} 超出资源格范围 [0, {n_cells})"
            )));
        }
    }

    let mut g = unsafe { grid.as_slice_mut()? };
    let mut out = unsafe { out_taken.as_slice_mut()? };
    consume::consume_many(&mut g, f, amounts.as_slice()?, &mut out);
    Ok(())
}

/// step_vectors_stage1：第 1~3 步（光合/代谢/维持），就地更新 energy、stomach。
///
/// 语义与 `sphere_engine._step_population` 第 1、2、3 步逐位一致（见
/// `src/step_vectors.rs` 说明）。所有环境量（eff_activity / illumination）
/// 由 Python 侧先算好传入；age 只读（i64，+1 由 stage2 完成）。
#[pyfunction]
fn step_vectors_stage1(
    energy: Bound<'_, PyArray1<f64>>,
    stomach: Bound<'_, PyArray1<f64>>,
    genes: PyReadonlyArray2<'_, f64>,
    eff_activity: PyReadonlyArray1<'_, f64>,
    illumination: PyReadonlyArray1<'_, f64>,
    age: PyReadonlyArray1<'_, i64>,
    photo_max: f64,
    base_metabolism: f64,
    eat_efficiency: f64,
    growth_mult: f64,
    senile_mult: f64,
    maturity_fraction: f64,
    senile_fraction: f64,
    homeo_upkeep: f64,
    day_length: f64,
) -> PyResult<()> {
    let n = unsafe { energy.as_array().len() };
    require_len("stomach", unsafe { stomach.as_array().len() }, n)?;
    require_len("eff_activity", eff_activity.as_array().len(), n)?;
    require_len("illumination", illumination.as_array().len(), n)?;
    require_len("age", age.as_array().len(), n)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut s = unsafe { stomach.as_slice_mut()? };
    let (ea, il, ag) = (eff_activity.as_slice()?, illumination.as_slice()?, age.as_slice()?);

    step_vectors::step_vectors_stage1(
        &mut e, &mut s, g_slice, gene_count, ea, il, ag,
        photo_max, base_metabolism, eat_efficiency,
        growth_mult, senile_mult, maturity_fraction, senile_fraction,
        homeo_upkeep, day_length,
    );
    Ok(())
}

/// step_vectors_stage2：第 5~8 步（移动扣费/年龄/死亡/冷却/繁殖候选）。
///
/// 在 Python 完成进食与 RNG 抽签后调用。`moved_raw` 是本次"要不要走"的
/// 抽样（Python 抽），能量门槛在本函数内置判定；四个 out_* 掩码由调用方
/// 预分配（长度 n），供 Python 做出生与死亡分流。
#[pyfunction]
fn step_vectors_stage2(
    energy: Bound<'_, PyArray1<f64>>,
    age: Bound<'_, PyArray1<i64>>,
    cooldown: Bound<'_, PyArray1<f64>>,
    genes: PyReadonlyArray2<'_, f64>,
    moved_raw: PyReadonlyArray1<'_, bool>,
    move_cost_ind: PyReadonlyArray1<'_, f64>,
    out_moved: Bound<'_, PyArray1<bool>>,
    out_starved: Bound<'_, PyArray1<bool>>,
    out_expired: Bound<'_, PyArray1<bool>>,
    out_repro: Bound<'_, PyArray1<bool>>,
    day_length: f64,
    maturity_fraction: f64,
    max_energy: f64,
) -> PyResult<()> {
    let n = unsafe { energy.as_array().len() };
    require_len("age", unsafe { age.as_array().len() }, n)?;
    require_len("cooldown", unsafe { cooldown.as_array().len() }, n)?;
    require_len("moved_raw", moved_raw.as_array().len(), n)?;
    require_len("move_cost_ind", move_cost_ind.as_array().len(), n)?;
    require_len("out_moved", unsafe { out_moved.as_array().len() }, n)?;
    require_len("out_starved", unsafe { out_starved.as_array().len() }, n)?;
    require_len("out_expired", unsafe { out_expired.as_array().len() }, n)?;
    require_len("out_repro", unsafe { out_repro.as_array().len() }, n)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut a = unsafe { age.as_slice_mut()? };
    let mut c = unsafe { cooldown.as_slice_mut()? };
    let (mr, mci) = (moved_raw.as_slice()?, move_cost_ind.as_slice()?);
    let mut om = unsafe { out_moved.as_slice_mut()? };
    let mut os = unsafe { out_starved.as_slice_mut()? };
    let mut oe = unsafe { out_expired.as_slice_mut()? };
    let mut orr = unsafe { out_repro.as_slice_mut()? };

    step_vectors::step_vectors_stage2(
        &mut e, &mut a, &mut c, g_slice, gene_count, mr, mci,
        &mut om, &mut os, &mut oe, &mut orr,
        day_length, maturity_fraction, max_energy,
    );
    Ok(())
}

#[pymodule]
fn sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regrow_rs, m)?)?;
    m.add_function(wrap_pyfunction!(consume_many, m)?)?;
    m.add_function(wrap_pyfunction!(step_vectors_stage1, m)?)?;
    m.add_function(wrap_pyfunction!(step_vectors_stage2, m)?)?;
    Ok(())
}