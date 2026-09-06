//! regrow：资源场再生（模块三 · 3.2 移植）。
//!
//! 纯 Rust 数值核心（不依赖 pyo3 / numpy，方便独立单测）。
//! 语义与 Python 侧 `world/resource_field.py::ResourceField.regrow` 逐位等价：
//!
//! ```text
//! factor = clamp((temperature + 20) / 20, 0, 1) ^ temp_sensitivity
//! growth = regrow_rate × factor
//! grid   = min(capacity, grid + growth)
//! ```
//!
//! 对拍保证：`temp_sensitivity == 1.0` 时跳过 `powf`（默认配置正是 1.0），
//! 与 numpy 的线性因子完全一致，可逐位相等断言；非 1 时用容差断言。

/// 就地更新每格食物存量（grid 被原地改写，返回即更新后的存量）。
///
/// 入参长度约定：`grid.len() == capacity.len() == temperature.len()`。
pub fn regrow(
    grid: &mut [f64],
    capacity: &[f64],
    temperature: &[f64],
    regrow_rate: f64,
    temp_sensitivity: f64,
) {
    debug_assert_eq!(grid.len(), capacity.len());
    debug_assert_eq!(grid.len(), temperature.len());

    if temp_sensitivity == 1.0 {
        // 线性因子：无 pow，与 numpy 逐位一致
        for i in 0..grid.len() {
            let factor = clamp01((temperature[i] + 20.0) / 20.0);
            grid[i] = take_min(grid[i] + regrow_rate * factor, capacity[i]);
        }
    } else {
        for i in 0..grid.len() {
            let factor = clamp01((temperature[i] + 20.0) / 20.0).powf(temp_sensitivity);
            grid[i] = take_min(grid[i] + regrow_rate * factor, capacity[i]);
        }
    }
}

/// clamp(x, 0, 1)，NaN 输入按 0 处理（numpy 的 clip 对 NaN 返回 NaN，
/// 但温度场不产生 NaN，此处防御性归 0 以保持"不会涨过/涨出定义域"）。
#[inline]
fn clamp01(x: f64) -> f64 {
    if !(x > 0.0) {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// 与 np.minimum(cap, x) 等价（相等时取 cap，值相同）。
#[inline]
fn take_min(x: f64, cap: f64) -> f64 {
    if x < cap {
        x
    } else {
        cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(grid: &mut [f64], cap: &[f64], temp: &[f64], rate: f64, sens: f64) -> Vec<f64> {
        regrow(grid, cap, temp, rate, sens);
        grid.to_vec()
    }

    #[test]
    fn respects_capacity() {
        let mut grid = vec![39.9, 40.0, 0.0];
        let cap = vec![40.0, 40.0, 40.0];
        let temp = vec![30.0, 30.0, 30.0];
        let out = run(&mut grid, &cap, &temp, 0.5, 1.0);
        assert_eq!(out, vec![40.0, 40.0, 0.5]); // 满的格不长，空格长 0.5
    }

    #[test]
    fn cold_stops_growth() {
        let mut grid = vec![0.0, 0.0, 0.0];
        let cap = vec![40.0, 40.0, 40.0];
        let temp = vec![30.0, 0.0, -20.0];
        let out = run(&mut grid, &cap, &temp, 0.5, 1.0);
        assert_eq!(out, vec![0.5, 0.5, 0.0]); // ≥0° 满速；≤-20° 停摆
    }

    #[test]
    fn sensitivity_scales_factor() {
        // sens=2：30° → factor 2.5^2=6.25，但 clip 到 1 → 0.5；
        // 4° → (24/20)^2 = 1.44 → clip 1 → 0.5；-4° → (16/20)^2=0.64 → 0.32
        let mut grid = vec![0.0, 0.0, 0.0];
        let cap = vec![40.0, 40.0, 40.0];
        let temp = vec![30.0, 4.0, -4.0];
        let out = run(&mut grid, &cap, &temp, 0.5, 2.0);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
        assert!((out[2] - 0.32).abs() < 1e-12);
    }
}