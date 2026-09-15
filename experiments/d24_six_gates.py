"""D-24 六门判读（R53 预注册判据的机械执行；跑前锁定，不得事后改）。

六门（R53，`_share/讨论板.md` 17:41 帖）：
  G-A 🔴 oracle 臂必须检出阳性效应   —— oracle 臂的 ⑤/⑥ 显著高于零模型臂（CI 不覆盖 0）
  G-B    codebook_conv 有判别力     —— 4 机制臂非常数（时序 std>0）且 3 机制臂恒 1.0
  G-C    区制可分层                 —— pred_frac 分 ≥2 层且每层 n≥3
  G-D    ⑤ 可计算且方向可读         —— 非饱和窗 ⑤ 非全 NaN 且方向可报
  G-E    ⑥ 有暴露                   —— ⑥a（暴露率）> 0
  G-F    零模型可分离               —— 主臂 vs 零模型 g15 差异可报 CI（D-7 compare_groups）

裁定语义：**六门全过 ⇒ 允许启动标杆批 E-BM；任一门未过 ⇒ 停下修仪器，
不得靠"多跑 seed"绕过**（R53 原文）。

R85/R90 修订（内评 `_eval/预注册-⑤主指标spearman_rho-v1-20260915.md`，跑前锁定）：
  ⑤ 主统计量 = **Spearman ρ**（原 OLS `slope_g15`）；OLS 降辅助列且**禁单独引用**（R90-a）；
  ⑤ 按 `pred_frac` **分层报**（R90-b）；`zero` 臂 |ρ| ≤ 4/√(n−3) 作**仪器自检**（R90-c）；
  ⑤ 另需**最小效应量** |ρ| ≥ `MIN_EFFECT_RHO`（提案 d-1，待 R94）。
  ⇒ G-D 的"可读"是**在位性门**（R87-1 分级表述），**方向不入门判**。

用法：
    python experiments/d24_six_gates.py [--dir _rerun_logs/d24]

输出：控制台判读表 + `<dir>/_six_gates.json`（含每门的原始量与判据值）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observatory.inference import compare_groups  # noqa: E402  D-7 工具（G-F/G-A）


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

ARMS = ("main", "control", "zero", "sigoff", "oracle")
SEEDS = (42, 43, 44)

# ---------------------------------------------------------------------------
# R85 + R90 a/b/c（内评 `_eval/预注册-⑤主指标spearman_rho-v1-20260915.md`）——
# **跑前锁定，不得事后改**；本段只改"读数方式"，不改机制。
# ---------------------------------------------------------------------------
# 🔴 R85：⑤ 的**主统计量**从 OLS `slope_g15` 换为 **Spearman ρ**。
#   [实测] 依据：D-24 的 `zero_s43` OLS = **+15.19** 而 ρ = **−0.0099**
#   （协变量方差塌缩 ⇒ 斜率爆炸）；全批 8/15 臂 ρ 与 OLS **异号**。
# R90-a：OLS 降为**辅助列**，同报但**禁单独引用于门判定**。
# R90-c：`zero` 臂（冻结 g15）|ρ| 必须落在 ±4/√(n−3) 内 ⇒ 仪器自检（真关联 = 0）。
# 提案 d-1（**待 R94 裁定**）：最小效应量 |ρ| ≥ MIN_EFFECT_RHO，
#   防"大 n 下 ρ=0.02 也统计显著"（[实测] Fisher z 在 D-24 上 p≈0 而 G-A 仍不过）。
MAIN_SEL_METRIC = "spearman_rho"
AUX_SEL_METRIC = "slope_g15"
AUX_METRIC_NOTE = "OLS 辅助列（R90-a）：禁单独引用于门判定"
MIN_EFFECT_RHO = 0.10          # 提案 d-1；阈值只在此一处（待 R94）
ZERO_SELFCHECK_SIGMAS = 4.0    # R90-c：|ρ| ≤ 4/√(n−3)


def _load(dirp: Path) -> dict[str, dict]:
    """读全部 summary + CSV 终局行。键 = f"{arm}_s{seed}"。"""
    out: dict[str, dict] = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_s{seed}"
            sp = dirp / f"{key}.summary.json"
            cp = dirp / f"{key}.csv"
            if not sp.exists():
                out[key] = {"missing": True}
                continue
            d = json.loads(sp.read_text(encoding="utf-8"))
            res, sw = d.get("result", {}), d.get("switches", {})
            row = {
                "missing": False,
                "arm": arm,
                "seed": seed,
                "final_N": res.get("final_N"),
                "final_tick": res.get("final_tick"),
                "eco_gate": res.get("eco_gate_pass"),
                "pred_frac": res.get("final_pred_frac"),
                "regime": res.get("regime"),
                "sel": res.get("selection_gradient") or {},
                "resp": res.get("signal_response") or {},
                "oracle": res.get("oracle") or {},
                "codebook_on": bool(sw.get("arbitrary_codebook")),
            }
            # g15 终值 + codebook_conv 时序（取 CSV；列序与 runner 的 fields 一致）
            row["g15_final"] = None
            cb_series: list[float] = []
            if cp.exists():
                rows = list(csv.DictReader(cp.open(encoding="utf-8")))
                if rows:
                    try:
                        row["g15_final"] = float(rows[-1]["g15"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    cb_series = [
                        float(r["codebook_conv"])
                        for r in rows
                        if r.get("codebook_conv") not in (None, "")
                    ]
            row["codebook_series"] = cb_series
            row["codebook_std"] = float(np.std(cb_series)) if len(cb_series) >= 2 else None
            out[key] = row
    return out


def _grp(data: dict, arm: str, pick) -> list[float]:
    vals = []
    for seed in SEEDS:
        r = data.get(f"{arm}_s{seed}", {})
        if r.get("missing"):
            continue
        v = pick(r)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    return vals


def judge(data: dict, dirp: Path) -> dict:
    gates: dict[str, dict] = {}

    # ---- G-A：oracle vs zero 的 ⑤/⑥（CI 不覆盖 0，方向为正）----
    # 🔴 R85：⑤ 主统计量 = **Spearman ρ**（原为 OLS slope_g15，见顶部常量注释）；
    # R90-a：OLS 同报为辅助列、禁单独引用；提案 d-1：⑤ 另需最小效应量。
    ga: dict = {"detail": {}}
    ok_all = True
    for metric, pick in (
        (f"⑤_{MAIN_SEL_METRIC}(R85主)",
         lambda r: (r["sel"].get("non_sat") or {}).get(MAIN_SEL_METRIC)),
        ("⑥_triple", lambda r: (r["resp"]).get("resp_triple")),
    ):
        a = _grp(data, "oracle", pick)
        b = _grp(data, "zero", pick)
        if len(a) >= 2 and len(b) >= 2:
            ir = compare_groups(np.array(a), np.array(b), random_state=42)
            detail = {
                "oracle": [round(v, 6) for v in a], "zero": [round(v, 6) for v in b],
                "median_diff": round(ir.median_diff, 6),
                "ci_low": round(ir.ci_low, 6), "ci_high": round(ir.ci_high, 6),
                "p_value": round(ir.p_value, 4),
                "ci_excludes_zero": bool(ir.ci_low > 0 or ir.ci_high < 0),
                "direction_positive": bool(ir.median_diff > 0),
            }
            ok = bool(detail["ci_excludes_zero"] and detail["direction_positive"])
            if metric.startswith("⑤"):
                med_o = float(np.median(a))
                detail["effect_size_rho_oracle"] = round(med_o, 6)
                detail["min_effect_rho"] = MIN_EFFECT_RHO
                detail["effect_size_ok"] = bool(abs(med_o) >= MIN_EFFECT_RHO)
                detail["effect_size_note"] = (
                    "提案 d-1（待 R94 裁定）：|ρ|≥0.10 才可作为阳性证据（ρ²≈0.01）"
                )
                ok = ok and detail["effect_size_ok"]
                # R90-a：辅助列同报（且**不得**单独引用于门判定）
                detail[AUX_METRIC_NOTE] = [
                    round(v, 6) for v in _grp(
                        data, "oracle",
                        lambda r: (r["sel"].get("non_sat") or {}).get(AUX_SEL_METRIC))
                ]
            ok_all &= ok
        else:
            detail = {"oracle": a, "zero": b, "error": "样本不足，不可判"}
            ok_all = False
        ga["detail"][metric] = detail
    # R90-c：zero 臂仪器自检（冻结 g15 ⇒ 真关联 = 0；违反 ⇒ ⑤ 实现有 bug）
    z_rows: list[dict] = []
    for seed in SEEDS:
        ns = (data.get(f"zero_s{seed}", {}).get("sel", {}) or {}).get("non_sat") or {}
        rho, n = ns.get(MAIN_SEL_METRIC), ns.get("n")
        if rho is None or not n or n < 6:
            continue
        bound = ZERO_SELFCHECK_SIGMAS / float(np.sqrt(n - 3))
        z_rows.append({"seed": seed, "rho": round(float(rho), 6), "n": int(n),
                       "bound": round(bound, 6),
                       "pass": bool(abs(float(rho)) <= bound)})
    z_ok = bool(z_rows) and all(x["pass"] for x in z_rows)
    ga["detail"]["R90-c_zero仪器自检"] = {
        "通过": z_ok,
        "准则": f"|ρ| ≤ {ZERO_SELFCHECK_SIGMAS:g}/√(n−3)",
        "逐seed": z_rows,
        "注": "zero 臂冻结 g15 ⇒ 真关联=0；违反则说明 ⑤ 口径/键控/排序有 bug",
    }
    ok_all &= z_ok
    ga["pass"] = bool(ok_all)
    gates["G-A"] = ga

    # ---- G-B：codebook_conv 判别力 ----
    cb1_stds = {k: v["codebook_std"] for k, v in data.items()
                if not v.get("missing") and v.get("codebook_on")}
    cb0_all_one = all(
        abs(x - 1.0) < 1e-9
        for k, v in data.items()
        if not v.get("missing") and not v.get("codebook_on")
        for x in v["codebook_series"]
    ) and any(
        v["codebook_series"] for k, v in data.items()
        if not v.get("missing") and not v.get("codebook_on")
    )
    varying = [k for k, s in cb1_stds.items() if s is not None and s > 0]
    gates["G-B"] = {
        "pass": bool(len(varying) >= 2 and cb0_all_one),
        "detail": {"cb1_时序std非零臂": varying, "cb0_恒1.0": bool(cb0_all_one)},
    }

    # ---- G-C：区制可分层（全 15 run，阈值 0.9）----
    pf = [(k, v["pred_frac"]) for k, v in data.items()
          if not v.get("missing") and v.get("pred_frac") is not None]
    hi = [k for k, x in pf if x >= 0.9]
    lo = [k for k, x in pf if x < 0.9]
    gates["G-C"] = {
        "pass": bool(len(hi) >= 3 and len(lo) >= 3),
        "detail": {"捕食主导_n": len(hi), "非捕食_n": len(lo),
                   "警告": "两层不得合并均值（R38③）"},
    }

    # ---- G-D：⑤ 非全 NaN 且方向可读（主臂）----
    # R85：主统计量 = Spearman ρ；OLS 同报为辅助列（R90-a 禁单独引用）。
    rho_d = {s: (data.get(f"main_s{s}", {}).get("sel", {}).get("non_sat") or {}).get(MAIN_SEL_METRIC)
             for s in SEEDS}
    ols_d = {s: (data.get(f"main_s{s}", {}).get("sel", {}).get("non_sat") or {}).get(AUX_SEL_METRIC)
             for s in SEEDS}
    readable = [v for v in rho_d.values() if v is not None]
    gates["G-D"] = {
        "pass": bool(len(readable) >= 2),
        "detail": {"main_非饱和窗ρ(R85主)": rho_d,
                   AUX_METRIC_NOTE: ols_d,
                   "注": "在位性门（R87-1 分级表述）；个体层选择梯度 ≠ 群体均值会涨（V-7 附注）"},
    }

    # ---- G-E：⑥a 暴露率 > 0（主臂）----
    ea = {s: (data.get(f"main_s{s}", {}).get("resp", {}) or {}).get("resp_a_exposure")
          for s in SEEDS}
    gates["G-E"] = {
        "pass": bool(all((v or 0) > 0 for v in ea.values())),
        "detail": {"main_⑥a": ea},
    }

    # ---- G-F：主臂 vs 零模型 g15 差异可报 CI ----
    a = _grp(data, "main", lambda r: r.get("g15_final"))
    b = _grp(data, "zero", lambda r: r.get("g15_final"))
    if len(a) >= 2 and len(b) >= 2:
        ir = compare_groups(np.array(a), np.array(b), random_state=42)
        gates["G-F"] = {
            "pass": True,   # 预注册原文="差异可报 CI"（工具已接入）；显著性如实另报
            "detail": {
                "main_g15终值": [round(v, 4) for v in a],
                "zero_g15终值": [round(v, 4) for v in b],
                "median_diff": round(ir.median_diff, 6),
                "ci_low": round(ir.ci_low, 6), "ci_high": round(ir.ci_high, 6),
                "p_value": round(ir.p_value, 4),
                "ci_excludes_zero": bool(ir.ci_low > 0 or ir.ci_high < 0),
                "注": "预注册判据=『可报 CI』；CI 是否覆盖 0 单列如实报告，不计入门判",
            },
        }
    else:
        gates["G-F"] = {"pass": False, "detail": {"error": "样本不足，无法报 CI"}}

    # ---- 附加（不计门）：R90-b 分层 ρ + oracle 记账与 O-1 辅助 ----
    # R90-b：⑤ 必须**按区制分层报**（两层样本量差 34–46 倍，合并会污染比较）。
    layered: dict[str, dict] = {"捕食主导(≥0.9)": {}, "非捕食(<0.9)": {}}
    for k, v in data.items():
        if v.get("missing"):
            continue
        pf = v.get("pred_frac")
        rho = (v.get("sel", {}).get("non_sat") or {}).get(MAIN_SEL_METRIC)
        if pf is None or rho is None:
            continue
        lay = "捕食主导(≥0.9)" if pf >= 0.9 else "非捕食(<0.9)"
        layered[lay][k] = round(float(rho), 6)
    aux = {
        "R90-b_分层rho": layered,
        "R90-b_注": "两层不得合并均值（R38③）；各层另有各自 n（见 summary.sel.non_sat.n）",
        "oracle_return_ratio": {k: (v.get("oracle") or {}).get("oracle_return_ratio")
                                for k, v in data.items() if k.startswith("oracle")},
        "eco_gate_pass": {k: v.get("eco_gate") for k, v in data.items()
                          if not v.get("missing")},
    }
    return {"gates": gates, "aux": aux,
            "all_pass": all(g["pass"] for g in gates.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_rerun_logs/d24")
    args = ap.parse_args()
    dirp = Path(args.dir)
    if not dirp.is_absolute():
        dirp = ROOT / dirp
    data = _load(dirp)
    missing = [k for k, v in data.items() if v.get("missing")]
    if missing:
        print(f"⚠️ 缺 {len(missing)} 份 summary：{missing}")
    result = judge(data, dirp)

    print("=" * 88)
    print("D-24 六门判读（R53 预注册；六门全过 ⇒ 允许 E-BM；任一门未过 ⇒ 停下修仪器）")
    print("=" * 88)
    names = {"G-A": "oracle 检出阳性(⑤ρ/⑥ CI + 效应量)", "G-B": "codebook_conv 判别力",
             "G-C": "区制可分层(n≥3)", "G-D": "⑤ 可算方向可读(ρ 主/OLS 辅)",
             "G-E": "⑥a 暴露>0", "G-F": "主臂vs零模型可报CI"}
    for g in ("G-A", "G-B", "G-C", "G-D", "G-E", "G-F"):
        mark = "✅" if result["gates"][g]["pass"] else "❌"
        print(f"{mark} {g} {names[g]}")
        for k, v in result["gates"][g]["detail"].items():
            print(f"     {k}: {v}")
    lay = result["aux"].get("R90-b_分层rho") or {}
    if any(lay.values()):
        print("ℹ️ R90-b 分层 ρ（两层不得合并，R38③）：")
        for name, d in lay.items():
            if d:
                print(f"     {name}: {d}")
    o7 = [r for r in result["aux"]["oracle_return_ratio"].values() if r is not None]
    if o7:
        print(f"ℹ️ O-7（不计门）oracle_return_ratio ∈ [{min(o7):.4f}, {max(o7):.4f}]"
              f"（>1 ⇒ 补贴而非补偿；≪1 ⇒ 回馈偏弱，G-A 不过时优先查此参数）")
    print("-" * 88)
    verdict = "✅ 六门全过 ⇒ 允许启动标杆批 E-BM" if result["all_pass"] \
        else "❌ 存在未过之门 ⇒ 停下修仪器，不得加 seed 绕过"
    print(verdict)
    (dirp / "_six_gates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"判读已写：{dirp / '_six_gates.json'}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
