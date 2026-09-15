"""R97 ⑤ 配对校准批的**判读**（R105 锁定判据 + ρ v3 算法 + v1.1 §4.3/§4.3.1 必报项）。

判据（R105，**跑前锁定、不得事后改**）
----------------------------------
> **校准成功 ⟺「`ratio` 跨 seed 下分位 ∈ [1.2, 1.5]」∧「合并 ρ 的种子级 95% 单侧 CI 下界 > 0」。**
> CI 用 **t**（σ 未知 ⇒ 保守）；**n = 6**（R44：t 口径 0.858；n=4 仅 0.634 不足）；
> R94 floor 0.10 降为**设计备择 μ**（不再是门槛）；R3 符号检验不并列。

必报（v1.1 §4.3 / §4.3.1，`[内评]` 最小补丁）
------------------------------------------
- `ratio` **逐 seed + 区制（`pred_frac` 分层）** + **跨 seed 下分位**（不得用均值）；
- ⑥a/⑥b **与接收侧净效应并报**；receipts/payments 与**偿付截断率**（`oracle_ledger`）；
- 🔴 **⑥ 期望方向已预注册**：⑥b 期望 **> 0 且与科学臂同号**；若 ⑥b 下降或不显著
  ⇒ **先按"激励结构改变（接收侧为负）"解释，不得直接归因仪器灵敏度**，并按 R53 停下上板请裁。

用法
----
    python experiments/r97_calibration_judge.py [--dir _rerun_logs/r97cal]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observatory.statistics import merge_rho_cluster, t_quantile  # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RATIO_LO, RATIO_HI = 1.2, 1.5      # R105：ratio 下分位须落此区间
NAME_RE = re.compile(r"^(?P<arm>rand_)?m(?P<m>[0-9.]+)_s(?P<seed>\d+)$")


def load(dirp: Path) -> dict[str, dict]:
    """→ {"m1.3_s42": {...}}；`rand_` 前缀 = 条件 7 随机信号自检臂。"""
    out: dict[str, dict] = {}
    for sp in sorted(dirp.glob("m*_s*.summary.json")):
        name = sp.name.replace(".summary.json", "")
        mm = NAME_RE.match(name)
        if not mm:
            continue
        d = json.loads(sp.read_text(encoding="utf-8"))
        res, sw = d.get("result", {}), d.get("switches", {})
        orc = res.get("oracle") or {}
        resp = res.get("signal_response") or {}
        sg = (res.get("selection_gradient") or {}).get("non_sat") or {}
        out[name] = {
            "m": float(mm.group("m")),
            "seed": int(mm.group("seed")),
            "is_rand": bool(mm.group("arm")),
            "ratio": orc.get("oracle_return_ratio"),
            "N": res.get("final_N"),
            "pred_frac": res.get("final_pred_frac"),
            "rho": sg.get("spearman_rho"),
            "resp_a": resp.get("resp_a_exposure"),
            "resp_b": resp.get("resp_b_delta"),
            "resp_tri": resp.get("resp_triple"),
            "rs": orc.get("receiver_side") or {},
            "ledger": orc.get("ledger") or {},
            "flag": bool(sw.get("is_calibration_arm")),
            "sw_m": sw.get("oracle_gain_multiplier"),
        }
    return out


def _by_arm(rows: dict[str, dict], m: float, rand: bool = False) -> list[dict]:
    return sorted([r for r in rows.values() if r["m"] == m and r["is_rand"] == rand],
                  key=lambda r: r["seed"])


def _lower_quantile(vals: list[float]) -> dict:
    """跨 seed 下分位（不得用均值）：n=6 时用 **min** 与 **10% 分位**（同值），并附中位作参考。"""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"min": None, "p10": None, "median": None, "n": 0}
    return {"min": float(a.min()), "p10": float(np.quantile(a, 0.10)),
            "median": float(np.median(a)), "n": int(a.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_rerun_logs/r97cal")
    args = ap.parse_args()
    dirp = Path(args.dir)
    if not dirp.is_absolute():
        dirp = ROOT / dirp
    rows = load(dirp)
    if not rows:
        print(f"无数据：{dirp}（先跑 `batch_runner --preset r97cal` 与 `--preset r97cal_rand`）")
        return 1

    ms = sorted({r["m"] for r in rows.values() if not r["is_rand"]})
    print("=" * 100)
    print("R97 ⑤ 配对校准批复判读（R105 判据：ratio 下分位 ∈ [1.2,1.5] ∧ 合并 ρ 种子级 CI 下界 > 0，t，n=6）")
    print(f"数据源 {dirp.relative_to(ROOT)}：{len(rows)} run；m 档 {ms}；"
          f"随机自检臂 {sum(1 for r in rows.values() if r['is_rand'])} run")
    print("=" * 100)

    # ---- ① ratio 逐 seed + 下分位 + 区制 ----
    print("\n① `ratio` 逐 seed（**不得用均值**；下分位 = min = p10，n=6）")
    hdr = f"{'arm':>10}{'seed':>6}{'ratio':>10}{'N':>7}{'pred_frac':>11}{'区制':>14}"
    print(hdr); print("-" * len(hdr))
    verdict_ratio: dict[str, dict] = {}
    for m in ms:
        rs = _by_arm(rows, m)
        for r in rs:
            pf = r["pred_frac"] or 0.0
            print(f"{'m=%.1f' % m:>10}{r['seed']:>6}{r['ratio']:>10.4f}{str(r['N']):>7}"
                  f"{pf:>11.4f}{('捕食主导' if pf >= 0.9 else '非捕食'):>14}")
        q = _lower_quantile([r["ratio"] for r in rs])
        ok = q["min"] is not None and RATIO_LO <= q["min"] <= RATIO_HI
        verdict_ratio[m] = {"q": q, "ok": ok}
        print(f"  ⇒ m={m}: 下分位(min)={q['min']:.4f}  中位={q['median']:.4f}"
              f"  区间判据 [{'%.1f' % RATIO_LO},{'%.1f' % RATIO_HI}] ⇒ {'✅ 达标' if ok else '❌ 未达'}")

    # ---- ② ⑤ 簇级合并 ρ（R105 判据本体）----
    print("\n② ⑤ 合并 ρ 的**种子级单侧 CI 下界**（Fisher z + t，n=seed 数）")
    hdr2 = (f"{'arm':>10}{'n':>4}{'z̄':>10}{'s_z':>9}{'t_crit':>9}{'CI下界(z)':>12}"
            f"{'CI下界(ρ)':>11}{'判定':>10}")
    print(hdr2); print("-" * len(hdr2))
    verdict_rho: dict[float, dict] = {}
    for m in ms:
        rs = _by_arm(rows, m)
        mr = merge_rho_cluster([r["rho"] for r in rs])
        verdict_rho[m] = mr
        print(f"{'m=%.1f' % m:>10}{mr['n_seeds']:>4}{str(mr['zbar']):>10}{str(mr['s_z']):>9}"
              f"{str(mr['t_crit']):>9}{str(mr['ci_low_z']):>12}{str(mr['ci_low_rho']):>11}"
              f"{('✅ 通过' if mr['pass'] else '❌ 不通过'):>10}")
        print(f"    逐 seed ρ = {mr['rho_by_seed']}")

    # ---- ③ ⑥ + 接收侧（v1.1 §4.3/§4.3.1 必报）----
    print("\n③ ⑥ 与**接收侧净效应并报**（⑥b 期望 >0 且与科学臂同号；下降 ⇒ 先按「激励结构改变」解释）")
    hdr3 = (f"{'arm':>10}{'⑥a':>9}{'⑥b':>9}{'⑥tri':>9}{'落点摄入':>10}{'回付/笔':>9}"
            f"{'净(吃-付)':>11}{'全员基线':>10}{'偿付截断n':>11}{'额度截断n':>11}")
    print(hdr3); print("-" * len(hdr3))
    for m in ms:
        rs = _by_arm(rows, m)
        def _mean(key, sub="rs"):
            vs = [r[sub].get(key) for r in rs if r[sub].get(key) is not None]
            return float(np.mean(vs)) if vs else float("nan")
        print(f"{'m=%.1f' % m:>10}{_mean('resp_a', ''):>9.4f}"
              f"{_mean('resp_b', ''):>9.4f}{_mean('resp_tri', ''):>9.4f}"
              f"{_mean('mean_intake_at_paid_cell'):>10.4f}{_mean('mean_payment_per_event'):>9.4f}"
              f"{_mean('net_eat_minus_pay'):>11.4f}{_mean('mean_intake_all_eaters'):>10.4f}"
              f"{sum(int(r['ledger'].get('payer_trunc_n', 0)) for r in rs):>11d}"
              f"{sum(int(r['ledger'].get('budget_trunc_n', 0)) for r in rs):>11d}")
    base_b = float(np.mean([r["resp_b"] for r in _by_arm(rows, ms[0])
                            if r["resp_b"] is not None])) if ms else float("nan")
    print(f"  （科学臂基线 = m={ms[0]} 的 ⑥b 均值 = {base_b:.4f}；逐臂须**与该基线同号**）")

    # ---- ④ 条件 7：随机信号自检 ----
    rand_rows = [r for r in rows.values() if r["is_rand"]]
    if rand_rows:
        print("\n④ 条件 7 随机信号自检（`signal_mode=random` ⇒ 信号与个体状态无关 = **无信息**）")
        for m in sorted({r["m"] for r in rand_rows}):
            rr = [r for r in rand_rows if r["m"] == m]
            q = _lower_quantile([r["ratio"] for r in rr])
            mr = merge_rho_cluster([r["rho"] for r in rr])
            print(f"  rand m={m}: ratio 下分位={q['min']:.4f}（n={q['n']}）"
                  f"  ρ CI下界(z)={mr['ci_low_z']} ⇒ {'ρ 亦为正' if mr['pass'] else 'ρ 未过'}")
            print(f"    对照 state m={m}: ratio 下分位="
                  f"{verdict_ratio.get(m, {}).get('q', {}).get('min')}"
                  f"  ρ CI下界(z)={verdict_rho.get(m, {}).get('ci_low_z')}")
        print("  ⇒ 若 random 档同样上升 ⇒ **实证「增益不依赖信号内容」**（V-1 C-4 的可执行检验）")

    # ---- ⑤ 判定 ----
    print("\n⑤ 判定（R105）")
    for m in ms:
        ro = verdict_ratio[m]["ok"]
        rp = bool(verdict_rho[m]["pass"])
        if m == 1.0:
            tag = "配对基线（不参与「锁 m」，仅作对照）"
        elif ro and rp:
            tag = "**✅ 校准成功 ⇒ 锁定 m**"
        else:
            tag = "❌ 未达标 ⇒ 可上调 m（≤1.5 内、≤2 轮，每轮预注册预期位移）"
        print(f"  m={m}: ratio 下分位∈[{RATIO_LO},{RATIO_HI}]? {ro}；ρ CI>0? {rp} ⇒ {tag}")
    print("\n⚠️ 判读纪律：本批**全部 run 均登记为校准臂** ⇒ 被 R100 条件 5 拒收于科学判读；"
          "本报告的结论**只用于校准 `m`**，不得表述为科学阳性（v1.1 §七.1/§七.3）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
