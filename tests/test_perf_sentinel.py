"""G4 性能哨兵基线（D5 C3）—— 只记录不优化。

固定两个哨兵场景：
  - 小世界：60×120, N=3240（45%密度）
  - 大世界：200×400, N=12000（15%密度）

记录：速率(tick/s) + cProfile 前10热点 + 计数器(bincount次数/跨语言调用次数)。
产出：results/perf_sentinel/baseline.json

用法：
  pytest tests/test_perf_sentinel.py -v -m perf_sentinel   # 跑哨兵
  （常规 pytest 不跑，因为太慢；用 marker 隔离）
"""
import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "perf_sentinel"

pytestmark = pytest.mark.perf_sentinel


def _make_engine(rows: int, cols: int, max_count: int, use_sim_core: bool = True) -> SphereEngine:
    cfg = SimConfig(seed=42)
    cfg.world.rows = rows
    cfg.world.cols = cols
    cfg.population.initial_count = min(max_count, 500)
    cfg.population.max_count = max_count
    cfg.simulation.use_sim_core = use_sim_core
    cfg.simulation.ticks = 1000
    return SphereEngine(cfg)


def _warmup(e: SphereEngine, ticks: int = 50):
    """预热到稳定种群密度。"""
    e.run(ticks)
    # 如果种群还没到上限，继续跑
    target = e.config.population.max_count
    for _ in range(20):
        if e.alive_count() >= target * 0.9:
            break
        e.run(100)


def _profile_run(e: SphereEngine, ticks: int) -> tuple[float, dict]:
    """跑 ticks tick，返回 (速率, profile数据)。"""
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    e.run(ticks)
    elapsed = time.perf_counter() - t0
    pr.disable()

    rate = ticks / elapsed if elapsed > 0 else 0

    # 提取前10热点
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    hotspots = []
    for line in s.getvalue().splitlines()[4:19]:  # 跳过表头
        parts = line.split()
        if len(parts) >= 6:
            try:
                hotspots.append({
                    "ncalls": parts[0],
                    "tottime": float(parts[1]),
                    "cumtime": float(parts[3]),
                    "function": " ".join(parts[5:])[:80],
                })
            except (ValueError, IndexError):
                pass

    return rate, {"elapsed_s": round(elapsed, 3), "rate_tick_s": round(rate, 1), "hotspots": hotspots}


def _count_bincount_calls(e: SphereEngine) -> int:
    """估算每 tick 的 bincount 调用次数（通过源码静态分析）。"""
    # 静态计数：sphere_engine.py 中 np.bincount 出现次数
    engine_path = PROJECT_ROOT / "simulation" / "sphere_engine.py"
    if engine_path.exists():
        with open(engine_path) as f:
            source = f.read()
        return source.count("np.bincount") + source.count("bincount(")
    return 0


def _count_cross_language_calls(e: SphereEngine) -> dict:
    """跨语言调用次数（sim_core 的方法数，作为代理指标）。"""
    if not e._use_sim_core:
        return {"status": "python_only", "calls_per_tick": 0}
    sim_core = e._sim_core
    methods = [m for m in dir(sim_core) if not m.startswith("_") and callable(getattr(sim_core, m))]
    return {"status": "rust_enabled", "available_methods": len(methods), "method_list": methods[:20]}


@pytest.mark.parametrize("scenario", [
    {"name": "small_world", "rows": 60, "cols": 120, "max_count": 3240, "ticks": 200},
    {"name": "large_world", "rows": 200, "cols": 400, "max_count": 12000, "ticks": 50},
])
def test_perf_sentinel(scenario):
    """性能哨兵：记录基线数据，不做断言（只记录不优化）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 性能哨兵: {scenario['name']} ===")
    print(f"世界: {scenario['rows']}x{scenario['cols']}, 目标N={scenario['max_count']}")

    e = _make_engine(scenario["rows"], scenario["cols"], scenario["max_count"])
    print("预热中...")
    _warmup(e)
    n = e.alive_count()
    print(f"预热完成: N={n}")

    rate, profile = _profile_run(e, scenario["ticks"])
    print(f"速率: {rate:.1f} tick/s")

    baseline = {
        "scenario": scenario["name"],
        "world": {"rows": scenario["rows"], "cols": scenario["cols"]},
        "population": {"target": scenario["max_count"], "actual": n},
        "use_sim_core": e._use_sim_core,
        "rate_tick_s": round(rate, 1),
        "elapsed_s": profile["elapsed_s"],
        "ticks_profiled": scenario["ticks"],
        "bincount_sources": _count_bincount_calls(e),
        "cross_language": _count_cross_language_calls(e),
        "top_hotspots": profile["hotspots"][:10],
        "commit": os.popen(f"git -C {PROJECT_ROOT} rev-parse --short HEAD").read().strip() or "unknown",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = OUT_DIR / f"baseline_{scenario['name']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"基线已保存: {out_path}")

    # 不做断言（只记录不优化），但打印关键指标
    print(f"  速率: {rate:.1f} tick/s")
    print(f"  前3热点:")
    for h in profile["hotspots"][:3]:
        print(f"    {h['cumtime']:.3f}s  {h['function']}")
