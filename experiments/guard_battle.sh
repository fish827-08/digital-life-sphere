#!/usr/bin/env bash
# guard_battle.sh — 千代战役 battle_main 系综守卫（幂等）
# 逻辑：
#   1) run_ensemble 主进程存活 → 跳过
#   2) 10 seed (42~51) CSV 最新 tick >= 1000000 且 summary.json 存在 → 完成，跳过
#   3) 否则 nohup 后台拉起续跑（run_long_experiment 自动从快照接续）
# 异常才上报：拉起后短时间进程死亡 / 连续多次仍无法推进
set -u
umask 022

PROJ=/workspace
LOG="$PROJ/.guard_battle.log"
RUN_LOG="$PROJ/experiments/run_ensemble_guard.log"
STATE="$PROJ/.guard_battle_state"
SEEDS="42 43 44 45 46 47 48 49 50 51"
N_TICKS=1000000
FAIL_LIMIT=3

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

# ---- 1) 主进程存活检查 ----
if pgrep -f "[r]un_ensemble\.py" >/dev/null 2>&1; then
    log "guard: run_ensemble 主进程存活，跳过"
    exit 0
fi

# ---- 2) 完成检查 ----
all_done=1
for s in $SEEDS; do
    csv="$PROJ/experiments/long_battle_main_s${s}.csv"
    if [ ! -f "$csv" ]; then
        all_done=0; break
    fi
    last_tick=$(tail -n 1 "$csv" 2>/dev/null | cut -d, -f1 | tr -d '[:space:]')
    if ! [[ "$last_tick" =~ ^[0-9]+$ ]] || [ "$last_tick" -lt "$N_TICKS" ] 2>/dev/null; then
        all_done=0; break
    fi
done
if [ "$all_done" -eq 1 ] && [ -f "$PROJ/results/ensemble/battle_main/summary.json" ]; then
    log "guard: 已完成（全部 seed >= ${N_TICKS} tick 且 summary.json 存在），跳过"
    rm -f "$STATE"
    exit 0
fi

# ---- 3) 连续失败计数 + 拉起续跑 ----
fail_streak=0
last_launch_ts=
pid=
if [ -f "$STATE" ]; then
    . "$STATE" 2>/dev/null || true
fi
if [ -n "${last_launch_ts:-}" ] && [ "$last_launch_ts" != "0" ]; then
    # 上次拉起未完成（本次又走到拉起分支）→ 连续失败 +1
    fail_streak=$(( ${fail_streak:-0} + 1 ))
else
    fail_streak=0
fi

cd "$PROJ" || exit 1
# 固定解释器：系统/镜像缺 numpy，3.12.13 已装 numpy + sim_core 按此版本构建
PYBIN=/root/.pyenv/versions/3.12.13/bin/python3
nohup "$PYBIN" experiments/run_ensemble.py \
    --rows 60 --cols 120 --ticks 1000000 --segment 50000 --tag battle_main \
    --seeds 42 43 44 45 46 47 48 49 50 51 --jobs 4 --rotation-period 150 \
    >> "$RUN_LOG" 2>&1 &
GPID=$!

cat > "$STATE" <<EOF
last_launch_ts=$(date +%s)
fail_streak=$fail_streak
pid=$GPID
EOF
log "guard: 拉起续跑 battle_main (pid=$GPID, fail_streak=$fail_streak)"

if [ "$fail_streak" -ge "$FAIL_LIMIT" ]; then
    log "guard: FATAL 连续 ${fail_streak} 次拉起均未推进"
    echo "guard_battle FATAL: battle_main 连续 ${fail_streak} 次拉起均无法推进，详见 $LOG / $RUN_LOG"
    exit 1
fi

# ---- 4) 短时健康检查（拉起后 60s 内进程死亡 → 立即上报） ----
sleep 60
if ! kill -0 "$GPID" 2>/dev/null; then
    log "guard: FATAL 续跑拉起后短时间进程死亡 (pid=$GPID)"
    echo "guard_battle FATAL: battle_main 续跑拉起后 60s 内进程死亡 (pid=$GPID)，详见 $LOG / $RUN_LOG"
    exit 1
fi
log "guard: 健康检查通过 (pid=$GPID 存活)"
exit 0