#!/usr/bin/env bash
# guard_battle.sh — 千代战役守护（幂等）：进程存活则跳过；未完成则从快照续跑
# 用途：沙箱/会话被打断后，由定时任务自动拉起实验，利用 run_long_experiment 的快照接续机制。
# 用法: bash experiments/guard_battle.sh
set -u
cd /workspace || exit 1

LOG=/workspace/.guard_battle.log
PY=/root/.pyenv/versions/3.12.13/bin/python
TICKS=1000000
TAG=battle_main

ts() { date +"%Y-%m-%d %H:%M:%S"; }

# 1) 已有 run_ensemble 主进程在跑 → 不动（防双写同一 CSV）
if pgrep -f "run_ensemble.py" >/dev/null 2>&1; then
    echo "$(ts) run_ensemble 已存活，跳过（无需守护动作）" >> "$LOG"
    exit 0
fi

# 2) 完成判定：10 个 seed 的最新 tick 均 >= TICKS 且汇总存在
DONE=1
for s in 42 43 44 45 46 47 48 49 50 51; do
    f="/workspace/experiments/long_${TAG}_s${s}.csv"
    if [ ! -f "$f" ]; then DONE=0; break; fi
    last=$(tail -1 "$f" 2>/dev/null | cut -d, -f1)
    if [ "${last:-0}" -lt "$TICKS" ]; then DONE=0; break; fi
done
[ -f /workspace/results/ensemble/${TAG}/summary.json ] || DONE=0
if [ "$DONE" = 1 ]; then
    echo "$(ts) 千代战役已完成（全部 seed >= ${TICKS} tick + summary.json），无需续跑" >> "$LOG"
    exit 0
fi

# 3) 未完成 → 从快照接续拉起（nohup 脱离会话，cron 退出不杀）
echo "$(ts) 检测到实验未完成，从快照续跑拉起 run_ensemble" >> "$LOG"
nohup "$PY" /workspace/experiments/run_ensemble.py --rows 60 --cols 120 \
    --ticks "$TICKS" --segment 50000 --tag "$TAG" \
    --seeds 42 43 44 45 46 47 48 49 50 51 --jobs 4 --rotation-period 150 \
    >> /workspace/experiments/run_ensemble_guard.log 2>&1 &
echo $! > /workspace/experiments/guard_battle.pid
echo "$(ts) 已拉起续跑 PID=$(cat /workspace/experiments/guard_battle.pid)" >> "$LOG"
exit 0