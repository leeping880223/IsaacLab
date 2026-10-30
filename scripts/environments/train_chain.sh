#!/usr/bin/env bash
# 手腕轉正：先無自碰撞從頭訓練，再開自碰撞接著訓練。
# 用法：nohup scripts/environments/train_chain.sh > chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG=logs/rsl_rl/openarm_lift
P=IsaacContrib-Lift-Cube-OpenArm-Bimanual-Official

# 找出某個 run_name 最新的 run 資料夾裡，編號最大的 model_*.pt
latest_model() {
  local dir
  dir=$(ls -dt "$LOG"/*_"$1" | head -1)
  ls "$dir"/model_*.pt | sort -V | tail -1
}

# 最後 50 筆成功率的平均低於門檻就停止整個流程
check_success() {
  local dir
  dir=$(ls -dt "$LOG"/*_"$1" | head -1)
  uv run --no-sync python - "$dir" "$2" <<'EOF'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload()
v = [e.value for e in ea.Scalars("Metrics/success_rate")[-50:]]
s = sum(v) / len(v)
print(f"[chain] {sys.argv[1]} success = {s:.3f} (threshold {sys.argv[2]})")
sys.exit(0 if s >= float(sys.argv[2]) else 1)
EOF
}

# train <task 後綴> <run_name> <迭代次數> [其他參數...]
train() {
  local task=$1 run=$2 iters=$3
  shift 3
  echo "[chain] ===== $task ($run, $iters iters) ====="
  uv run --no-sync isaaclab train --rl_library rsl_rl --task "$P-$task" \
    --num_envs 1024 --viz none --max_iterations "$iters" --run_name "$run" "$@"
}

# 階段 1：手腕轉正，無自碰撞，從頭訓練
train WristAligned wrist_aligned 10000
check_success wrist_aligned 0.9

# 階段 2：開自碰撞，從階段 1 最後的 checkpoint 接著訓練
train WristAligned-SelfCollision wrist_aligned_sc 10000 --checkpoint "$(latest_model wrist_aligned)"
check_success wrist_aligned_sc 0.9

echo "[chain] all stages done"
