#!/usr/bin/env bash
set -euo pipefail

# Default script (simulation)
python_script="scripts/eval.py"

# Defaults
reward_config=""
human_config=""
realrobot=""

# Lists
seeds=("12345")
demos=(10)

usage() {
  cat <<'EOF'
Usage: eval.sh [options]

Options:
  -t <traj_info>          (default: obj20_sg10_persg5)
  -m <model>              (default: maskedrl)
  -C <reward_cfg.yaml>   (override reward config path; if omitted, auto-built from -t/-m)
  -H <human_cfg.yaml>     (override human config path)
  -p <python_script.py>   (default: scripts/eval.py)
  -s <seeds>              CSV or range, e.g. "12345,23451" (default: 12345)
  -d <demos>              CSV or range, e.g. "1-9" or "1,3,5" (default: 10)
  -r                      (use real robot mode)
  -h                      (show this help)

Examples:
  ./eval.sh -t obj20_sg10_persg5 -m maskedrl -s 12345 -d 10
  ./eval.sh -r -t obj20_sg10_persg5 -m maskedrl -s 12345 -d 10
EOF
}

# Parse CSV/range specs into an array variable by name
parse_list() {
  local spec="$1"; local __name="$2"
  local arr=()
  IFS=',' read -ra parts <<< "$spec"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
      local a=${part%-*}; local b=${part#*-}
      if (( a <= b )); then
        for i in $(seq "$a" "$b"); do arr+=("$i"); done
      else
        for i in $(seq "$a" -1 "$b"); do arr+=("$i"); done
      fi
    elif [[ -n "$part" ]]; then
      arr+=("$part")
    fi
  done
  eval "$__name"='("${arr[@]}")'
}

traj_info="obj20_sg10_persg5"
model="maskedrl"

while getopts ":t:m:C:H:p:s:d:rh" opt; do
  case "$opt" in
    t) traj_info="$OPTARG" ;;
    m) model="$OPTARG" ;;
    C) reward_config="$OPTARG" ;;
    H) human_config="$OPTARG" ;;
    p) python_script="$OPTARG" ;;
    s) parse_list "$OPTARG" seeds ;;
    d) parse_list "$OPTARG" demos ;;
    r) realrobot="--realrobot" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Build reward config path if not explicitly provided
if [[ -z "$reward_config" ]]; then
  reward_config="../config/reward_learning/${traj_info}/${model}.yaml"
fi

# Set default human config if not provided
if [[ -z "$human_config" ]]; then
  if [[ -n "$realrobot" ]]; then
    human_config="../config/humans/frankarobot_multiple_humans_validfeat1and2.yaml"
  else
    human_config="../config/humans/frankarobot_multiple_humans.yaml"
  fi
fi

# Note: realrobot flag will be passed directly to eval.py

echo "== Config =="
echo "traj_info:   $traj_info"
echo "model:       $model"
echo "reward_cfg:  $reward_config"
echo "human_cfg:   $human_config"
echo "script:      $python_script"
echo "seeds:       ${seeds[*]}"
echo "demos:       ${demos[*]}"
echo "realrobot:   ${realrobot:-false}"

for seed in "${seeds[@]}"; do
  echo ">>> Seed: $seed"
  for dq in "${demos[@]}"; do
    echo "  - demos: $dq"
    cmd=( python3 "$python_script"
          --seed "$seed"
          --config "$reward_config"
          -hc "$human_config"
          -dq "$dq" )
    if [[ -n "$realrobot" ]]; then
      cmd+=( --realrobot )
    fi
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
  done
done
