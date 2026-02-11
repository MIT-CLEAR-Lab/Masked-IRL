#!/usr/bin/env bash
set -euo pipefail

# Default script (simulation)
python_script="scripts/train.py"

# Defaults
reward_config=""
human_config=""
wandb="1"
realrobot=""

# Lists (support CSV and ranges like 1-9)
seeds=("12345")
demos=(1 2 3 4 5 6 7 8 9)
thetas=(50)

usage() {
  cat <<'EOF'
Usage: train.sh [options]

Options:
  -t <traj_info>          (default: obj20_sg10_persg5)
  -m <model>              (default: maskedrl)
  -C <reward_cfg.yaml>   (override reward config path; if omitted, auto-built from -t/-m)
  -H <human_cfg.yaml>     (override human config path)
  -p <python_script.py>   (default: scripts/train.py)
  -s <seeds>              CSV or range, e.g. "12345,23451" or "12345-12347" (default: 12345)
  -d <demos>              CSV or range, e.g. "1-9" or "1,3,5"            (default: 1-9)
  -n <thetas>             CSV, e.g. "10,50,100"                          (default: 50)
  -S <state_dim>          (optional: state dimension to pass to python script)
  -l <lr>                 (optional: learning rate)
  -b <batch_size>         (optional: batch size)
  -I <num_iterations>     (optional: number of iterations)
  -L <llm_state_mask_path> (optional: LLM state mask path)
  -A <language_ambiguity> (optional: language ambiguity: None, omit-referent, omit-expression, paraphrase)
  -D <llm_disambiguation> (optional: llm disambiguation: None, llm, vlm)
  -r                      (use real robot mode)
  -w <0|1|true|false>     enable wandb flag (default: 1)

Examples:
  ./train.sh -t obj20_sg10_persg5 -m maskedrl \
      -s 12345,23451 -d 1-3 -n 50,100 -S 42
  ./train.sh -r -t obj20_sg10_persg5 -m maskedrl -s 12345 -d 5 -n 50
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

state_dim=""
lr=""
batch_size=""
num_iterations=""
llm_state_mask_path=""
language_ambiguity=""
llm_disambiguation=""
traj_info="obj20_sg10_persg5"
model="maskedrl"

while getopts ":t:m:C:H:p:s:d:n:S:l:b:I:L:A:D:rw:h" opt; do
  case "$opt" in
    t) traj_info="$OPTARG" ;;
    m) model="$OPTARG" ;;
    C) reward_config="$OPTARG" ;;
    H) human_config="$OPTARG" ;;
    p) python_script="$OPTARG" ;;
    s) parse_list "$OPTARG" seeds ;;
    d) parse_list "$OPTARG" demos ;;
    n) parse_list "$OPTARG" thetas ;;
    S) state_dim="$OPTARG" ;;
    l) lr="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    I) num_iterations="$OPTARG" ;;
    L) llm_state_mask_path="$OPTARG" ;;
    A) language_ambiguity="$OPTARG" ;;
    D) llm_disambiguation="$OPTARG" ;;
    r) realrobot="--realrobot" ;;
    w) wandb="$OPTARG" ;;
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
    human_config="../config/humans/frankarobot_multiple_humans_simple.yaml"
  fi
fi

# Note: realrobot flag will be passed directly to train.py

echo "== Config =="
echo "traj_info:   $traj_info"
echo "model:       $model"
echo "reward_cfg:  $reward_config"
echo "human_cfg:   $human_config"
echo "script:      $python_script"
echo "seeds:       ${seeds[*]}"
echo "demos:       ${demos[*]}"
echo "thetas:      ${thetas[*]}"
echo "state_dim:   $state_dim"
echo "lr:          $lr"
echo "batch_size:  $batch_size"
echo "num_iterations: $num_iterations"
echo "llm_state_mask_path: $llm_state_mask_path"
echo "language_ambiguity: $language_ambiguity"
echo "llm_disambiguation: $llm_disambiguation"
echo "realrobot:   ${realrobot:-false}"
echo "wandb:       $wandb"

for seed in "${seeds[@]}"; do
  echo ">>> Seed: $seed"
  for dq in "${demos[@]}"; do
    echo "  - demos: $dq"
    for num_thetas in "${thetas[@]}"; do
      echo "    * thetas: $num_thetas"
      cmd=( python3 "$python_script"
            --seed "$seed"
            --config "$reward_config"
            -hc "$human_config"
            -dq "$dq"
            --num_train_thetas "$num_thetas" )
      if [[ -n "$state_dim" ]]; then
        cmd+=( --state_dim "$state_dim" )
      fi
      if [[ -n "$lr" ]]; then
        cmd+=( --lr "$lr" )
      fi
      if [[ -n "$batch_size" ]]; then
        cmd+=( --batch_size "$batch_size" )
      fi
      if [[ -n "$num_iterations" ]]; then
        cmd+=( --num_iterations "$num_iterations" )
      fi
      if [[ -n "$llm_state_mask_path" ]]; then
        cmd+=( --llm_state_mask_path "$llm_state_mask_path" )
      fi
      if [[ -n "$language_ambiguity" ]]; then
        cmd+=( --language_ambiguity "$language_ambiguity" )
      fi
      if [[ -n "$llm_disambiguation" ]]; then
        cmd+=( --llm_disambiguation "$llm_disambiguation" )
      fi
      if [[ -n "$realrobot" ]]; then
        cmd+=( --realrobot )
      fi
      if [[ "$wandb" == "1" || "$wandb" == "true" ]]; then
        cmd+=( --wandb )
      fi
      echo "+ ${cmd[*]}"
      "${cmd[@]}"
    done
  done
done
