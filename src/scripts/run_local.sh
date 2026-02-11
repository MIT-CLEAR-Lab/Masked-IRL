#!/usr/bin/env bash
set -euo pipefail

# Unified local training script
# Supports simulation, real robot, language ambiguity, and batch runs

# Defaults
MODE="simulation"  # simulation, realrobot
TRAJ_INFO="obj20_sg10_persg5"
MODEL="maskedrl"
SEED="12345"
NUM_DEMOS=10
NUM_THETAS=34
STATE_DIM=19
LR="0.0001"
BATCH_SIZE=128
TRAIN_ITER=1000
FINETUNE_ITER=0
LANGUAGE_AMBIGUITY=""
LLM_DISAMBIGUATION=""
OMP_NUM_THREADS=3

# Batch mode options
BATCH_MODE=false
SEEDS=()
MODELS=()

usage() {
  cat <<'EOF'
Usage: run_local.sh [options]

Options:
  -m, --mode <simulation|realrobot>    Training mode (default: simulation)
  -t, --traj-info <info>               Trajectory info (default: obj20_sg10_persg5)
  -M, --model <model>                  Model type (default: maskedrl)
  -s, --seed <seed>                    Random seed (default: 12345)
  -d, --demos <num>                    Number of demos (default: 10)
  -n, --thetas <num>                   Number of thetas (default: 34)
  -S, --state-dim <dim>                State dimension (default: 19)
  -l, --lr <lr>                        Learning rate (default: 0.0001)
  -b, --batch-size <size>              Batch size (default: 128)
  -I, --train-iter <iter>              Training iterations (default: 1000)
  -F, --finetune-iter <iter>           Finetuning iterations (default: 0)
  -A, --language-ambiguity <type>      Language ambiguity: omit_referent, omit_expression, paraphrase
  -D, --llm-disambiguation <true|false> Enable LLM disambiguation (default: false)
  -O, --omp-threads <num>              OMP_NUM_THREADS (default: 3)
  --batch-seeds <seed1,seed2,...>      Run with multiple seeds
  --batch-models <model1,model2,...>   Run with multiple models
  -h, --help                           Show this help

Examples:
  # Single run (simulation)
  ./run_local.sh -s 12345 -d 10 -n 34

  # Real robot
  ./run_local.sh -m realrobot -s 12345 -d 10 -n 34

  # Language ambiguity
  ./run_local.sh -A omit_expression -D true -s 12345

  # Batch run with multiple seeds and models
  ./run_local.sh --batch-seeds 12405,24051,40512 --batch-models maskedrl,meirl -d 20 -n 40
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    -t|--traj-info)
      TRAJ_INFO="$2"
      shift 2
      ;;
    -M|--model)
      MODEL="$2"
      shift 2
      ;;
    -s|--seed)
      SEED="$2"
      shift 2
      ;;
    -d|--demos)
      NUM_DEMOS="$2"
      shift 2
      ;;
    -n|--thetas)
      NUM_THETAS="$2"
      shift 2
      ;;
    -S|--state-dim)
      STATE_DIM="$2"
      shift 2
      ;;
    -l|--lr)
      LR="$2"
      shift 2
      ;;
    -b|--batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -I|--train-iter)
      TRAIN_ITER="$2"
      shift 2
      ;;
    -F|--finetune-iter)
      FINETUNE_ITER="$2"
      shift 2
      ;;
    -A|--language-ambiguity)
      LANGUAGE_AMBIGUITY="$2"
      shift 2
      ;;
    -D|--llm-disambiguation)
      LLM_DISAMBIGUATION="$2"
      shift 2
      ;;
    -O|--omp-threads)
      OMP_NUM_THREADS="$2"
      shift 2
      ;;
    --batch-seeds)
      IFS=',' read -ra SEEDS <<< "$2"
      BATCH_MODE=true
      shift 2
      ;;
    --batch-models)
      IFS=',' read -ra MODELS <<< "$2"
      BATCH_MODE=true
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Build command base
build_cmd() {
  local seed=$1
  local model=$2
  local cmd_args=(
    -t "$TRAJ_INFO"
    -m "$model"
    -s "$seed"
    -d "$NUM_DEMOS"
    -n "$NUM_THETAS"
    -S "$STATE_DIM"
    -l "$LR"
    -b "$BATCH_SIZE"
    -I "$TRAIN_ITER"
  )
  
  # Add real robot flag if needed
  if [[ "$MODE" == "realrobot" ]]; then
    cmd_args+=(-r)
  fi
  
  # Add language ambiguity if specified
  if [[ -n "$LANGUAGE_AMBIGUITY" ]]; then
    cmd_args+=(-A "$LANGUAGE_AMBIGUITY")
  fi
  
  # Add LLM disambiguation if specified
  if [[ -n "$LLM_DISAMBIGUATION" ]]; then
    cmd_args+=(-D "$LLM_DISAMBIGUATION")
  fi
  
  echo "${cmd_args[@]}"
}

# Run single experiment
run_experiment() {
  local seed=$1
  local model=$2
  local cmd_args=$(build_cmd "$seed" "$model")
  
  echo "=========================================="
  echo "Running: seed=$seed, model=$model, mode=$MODE"
  echo "Command: OMP_NUM_THREADS=$OMP_NUM_THREADS bash scripts/train.sh ${cmd_args}"
  echo "=========================================="
  
  OMP_NUM_THREADS=$OMP_NUM_THREADS bash scripts/train.sh ${cmd_args}
}

# Main execution
if [[ "$BATCH_MODE" == true ]]; then
  # Batch mode: iterate over seeds and models
  declare -a local_seeds=("${SEEDS[@]}")
  declare -a local_models=("${MODELS[@]}")
  
  # Use defaults if not provided
  if [[ ${#local_seeds[@]} -eq 0 ]]; then
    local_seeds=("$SEED")
  fi
  if [[ ${#local_models[@]} -eq 0 ]]; then
    local_models=("$MODEL")
  fi
  
  for seed in "${local_seeds[@]}"; do
    for model in "${local_models[@]}"; do
      run_experiment "$seed" "$model"
    done
  done
else
  # Single run mode
  run_experiment "$SEED" "$MODEL"
fi

echo "Done"
