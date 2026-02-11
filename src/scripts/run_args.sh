#!/usr/bin/env bash
set -euo pipefail

# Defaults (feel free to change)
t_default="obj20_sg10_persg5"
m_default="maskedrl"
g_default=""
s_default="12345"
d_default="1-9"
n_default="50"
w_default="1"

# Parse a few flags (add more if you want)
while getopts t:m:g:s:d:n:w:h flag; do
  case "${flag}" in
    t) t_val=${OPTARG} ;;
    m) m_val=${OPTARG} ;;
    g) g_val=${OPTARG} ;;
    s) s_val=${OPTARG} ;;
    d) d_val=${OPTARG} ;;
    n) n_val=${OPTARG} ;;
    w) w_val=${OPTARG} ;;
    h) echo "Usage: $0 -t traj -m model -s seeds -d demos -n thetas -w 0|1"; exit 0 ;;
    *) echo "Unknown flag: -$flag"; exit 1 ;;
  esac
done
shift $((OPTIND-1))

t_val=${t_val:-$t_default}
m_val=${m_val:-$m_default}
g_val=${g_val:-$g_default}
s_val=${s_val:-$s_default}
d_val=${d_val:-$d_default}
n_val=${n_val:-$n_default}
w_val=${w_val:-$w_default}

echo "Calling train.sh with:"
echo "  -t $t_val -m $m_val -s $s_val -d $d_val -n $n_val -w $w_val"
bash ./scripts/train.sh -t "$t_val" -m "$m_val" -s "$s_val" -d "$d_val" -n "$n_val" -w "$w_val"
