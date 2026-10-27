#!/usr/bin/env bash
# GUI-Lens on the full ScreenSpot-Pro with the API base models of the paper.
# Usage: bash scripts/run_guilens_ssp.sh <deepseek|glm|minimax> [data dir]
set -e
DATA=${2:-data/ScreenSpot-Pro}
case "$1" in
  deepseek) python run_guilens.py --model deepseek-v4.1-flash --config configs/guilens.yaml --thinking low \
              --data-dir "$DATA" --output-dir runs/guilens_deepseek --all --resume ;;
  glm)      python run_guilens.py --model glm-5.3-flash-ark --config configs/guilens_norm.yaml --thinking low \
              --data-dir "$DATA" --output-dir runs/guilens_glm --all --resume ;;
  minimax)  python run_guilens.py --model minimax-m3-go --config configs/guilens_norm.yaml --thinking low \
              --data-dir "$DATA" --output-dir runs/guilens_minimax --all --resume ;;
  *) echo "usage: $0 <deepseek|glm|minimax> [data dir]"; exit 1 ;;
esac
python eval/score.py --data-dir "$DATA" runs/guilens_$1/results.jsonl
