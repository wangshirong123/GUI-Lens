#!/usr/bin/env bash
# Training-free baselines on the full ScreenSpot-Pro (same base-model call as the direct click).
# Usage: bash scripts/run_baselines_ssp.sh <deepseek|glm> [data dir]
set -e
export SSP_DIR=${2:-data/ScreenSpot-Pro} DET_DIR=runs/det
METHODS=direct,zoomclick,bami,adazoom,som,vote,uizoomer
[ -d "$DET_DIR" ] || python baselines/detect_elements.py           # detector elements for Set-of-Mark
case "$1" in
  deepseek) FMT=toolcall TOOL_MP=1.81 python baselines/run_baselines.py --methods $METHODS \
              --out runs/baselines_deepseek.jsonl --workers 16 ;;
  glm)      # OpenCode Go returns logprobs, which UI-Zoomer needs
            FMT=pixel FIT_MP=6.0 OC_BASE=https://opencode.ai/zen/go/v1 OC_MODEL=glm-5.3-flash \
              python baselines/run_baselines.py --methods $METHODS --out runs/baselines_glm.jsonl --workers 12 ;;
  *) echo "usage: $0 <deepseek|glm> [data dir]"; exit 1 ;;
esac
python eval/score.py --data-dir "$SSP_DIR" runs/baselines_$1.jsonl
