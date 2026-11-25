#!/bin/bash
cd ~/weavesight
PY=~/envs/weavesight/bin/python
LOG=~/results/aitex_summary.log
KEY="^(RAW|IMAGE|P99|val_max|mean_plus|MAP SIZE|MASK FRACTION|PIXEL AUROC|AUPRO|WARNING)|fit seconds|image_AUROC|pixel_AUROC|Error|Traceback|assert"
run() {
  echo "##### $1 aitex_$2 $(date +%H:%M)" >> $LOG
  if [ "$1" = dinomaly ]; then
    $PY src/run_dinomaly_split.py --category aitex_$2 --max_steps 1000 2>&1 | grep -E "$KEY" >> $LOG
  else
    $PY src/run_patchcore_split.py --category aitex_$2 2>&1 | grep -E "$KEY" >> $LOG
  fi
  $PY src/calibrate_raw.py --model $1 --category aitex_$2 2>&1 | grep -E "$KEY" >> $LOG
  $PY src/compute_pro.py --model $1 --category aitex_$2 2>&1 | grep -E "$KEY" >> $LOG
}
for f in 00 01 02 03 04 05 06; do run patchcore $f; done
for f in 00 01 02 04 06 05; do run dinomaly $f; done
echo "##### ALL DONE $(date +%H:%M)" >> $LOG
