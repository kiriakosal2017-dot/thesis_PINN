#!/bin/bash
# Reproduce every number in the paper: closed-form baselines, all neural models over five seeds,
# summary tables, few-shot adaptation, blocked cross-validation and uncertainty calibration.
set -e
JOBS=${JOBS:-7}
mkdir -p results_v2/logs
python -W ignore -m experiments.train --baselines
TAGS="pinode pinode_gru pinode_static pinode_relax pinode_fixedkq pinode_raw pinode_nd pinode_noweather data data_gru hybrid pikan"
for s in 0 1 2 3 4; do for t in $TAGS; do echo "$t $s"; done; done \
  | xargs -P "$JOBS" -L 1 sh -c 'python -W ignore -m experiments.train --tag $0 --seed $1 > results_v2/logs/$0_$1.log 2>&1'
python -W ignore -m experiments.summarise
(for s in 0 1 2; do echo "experiments.fewshot --seed $s"; done; for f in 0 1 2 3 4; do echo "experiments.blocked_cv --fold $f"; done) \
  | xargs -P "$JOBS" -L 1 sh -c 'python -W ignore -m $0 $1 $2 > results_v2/logs/$(echo $0_$2 | tr "." "_").log 2>&1'
python -W ignore -m experiments.uncertainty > results_v2/logs/uncertainty.log
