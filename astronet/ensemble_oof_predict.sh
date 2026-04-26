#!/bin/bash
# Generate out-of-fold (OOF) predictions: for each fold k, predict fold k
# with each of the 2 models that were trained without fold k. Output one
# CSV per (fold, seed) into oof_preds/. The meta-model script combines them.

set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

NAME=fa1t
FOLDS_DIR=data/folds
CKPT_ROOT=checkpoints/${NAME}_38_kfold
OUT_DIR=oof_preds
mkdir -p "${OUT_DIR}"

NUM_FOLDS=5
NUM_SEEDS=2

for k in $(seq 0 $((NUM_FOLDS - 1)))
do
    for s in $(seq 1 ${NUM_SEEDS})
    do
        # train.py wraps the checkpoint in a timestamped subdir
        # (AstroCNNModel_<cfg>_<ts>); pick the most recent one.
        parent="${CKPT_ROOT}/fold${k}_seed${s}"
        ckpt=$(ls -1d ${parent}/AstroCNNModel_* 2>/dev/null | sort | tail -n1)
        if [ -z "$ckpt" ]; then
            echo "!! No checkpoint found in ${parent}" >&2
            exit 1
        fi
        out_csv="${OUT_DIR}/fold${k}_seed${s}.csv"
        echo "=== Predict fold=${k} with seed=${s} model ==="
        echo "    ckpt: ${ckpt}"
        echo "    data: ${FOLDS_DIR}/fold_${k}.tfrecord"
        echo "    out:  ${out_csv}"
        python astronet/predict.py \
            --model_dir="${ckpt}" \
            --data_files="${FOLDS_DIR}/fold_${k}.tfrecord" \
            --output_file="${out_csv}"
    done
done
