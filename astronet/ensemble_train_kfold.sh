#!/bin/bash
# Train 10 base models for stacking: 5 folds x 2 seeds.
# Each model is trained on 4 of the 5 folds; the 5th fold is its held-out
# (eval) fold, used for early stopping and later for OOF predictions.

set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL=AstroCNNModel
CFG=final_alpha_1_tuned
NAME=fa1t
FOLDS_DIR=data/folds
OUT_ROOT=checkpoints/${NAME}_38_kfold
SPLIT_ROOT=data/folds_splits

NUM_FOLDS=5
NUM_SEEDS=2

# Build per-fold symlink directories so train.py's single-glob loader can
# read 4 folds at once (it does not understand comma-separated lists).
mkdir -p "${SPLIT_ROOT}"
for k in $(seq 0 $((NUM_FOLDS - 1)))
do
    train_dir="${SPLIT_ROOT}/holdout_${k}/train"
    eval_dir="${SPLIT_ROOT}/holdout_${k}/eval"
    rm -rf "${train_dir}" "${eval_dir}"
    mkdir -p "${train_dir}" "${eval_dir}"
    for j in $(seq 0 $((NUM_FOLDS - 1)))
    do
        src="$(pwd)/${FOLDS_DIR}/fold_${j}.tfrecord"
        if [ "$j" = "$k" ]; then
            ln -s "${src}" "${eval_dir}/fold_${j}.tfrecord"
        else
            ln -s "${src}" "${train_dir}/fold_${j}.tfrecord"
        fi
    done
done

for k in $(seq 0 $((NUM_FOLDS - 1)))
do
    train_glob="${SPLIT_ROOT}/holdout_${k}/train/*"
    eval_glob="${SPLIT_ROOT}/holdout_${k}/eval/*"
    for s in $(seq 1 ${NUM_SEEDS})
    do
        out_dir="${OUT_ROOT}/fold${k}_seed${s}"
        echo "=== Training fold=${k} seed=${s} ==="
        echo "    train: ${train_glob}"
        echo "    eval:  ${eval_glob}"
        echo "    out:   ${out_dir}"
        python astronet/train.py \
            --model=${MODEL} \
            --config_name=${CFG} \
            --train_files="${train_glob}" \
            --eval_files="${eval_glob}" \
            --train_steps=0 \
            --model_dir="${out_dir}"
    done
done
