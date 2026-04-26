#!/bin/bash
# Distributed variant of ensemble_train_kfold.sh.
# Each machine trains only a subset of the (fold, seed) jobs, then results
# are collected back onto one machine for OOF prediction and meta-training.
#
# Usage:
#   bash astronet/ensemble_train_kfold_distributed.sh <JOB_IDS>
#
# JOB_IDS is a comma-separated list of integers in 0..9, where each id maps
# to one (fold, seed) job in the table below. Run a disjoint subset on each
# PC. No coordination is required because each job writes to its own
# checkpoint directory.
#
#   JOB  FOLD  SEED   OUTPUT DIR
#   0    0     1      checkpoints/fa1t_38_kfold/fold0_seed1
#   1    0     2      checkpoints/fa1t_38_kfold/fold0_seed2
#   2    1     1      checkpoints/fa1t_38_kfold/fold1_seed1
#   3    1     2      checkpoints/fa1t_38_kfold/fold1_seed2
#   4    2     1      checkpoints/fa1t_38_kfold/fold2_seed1
#   5    2     2      checkpoints/fa1t_38_kfold/fold2_seed2
#   6    3     1      checkpoints/fa1t_38_kfold/fold3_seed1
#   7    3     2      checkpoints/fa1t_38_kfold/fold3_seed2
#   8    4     1      checkpoints/fa1t_38_kfold/fold4_seed1
#   9    4     2      checkpoints/fa1t_38_kfold/fold4_seed2
#
# Examples (3 PCs):
#   PC A:  bash astronet/ensemble_train_kfold_distributed.sh 0,1,2,3
#   PC B:  bash astronet/ensemble_train_kfold_distributed.sh 4,5,6
#   PC C:  bash astronet/ensemble_train_kfold_distributed.sh 7,8,9
#
# After all PCs finish, copy each PC's checkpoints/fa1t_38_kfold/fold*_seed*
# directories onto one machine (rsync, zip+transfer, or shared drive), then
# run astronet/ensemble_oof_predict.sh there.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <comma-separated job ids 0..9>" >&2
    echo "Example: $0 0,1,2  (runs jobs 0, 1, and 2 on this machine)" >&2
    exit 1
fi

export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL=AstroCNNModel
CFG=final_alpha_1_tuned
NAME=fa1t
FOLDS_DIR=data/folds
OUT_ROOT=checkpoints/${NAME}_38_kfold
SPLIT_ROOT=data/folds_splits

NUM_FOLDS=5
NUM_SEEDS=2

# Each machine must have data/folds/fold_{0..4}.tfrecord locally before
# running. Build the same per-fold symlink dirs as the single-machine script.
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

IFS=',' read -ra JOB_IDS <<< "$1"

for job in "${JOB_IDS[@]}"
do
    if ! [[ "$job" =~ ^[0-9]+$ ]] || [ "$job" -lt 0 ] || [ "$job" -gt 9 ]; then
        echo "!! Invalid job id: '$job' (must be 0..9)" >&2
        exit 1
    fi
    k=$((job / NUM_SEEDS))
    s=$(((job % NUM_SEEDS) + 1))
    train_glob="${SPLIT_ROOT}/holdout_${k}/train/*"
    eval_glob="${SPLIT_ROOT}/holdout_${k}/eval/*"
    out_dir="${OUT_ROOT}/fold${k}_seed${s}"
    echo "=== [job ${job}] fold=${k} seed=${s} ==="
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

echo
echo "Done. Outputs in ${OUT_ROOT}/. Send these directories back to the"
echo "coordinator machine before running ensemble_oof_predict.sh."
