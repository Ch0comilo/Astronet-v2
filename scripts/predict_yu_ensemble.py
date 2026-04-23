"""Yu AstroCNNModel ensemble inference on v2-format TFRecords.

Run in WSL (my_env) from the Yu repo root so the `astronet` package resolves
to Yu's, e.g.:

    cd /mnt/c/Users/danie/Documents/personal/test/astroyul/Astronet-Triage
    PYTHONPATH=. python \\
        /mnt/c/Users/danie/Documents/personal/test/Astronet-Triage/scripts/predict_yu_ensemble.py \\
        --model_root runs/fa1t_yu \\
        --tfrecord_glob '/mnt/c/Users/danie/Documents/personal/test/Astronet-Triage/data/tfrecords/test/*' \\
        --output_file /mnt/c/Users/danie/Documents/personal/test/Astronet-Triage/data/yu_ensemble_test.csv

Writes ONE wide CSV with columns:
    astro_id, disp_E, disp_N, disp_J, disp_S, disp_B, yu_pred_1, ..., yu_pred_N
The notebook only needs to read that file.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet import models
from astronet.util import configdict
from astronet.util import estimator_util


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_root', required=True,
                   help='Dir containing model_1..model_N subdirs')
    p.add_argument('--tfrecord_glob', required=True,
                   help='Glob for v2 test TFRecords')
    p.add_argument('--output_file', required=True,
                   help='Path to the single CSV to write')
    p.add_argument('--nruns', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    astro_ids, gv, lv = [], [], []
    disp_E, disp_N, disp_J, disp_S, disp_B = [], [], [], [], []
    for f in sorted(glob.glob(args.tfrecord_glob)):
        for rec in tf.python_io.tf_record_iterator(f):
            ex = tf.train.Example.FromString(rec)
            feat = ex.features.feature
            astro_ids.append(feat['astro_id'].int64_list.value[0])
            gv.append(list(feat['global_view'].float_list.value))
            lv.append(list(feat['local_view'].float_list.value))
            disp_E.append(feat['disp_E'].int64_list.value[0])
            disp_N.append(feat['disp_N'].int64_list.value[0])
            disp_J.append(feat['disp_J'].int64_list.value[0])
            disp_S.append(feat['disp_S'].int64_list.value[0])
            disp_B.append(feat['disp_B'].int64_list.value[0])

    gv_arr = np.asarray(gv, dtype=np.float32)
    lv_arr = np.asarray(lv, dtype=np.float32)
    print('Loaded {} TCEs'.format(len(astro_ids)))

    df = pd.DataFrame({
        'astro_id': astro_ids,
        'disp_E': disp_E, 'disp_N': disp_N, 'disp_J': disp_J,
        'disp_S': disp_S, 'disp_B': disp_B,
    })

    model_class = models.get_model_class('AstroCNNModel')
    config = configdict.ConfigDict(
        models.get_model_config('AstroCNNModel', 'local_global'))

    def make_input_fn():
        def input_fn():
            ds = tf.data.Dataset.from_tensor_slices(
                {'global_view': gv_arr, 'local_view': lv_arr})
            ds = ds.batch(args.batch_size)
            feats = ds.make_one_shot_iterator().get_next()
            return {'time_series_features': feats}
        return input_fn

    for i in range(1, args.nruns + 1):
        mdir = os.path.join(args.model_root, 'model_{}'.format(i))
        if not os.path.isdir(mdir):
            print('Skipping missing {}'.format(mdir))
            df['yu_pred_{}'.format(i)] = np.nan
            continue
        est = estimator_util.create_estimator(
            model_class, config.hparams, model_dir=mdir)
        preds = [float(np.squeeze(pr)) for pr in est.predict(make_input_fn())]
        df['yu_pred_{}'.format(i)] = preds
        print('Scored model_{}: {} predictions'.format(i, len(preds)))

    df.to_csv(args.output_file, index=False)
    print('Wrote {} ({} rows, {} cols)'.format(
        args.output_file, len(df), len(df.columns)))


if __name__ == '__main__':
    main()
