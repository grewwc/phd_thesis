import pandas as pd
from clean_utils.normalization import norm_features
from preprocess.get_more_feathres import get_more_features

from .gen_flux_txt import test_kepid
from models.utils import load_model
from os import path
import os
import signal
import sys

# just for signal handler
_same, _total, _all_diff, _all_fp, _all_fn, _wrong_local_view_kepids = \
    None, None, None, None, None, None


def __read_df(df):
    count = 0
    while count < len(df):
        yield df.iloc[count]
        count += 1


def _write_output():
    outfile = path.join(path.dirname(__file__), 'result.txt')

    with open(outfile, 'w') as f:
        f.write(f"precision:  {_same / _total * 100:.3f}%\n")
        f.write('\n' + '~' * 80 + '\n')
        f.write(f"differences: {len(_all_diff)}\n")
        # f.writelines(all_diff)
        _write_list(f, _all_diff)

        f.write('\n' + '~' * 80 + '\n')
        f.write(f'false positives: {len(_all_fp)}\n')
        # f.writelines(all_fp)
        _write_list(f, _all_fp)

        f.write('\n' + '~' * 80 + '\n')
        f.write(f'false negtives: {len(_all_fn)}\n')
        # f.writelines(all_fn)
        _write_list(f, _all_fn)

        f.write('\n' + '~' * 80 + '\n')
        f.write(
            f'cannot generate local views: {len(_wrong_local_view_kepids)}\n')
        # f.writelines(wrong_local_view_kepids)
        _write_list(f, _wrong_local_view_kepids)

        f.write('\n' + '~' * 80 + '\n')
        f.write(f'same / total:  {_same / _total * 100:.3f}%\n')


# def sig_handler(sig, frame):
#     _write_output()
#     os._exit(0)


def _write_list(f, data_list):
    for line in data_list:
        f.write(str(line))
        f.write('\n')


def compare(threashhold=0.5):
    global _same, _total, _all_diff, _all_fp, _all_fn, _wrong_local_view_kepids

    fname = path.join(path.dirname(__file__), 'robo.csv')
    df = pd.read_csv(fname)

    kepids_and_plnt = df[['kepid', 'tce_plnt_num', 'pred_class']]

    m = load_model()

    seen = {}
    _same = 0
    _total = 0
    fp, fn = 0, 0
    count = 1

    # kepid_count = -1
    # prev_kepid = None

    _wrong_local_view_kepids = []
    _all_diff = []
    _all_fp = []
    _all_fn = []

    # signal.signal(signal.SIGINT, sig_handler)

    for (kepid, plnt_num, pred_class) in __read_df(kepids_and_plnt):
        # if prev_kepid != kepid:
        #     kepid_count += 1
        #     prev_kepid = kepid
        try:
            if kepid not in seen:
                res = test_kepid(m, kepid)
                seen[kepid] = res

            prob_of_pc = seen[kepid][plnt_num]
            class_of_pc = '1' if float(prob_of_pc) > threashhold else '0'

            if str(pred_class) == class_of_pc:
                _same += 1
            else:
                print(f"diff: {kepid}-{plnt_num}")
                _all_diff.append(f'{kepid}-{plnt_num}')
                if str(pred_class) == '1':
                    fn += 1
                    _all_fn.append(f'{kepid}-{plnt_num} ({prob_of_pc})')

                if str(pred_class) == '0':
                    fp += 1
                    _all_fp.append(f'{kepid}-{plnt_num} ({prob_of_pc})')

            _total += 1
            print(
                f"{count}/{len(kepids_and_plnt)},  precision: {_same / _total * 100:.3f}%")
            count += 1
        except Exception as e:
            print(e)
            _wrong_local_view_kepids.append(kepid)

    _write_output()
