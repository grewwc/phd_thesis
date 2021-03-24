from config import *
sys.path.append(root_dir)

import sys
import os
import pandas as pd
from clean_utils.normalization import norm_kepid
from clean_utils.io import get_dr24_columns, get_dr25_columns


def compare(kepid):
    kepid = int(norm_kepid(kepid))
    columns = ['kepid', 'tce_plnt_num', 'tce_period',
               'tce_time0bk', 'tce_duration', 'tce_depth']
    df24 = get_dr24_columns(columns)
    df25 = get_dr25_columns(columns)

    dr24_info = df24[df24['kepid'] == kepid]
    dr25_info = df25[df25['kepid'] == kepid]

    # sort the DataFrame by planet number
    dr24_info = dr24_info.sort_values(by=['tce_plnt_num'])
    dr25_info = dr25_info.sort_values(by=['tce_plnt_num'])

    print(dr24_info[columns])
    print(dr25_info[columns])


def get_not_shown_in_dr24():
    columns = ['kepid', 'tce_plnt_num']
    df24 = get_dr24_columns(columns)
    df25 = get_dr25_columns(columns)

    def func(row): return f"{row['kepid']}-{row['tce_plnt_num']}"
    
    dr24_id_plnt = set(df24.apply(func, axis=1))
    dr25_id_plnt = set(df25.apply(func, axis=1))
    diff = dr25_id_plnt - dr24_id_plnt
    missing = dr24_id_plnt - dr25_id_plnt
    if missing:
        print(len(missing))
        print(list(missing)[:10])
        
    print(f'dr24: {len(dr24_id_plnt)},  '
          f'dr25: {len(dr25_id_plnt)},  '
          f'diff: {len(diff)}')
    return diff


if __name__ == '__main__':
    # compare(3442058)
    # compare(1028018)
    get_not_shown_in_dr24()
