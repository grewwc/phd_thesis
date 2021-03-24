from config import train_root_dir
import os
import numpy as np


def norm_kepid(kepid):
    return f'{int(kepid):09d}'


def get_global_fname_by_kepid(kepid, plnt_num, label):
    kepid = norm_kepid(kepid)
    label = str(label)
    fname = f'flux_{plnt_num}.txt'
    abs_name = os.path.join(
        train_root_dir, 'flux', label,
        kepid[:4], kepid, fname
    )
    return abs_name


def norm_features(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    try:
        values = (values - mean) / std
    except Warning as e:
        print(e, std)
    return values
