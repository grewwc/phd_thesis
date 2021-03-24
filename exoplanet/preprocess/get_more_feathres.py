from clean_utils.sorting import sort_df
from config import *
import os
import pandas as pd
import numpy as np
from clean_utils.normalization import norm_kepid, norm_features
import warnings


# warnings.filterwarnings('error')
def drop_unknown_label(dr24=True):
    if dr24:
        csv_path = os.path.join(csv_folder, csv_name)
        csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    else:
        csv_path = os.path.join(csv_folder, csv_name_25)
        csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk_25)

    if os.path.exists(csv_clean_path):
        return
    data = pd.read_csv(csv_path, comment='#')
    data = data[data['av_training_set'] != 'UNK']
    data.dropna(axis=1, inplace=True)
    data.to_csv(csv_clean_path, index=False)


def get_more_features(columns=None, kepid=None, dr24=True):
    if columns is None:
        columns = ['tce_period', 'tce_impact',
                   'tce_duration', 'tce_depth',
                   'tce_ror', 'tce_num_transits', 'tce_dor',
                   'tce_incl', 'tce_ingress',
                   'tce_eqt', 'tce_steff', 'tce_slogg',
                   'tce_model_snr', 'tce_model_chisq', 'tce_robstat',
                   'tce_prad', 'tce_sradius']
    if dr24:
        fname = os.path.join(csv_folder, csv_name_drop_unk)
    else:
        fname = os.path.join(csv_folder, csv_name_25)

    if not os.path.exists(fname):
        drop_unknown_label()

    df24 = pd.read_csv(fname, comment='#')
    df24['norm_kepid'] = df24['kepid'].apply(norm_kepid)
    if kepid is not None:
        df24 = df24[df24['norm_kepid'] == norm_kepid(kepid)]

    if dr24:
        sort_df(df24)
    else:
        df24.sort_values(by=['norm_kepid', 'tce_plnt_num'],
                         ascending=[True, True],
                         inplace=True, kind='mergesort')
    return df24[columns]


def write_more_features(columns=None):
    """
    :param columns: what features to write in dr24 file
        default: ['tce_impact', 'tce_duration', 'tce_depth']
    :return: None

    (feature_file, label_file) = (f1.txt, l1.txt)
    """
    feature_file = os.path.join(train_root_dir, 'features', 'f1.txt')
    label_file = os.path.join(train_root_dir, 'features', 'l1.txt')

    # register the two files to the GlobalVars

    if not os.path.exists(os.path.dirname(feature_file)):
        os.makedirs(os.path.dirname(feature_file))

    features = get_more_features(columns)
    labels = get_more_features('av_training_set')
    labels = list(map(lambda label: 1 if label == 'PC' else 0, labels))
    labels = np.array(labels).astype(np.int)
    # print(features.values)
    values = norm_features(features.values)
    np.savetxt(feature_file, values, fmt="%.6f")
    np.savetxt(label_file, labels)


if __name__ == '__main__':
    # print(get_more_features().head())
    write_more_features()
