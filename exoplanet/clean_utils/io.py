from config import *
import pandas as pd
import os
import numpy as np
from preprocess.get_more_feathres import write_more_features


def get_dr24_columns(columns):
    df = pd.read_csv(os.path.join(csv_folder, csv_name), comment='#')
    return df[columns]


def get_dr25_columns(columns):
    df = pd.read_csv(os.path.join(csv_folder, csv_name_25), comment='#')
    return df[columns]


def get_features_and_labels():
    feature_file = os.path.join(train_root_dir, 'features', 'f1.txt')
    label_file = os.path.join(train_root_dir, 'features', 'l1.txt')
    write_more_features()

    features = np.loadtxt(feature_file)
    labels = np.loadtxt(label_file)
    return features, labels


def drop_unknown_label():
    csv_path = os.path.join(csv_folder, csv_name)
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    if os.path.exists(csv_clean_path):
        return
    data = pd.read_csv(csv_path, comment='#')
    data = data[data['av_training_set'] != 'UNK']
    data.dropna(axis=1, inplace=True)
    data.to_csv(csv_clean_path, index=False)
