from clean_utils.normalization import norm_kepid


def sort_df(df):
    df['norm_kepid'] = df['kepid'].apply(norm_kepid)
    df['int_label'] = df['av_training_set'].apply(
        lambda x: 1 if x == 'PC' else 0
    )
    df.sort_values(by=['int_label', 'norm_kepid', 'tce_plnt_num'],
                   ascending=[False, True, True],
                   inplace=True, kind='mergesort')

    df = df.reset_index(drop=True)
    return df
