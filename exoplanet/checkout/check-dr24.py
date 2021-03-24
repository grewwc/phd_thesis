import pandas as pd
from config import *
from dr25.gen_flux_txt import test_kepid
from models.utils import load_model


def check():
    m = load_model()
    fname = os.path.join(csv_folder, csv_name_drop_unk)
    df24 = pd.read_csv(fname, comment='#')
    # df24['norm_kepid'] = df24['kepid'].apply(norm_kepid)
    #
    # df24['int_label'] = df24['av_training_set'].apply(
    #     lambda x: 1 if x == 'PC' else 0)
    #
    # df24.sort_values(by=['int_label', 'norm_kepid', 'tce_plnt_num'],
    #                  ascending=[False, True, True],
    #                  inplace=True, kind='mergesort')
    # count_kepid = -1

    kepids = set(df24['kepid'].values)
    prev_kepid = None

    count, total = 1, len(df24)
    diff_count = 0
    processed = 0
    with open('diff_kepid.txt', 'w') as f:
        with open('unk_kepid.txt', 'w') as f_unk:
            for kepid in kepids:
                # if prev_kepid != kepid:
                #     count_kepid += 1
                #     prev_kepid = kepid

                res = test_kepid(m, kepid, dr24=True)

                sub_df = df24[df24['kepid'] == int(kepid)]
                for plnt, prob in res.items():
                    cls = sub_df[sub_df['tce_plnt_num'] == int(
                        plnt)]['av_training_set'].values[0]

                    processed += 1
                    if (cls == 'PC' and prob < 0.5) \
                            or (cls != 'PC' and prob > 0.5):
                        # predict wrongly
                        diff_count += 1
                        print(f'diff rate: {diff_count / processed:.3f}')
                        print(f'{kepid}-{plnt} prob: {prob}',
                              file=f)
                        continue

                print(f'{count}/{total}')
                count += 1


if __name__ == '__main__':
    check()
