# package: preprocess

import json
import multiprocessing as mp
import pickle
import random
from collections import namedtuple
from itertools import chain, repeat

import pandas as pd
from astropy.io import fits
from clean_utils.log_utils import get_logger
from clean_utils.normalization import norm_kepid, norm_features
from clean_utils.sorting import sort_df
from preprocess.get_more_feathres import get_more_features
from tools.decorators import load_ctx, save_ctx
from utils.functions import *

# append kepler as python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

kepid = 'kepid'

__norm_features = None
Item = namedtuple('Item', "kepid, plnt_num, label, is_global, data")

df = None
__df_clean = None

_float_fmt = '%.6f'

# scramble patters
__scramble_1 = [13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4, 17]
__scramble_2 = [1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 17]
__scramble_3 = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17]

__scramble_1 = [x - 1 for x in __scramble_1]
__scramble_2 = [x - 1 for x in __scramble_2]
__scramble_3 = [x - 1 for x in __scramble_3]

scramble_patterns = [__scramble_1, __scramble_2, __scramble_3]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from config import *

# root dir is "kepler", same as os.getcwd()
go_download_target_name = os.path.join(
    os.getcwd(), "go_src", "download_target", "main.go")


def get_kepler_ids_from_csv(csv_name=csv_name):
    """this function removes duplicate ids, because some kepid has many moons
    """
    csv_filename = os.path.join(csv_folder, csv_name)
    data = pd.read_csv(csv_filename, comment="#")
    return list(set(data[kepid].values))


def get_columns_from_csv(csv_name=csv_name):
    csv_filename = os.path.join(csv_folder, csv_name)
    data = pd.read_csv(csv_filename, comment="#")
    return data.columns


def write_kepler_ids(fname=csv_name):
    # for golang to download the fits files
    kepids = get_kepler_ids_from_csv(fname)
    to_filename = kepid_filename if fname == csv_name else kepid_filename_25
    with open(to_filename, 'w') as f:
        for kepid in kepids:
            kepid = '{:09d}'.format(kepid)
            f.write(str(kepid) + '\n')


def download_target():
    if not os.path.exists(kepid_filename):
        write_kepler_ids()
    cmd = f"go run {go_download_target_name} -file {kepid_filename}"
    os.system(cmd)


def download_dr25_target():
    if not os.path.exists(kepid_filename_25):
        write_kepler_ids(csv_name_25)
    cmd = f'go run {go_download_target_name} -file {kepid_filename_25}'
    os.system(cmd)


# @main_tag
def download_by_id(kepid, to_dir="."):
    kepid = f'{int(kepid):09d}'
    cmd = f"go run {go_download_filename} -id {kepid} -writeTo {to_dir}"
    # print(cmd)
    os.system(cmd)


# @main_tag
def get_time_flux_by_ID(kepid,
                        quarter=None,
                        scramble_id: int = None):
    if scramble_id is not None:
        assert scramble_id == 0 or scramble_id == 1 or scramble_id == 2, \
            "only allow 3 patters (0, 1, 2)"

    if (quarter is not None) and (scramble_id is not None):
        warnings.warn(
            "quarter and scramble_id not compatible, ignore scramble_id")
        scramble_id = None

    all_time, all_flux = [], []
    kepid = f'{int(kepid):09d}'
    root_dir = os.path.join(train_root_dir, kepid[:4], kepid)
    files = os.listdir(root_dir)
    files = sorted(files)
    for file in files:
        file = os.path.join(root_dir, file)
        with fits.open(file) as hdu:
            cur_quarter = hdu[0].header['quarter']
            flux = hdu[1].data.PDCSAP_FLUX
            time = hdu[1].data.TIME
            # remove nan
            finite_mask = np.isfinite(flux)
            time = time[finite_mask]
            flux = flux[finite_mask]
            flux /= np.median(flux)

            if quarter is not None and str(cur_quarter) == str(quarter):
                return time, flux

            ##################################
            # later add remove outliers ???  #
            #                                #
            ##################################

            all_time.append(time)
            all_flux.append(flux)

    # quarter is beyond [1,17]
    if quarter is not None:
        print(f'quarter {quarter} not exist')
        return None

    # use scramble
    if scramble_id is not None:
        scr_time = [all_time[i]
                    for i in scramble_patterns[scramble_id]]
        scr_flux = [all_flux[i]
                    for i in scramble_patterns[scramble_id]]
        all_time = scr_time
        all_flux = scr_flux

    all_time = np.concatenate(all_time)
    all_flux = np.concatenate(all_flux)

    return all_time, all_flux


def drop_unknown_label():
    csv_path = os.path.join(csv_folder, csv_name)
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    if os.path.exists(csv_clean_path):
        return
    data = pd.read_csv(csv_path, comment='#')
    data = data[data['av_training_set'] != 'UNK']
    data.dropna(axis=1, inplace=True)
    data.to_csv(csv_clean_path, index=False)


# @main_tag
def get_PC_IDs(num=1, shuffle=True, nth=1):
    """
    if num == np.inf, return all (default 1)
    """
    global __df_clean
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    # if no cleaned csv file (with UNK label dropped), create one
    if not os.path.exists(csv_clean_path):
        drop_unknown_label()

    if __df_clean is None:
        __df_clean = pd.read_csv(csv_clean_path, comment='#')
    all_ids = set(__df_clean[__df_clean['av_training_set'] == 'PC']['kepid'])
    all_ids = [f'{int(id_):09d}' for id_ in all_ids]
    if shuffle:
        random.shuffle(all_ids)

    res = all_ids[(nth - 1) * num:nth * num] \
        if num != np.inf else all_ids
    if not shuffle:
        res = sorted(res)
    return res


# @main_tag
def get_NonPC_IDs(num=1, shuffle=True, nth=1):
    """
    if num == np.inf, return all (default 1)
    """
    global __df_clean
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    # if no cleaned csv file (with UNK label dropped), create one
    if not os.path.exists(csv_clean_path):
        drop_unknown_label()

    if __df_clean is None:
        __df_clean = pd.read_csv(csv_clean_path, comment='#')

    training_set = __df_clean['av_training_set']
    all_ids = set(__df_clean[training_set != 'PC']['kepid'].values)
    # all_pcs = set(df_clean[training_set == 'PC']['kepid'].values)

    # all_ids = all_ids
    all_ids = [f'{int(id_):09d}' for id_ in all_ids]

    if shuffle:
        random.shuffle(all_ids)
    res = all_ids[(nth - 1) * num:nth * num] \
        if num != np.inf else all_ids

    if not shuffle:
        res = sorted(res)
    return res


def __get_item_from_csv_by_IDs(kepids, item_name):
    global df
    csv_path = os.path.join(csv_folder, csv_name_drop_unk)

    if df is None:
        df = pd.read_csv(csv_path, comment="#")

    assert item_name in df.columns, f"{csv_path} don't have '{item_name}'"
    if not isinstance(kepids, list):
        kepids = [kepids]
    res = {}
    for id_ in kepids:
        target = df[df['kepid'] == int(id_)]
        training_set = target['av_training_set']
        target_is_pc = target[training_set == 'PC']
        target_non_pc = target[training_set != 'PC']
        pc_periods = zip(repeat('1', len(target_is_pc)),
                         target_is_pc[item_name])
        non_pc_periods = zip(repeat('0', len(target_non_pc)),
                             target_non_pc[item_name])

        res[f'{int(id_):09d}'] = list(chain(pc_periods, non_pc_periods))
    return res


def __get_multiple_items_from_csv_by_IDs(kepids, item_names):
    # assert isinstance(item_names, list), \
    #     "second arguments should be list"
    global df
    if df is None:
        csv_path = os.path.join(csv_folder, csv_name_drop_unk)
        df = pd.read_csv(csv_path, comment="#")
    if not isinstance(kepids, list):
        kepids = [kepids]
    res = []
    for item_name in item_names:
        # assert item_name in df.columns, f"{csv_path} don't have '{item_name}'"
        res_each_item = {}
        for id_ in kepids:
            int_id = int(id_)
            target = df[df['kepid'] == int_id]

            training_set = target['av_training_set']
            target_is_pc = target[training_set == 'PC']
            target_non_pc = target[training_set != 'PC']

            pc_items = zip(repeat('1', len(target_is_pc)),
                           target_is_pc[item_name])
            non_pc_items = zip(repeat('0', len(target_non_pc)),
                               target_non_pc[item_name])
            res_each_item[f'{int_id:09d}'] = list(
                chain(pc_items, non_pc_items))
        res.append(res_each_item)
    return res


def get_period_by_IDs(kepids):
    """
    kepid is either int, str, or list

    return dict (e.g.: {kepid: [(l1, p1), (l2, p2)]})
    'l1' is the label (0 / 1), and 'p1' is the period
    """
    return __get_item_from_csv_by_IDs(kepids, 'tce_period')


def get_tce_duration_by_IDs(kepids):
    """
    return tce durations in days
    """
    res = __get_item_from_csv_by_IDs(kepids, 'tce_duration')
    # hours -> days
    for kepid, values in res.items():
        res[kepid] = [(value[0], value[1] / 24.0) for value in values]
    return res


def get_tce_epochs_by_IDs(kepids):
    """
    return tce epochs
    """
    return __get_item_from_csv_by_IDs(kepids, 'tce_time0bk')


def write_flux_by_IDs(kepids,
                      overwirte=False,
                      num_bins=num_bins,
                      scramble_id=None):
    """
    write to "${train_root_dir}/flux/${label}kepid[:4]/kepid/flux_p[i].txt"
    flux_p[i]: some kepid may have multiple periods
    """
    periods, durations, first_epochs = \
        __get_multiple_items_from_csv_by_IDs(
            kepids, ['tce_period', 'tce_duration', 'tce_time0bk'])

    if not isinstance(kepids, list):
        kepids = [int(kepids)]

    for id_ in kepids:
        print('why', id_)
        id_str = f'{int(id_):09d}'
        period_list = [p[1] for p in periods[id_str]]
        t0_list = [t[1] for t in first_epochs[id_str]]
        duration_list = [d[1] / 24.0 for d in durations[id_str]]
        for i, ((label, p), (_, duration), (_, t0)) in \
                enumerate(zip(
                    periods[id_str],
                    durations[id_str],
                    first_epochs[id_str]
                )):

            fname = os.path.join(
                train_root_dir, 'flux', label,
                id_str[:4], id_str, f"flux_{i}.txt"
            )
            dirname = os.path.dirname(fname)
            # create directory
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # if flux already exists && !overwrite, continue
            if os.path.exists(fname) and not overwirte:
                continue

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # real work is done here
            time, flux = get_time_flux_by_ID(
                id_,
                scramble_id=scramble_id
            )

            time, flux = remove_points_other_tce(
                time, flux, p, period_list,
                t0_list, duration_list
            )

            duration /= 24.0

            # sigma_clip(time, flux)
            # print("begin", fname)
            binned = process_global(time, flux, p, t0, duration)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            np.savetxt(fname, binned)


def get_binned_normalized_flux_by_IDs(kepids,
                                      merge=True,
                                      overwrite=False,
                                      num_bins=num_bins,
                                      bin_width=None,
                                      scramble_id=None):
    """
    this is the main api for usage
    """
    flux_root_dir = os.path.join(train_root_dir, "flux")
    periods = get_period_by_IDs(kepids)
    if not isinstance(kepids, list):
        # assuming kepids is string or int
        kepids = [int(kepids)]

    count = 0
    kepids_len = len(kepids)
    pc, non_pc = [], []

    # write_flux_by_IDs can distinguish if overwrite
    # so we don't check overwrite here
    write_flux_by_IDs(
        kepids, overwirte=overwrite,
        num_bins=num_bins,
        scramble_id=scramble_id)

    for id_ in kepids:
        id_str = f'{int(id_):09d}'
        for i, (label, p) in enumerate(periods[id_str]):
            # label = '1' if id_str in yes else '0'
            fname = os.path.join(
                flux_root_dir, label, id_str[:4],
                id_str, f'flux_{i}.txt'
            )

            if label == '1':
                pc.append(np.loadtxt(fname).reshape(1, -1))
            else:
                non_pc.append(np.loadtxt(fname).reshape(1, -1))

        count += 1
        write_info(f'{count / kepids_len * 100: .2f}%')

    if merge:
        pc = np.concatenate(pc) if len(pc) != 0 else None
        non_pc = np.concatenate(non_pc) if len(non_pc) != 0 else None

    return pc, non_pc


@deperacated
def load_classification_info():
    yes, no = None, None
    yes_filename = os.path.join(classification_dir, yes_pickle)
    no_filename = os.path.join(classification_dir, no_pickle)
    if (not os.path.exists(yes_filename)) or \
            (not os.path.exists(no_filename)):
        write_classification_info()
    with open(yes_filename, 'rb') as f:
        yes = pickle.load(f)
    with open(no_filename, 'rb') as f:
        no = pickle.load(f)

    if (yes is None) or (no is None):
        print(f'cannot load {yes_filename}/{no_filename}')
        sys.exit(-1)
    return yes, no


@deperacated
def write_classification_info():
    all_pcs = get_PC_IDs(np.inf)
    all_others = get_NonPC_IDs(np.inf)
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    data = pd.read_csv(csv_clean_path, comment="#")
    yes, no = set(all_pcs), set(all_others)
    yes_filename = os.path.join(classification_dir, yes_pickle)
    no_filename = os.path.join(classification_dir, no_pickle)
    if not os.path.exists(os.path.dirname(yes_filename)):
        os.makedirs(os.path.dirname(yes_filename))

    with open(yes_filename, 'wb') as f:
        pickle.dump(yes, f)

    with open(no_filename, 'wb') as f:
        pickle.dump(no, f)

    print(f"finished writing {yes_filename} & {no_filename}")


def get_binned_normalized_PC_flux(num=1,
                                  merge=True,
                                  shuffle=False,
                                  num_bins=num_bins,
                                  overwrite=False,
                                  bin_width=None,
                                  return_kepids=False,
                                  scramble_id=None,
                                  nth=1):
    if num == np.inf and os.path.exists(all_pc_flux_filename) \
            and not overwrite:
        print('argument "return_kepids" is ignored')
        with load_ctx(all_pc_flux_filename):
            res = np.loadtxt(all_pc_flux_filename)
        return res

    if shuffle:
        if nth == 1:
            print("shuffle used")
        else:
            shuffle = False  # ignore shuffle
            print("shuffle ignored because nth != 1")

    all_pcs = get_PC_IDs(num=num, shuffle=shuffle, nth=nth)
    # print(all_pcs)
    pcs, others = get_binned_normalized_flux_by_IDs(
        all_pcs, merge=merge, overwrite=overwrite,
        num_bins=num_bins, bin_width=bin_width,
        scramble_id=scramble_id)

    if others is not None:
        # raise ValueError(f'{others} should be None')
        """
        do nothing
        """

    # 1. all_pc_flux_filename not exist, and num==np.inf
    # 2. overwrite
    # Under the two situations, the final file will be written
    if num == np.inf and (not os.path.exists(all_pc_flux_filename)
                          or overwrite):
        # write all data to a file
        # filename is import from config.py
        with save_ctx(all_pc_flux_filename):
            np.savetxt(all_pc_flux_filename, pcs, fmt=_float_fmt)

    res = pcs if not return_kepids else (pcs, all_pcs)
    return res


def get_binned_normalized_Non_PC_flux(num=1,
                                      merge=True,
                                      shuffle=False,
                                      overwrite=False,
                                      num_bins=num_bins,
                                      bin_width=None,
                                      return_kepids=False,
                                      scramble_id=None,
                                      nth=1):
    if num == np.inf and os.path.exists(all_non_pc_flux_filename) \
            and not overwrite:
        with load_ctx(all_non_pc_flux_filename):
            res = np.loadtxt(all_non_pc_flux_filename)
        return res

    if shuffle:
        if nth == 1:
            print("shuffle used")
        else:
            shuffle = False  # ignore shuffle
            print("shuffle ignored because nth != 1")

    all_others = get_NonPC_IDs(num=num, shuffle=shuffle, nth=nth)
    # print(all_others)
    pcs, others = get_binned_normalized_flux_by_IDs(
        all_others, merge=merge, overwrite=overwrite,
        num_bins=num_bins, bin_width=bin_width,
        scramble_id=scramble_id)

    if pcs is not None:
        # raise ValueError(f'{pcs} should be None')
        pass

    if num == np.inf and (not os.path.exists(all_non_pc_flux_filename)
                          or overwrite):
        # write to file (import from config.py)
        with save_ctx(all_non_pc_flux_filename):
            np.savetxt(all_non_pc_flux_filename, others,
                       fmt=_float_fmt)
    res = others if not return_kepids else (others, all_others)
    return res


def __get_train_data(total_num,
                     pc_ratio=0.5,
                     shuffle=True,
                     overwrite=False,
                     reshape=True,
                     return_kepids=False):
    if total_num == np.inf:
        print('option "pc_ratio" is ignored')
        if return_kepids:
            print('option "return_kepids" is ignored')
        # get all train_data
        pc_flux = get_binned_normalized_PC_flux(
            np.inf, shuffle=shuffle, overwrite=overwrite)
        non_pc_flux = get_binned_normalized_Non_PC_flux(
            np.inf, shuffle=shuffle, overwrite=overwrite)
    else:
        num_pc = int(total_num * pc_ratio)
        num_non_pc = total_num - num_pc
        if num_pc < 1 or num_non_pc < 1:
            raise ValueError('training set size cannot be 0')

        pc_flux = get_binned_normalized_PC_flux(
            num_pc, shuffle=shuffle, overwrite=overwrite,
            return_kepids=return_kepids)
        non_pc_flux = get_binned_normalized_Non_PC_flux(
            num_non_pc, shuffle=shuffle, overwrite=overwrite,
            return_kepids=return_kepids)
        if return_kepids:
            pc_flux, all_pc_ids = pc_flux
            non_pc_flux, all_non_pc_ids = non_pc_flux
        else:
            all_pc_ids, all_non_pc_ids = None, None

    train_x = np.concatenate([pc_flux, non_pc_flux])
    train_y = np.concatenate(
        [np.ones(len(pc_flux), dtype=np.int), np.zeros(len(non_pc_flux), dtype=np.int)])
    # to be consistent with keras
    if reshape:
        train_x = train_x.reshape(*train_x.shape, 1)

    if not return_kepids:
        return train_x, train_y
    return train_x, train_y, all_pc_ids, all_non_pc_ids


@deperacated
def get_global_train_test_data(total_num=np.inf,
                               pc_ratio=0.5,
                               train_ratio=0.8,
                               shuffle=True,
                               reshape=True,
                               return_kepids=False):
    """
    if return_kepids==False, return (train_x, train_y), (test_x, test_y)
    else return (train_x, train_y), (test_x, test_y), (pc_ids, non_pc_ids)
    """
    if not return_kepids:
        all_x, all_y = __get_train_data(
            total_num, pc_ratio=pc_ratio,
            shuffle=shuffle, overwrite=False,
            reshape=reshape,
            return_kepids=return_kepids
        )
    else:
        all_x, all_y, pc_ids, non_pc_ids = \
            __get_train_data(
                total_num, pc_ratio=pc_ratio,
                shuffle=shuffle, overwrite=False,
                reshape=reshape, return_kepids=return_kepids
            )
    len_train_x, len_train_y = len(all_x), len(all_y)
    assert len_train_x == len_train_y, \
        f"data and label size different ({len_train_x} != {len_train_y})"

    num_pc = int(len_train_x * train_ratio)
    num_non_pc = len_train_x - num_pc
    train_x, test_x = all_x[:num_pc], all_x[num_pc:]
    train_y, test_y = all_y[:num_pc], all_y[num_pc:]

    if not return_kepids:
        return (train_x, train_y), (test_x, test_y)
    return (train_x, train_y), (test_x, test_y), (pc_ids, non_pc_ids)


def get_binned_local_view_by_IDs(kepids,
                                 overwrite=False,
                                 merge=True,
                                 scramble_id=None):
    """
    return binned, normalized local views by kepler IDs
    kepids: list or single kepler ID
    """
    logger = get_logger('get_binned_local_view_by_IDs', 'kepler_io.log')

    # durations = get_tce_duration_by_IDs(kepids)
    periods, durations, first_epochs = \
        __get_multiple_items_from_csv_by_IDs(
            kepids, ['tce_period', 'tce_duration', 'tce_time0bk'])
    pc, non_pc = [], []
    if not isinstance(kepids, list):
        kepids = [kepids]
    count = 1
    test = 0
    for kepid in kepids:
        write_info(f'loading {count}/{len(kepids)}')

        norm_kepid = f'{int(kepid):09d}'
        kepid_duration = durations[norm_kepid]
        kepid_period = periods[norm_kepid]
        kepid_first_epoch = first_epochs[norm_kepid]

        period_list = [p[1] for p in kepid_period]
        t0_list = [t[1] for t in kepid_first_epoch]
        duration_list = [d[1] / 24.0 for d in kepid_duration]

        for i, ((label, period),
                (_, duration),
                (_, t0)) in enumerate(
            zip(kepid_period, kepid_duration, kepid_first_epoch)):

            fname = os.path.join(
                train_root_dir, 'flux', label,
                norm_kepid[:4], norm_kepid, f'local_flux_{i}.txt'
            )

            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            if (not os.path.exists(fname)) or overwrite:
                # real work is done here
                time, flux = get_time_flux_by_ID(
                    norm_kepid,
                    scramble_id=scramble_id
                )

                time, flux = remove_points_other_tce(
                    time, flux, period, period_list,
                    t0_list, duration_list
                )

                t0 %= period
                duration /= 24.0
                try:
                    binned_flux = process_local(
                        time, flux, period, t0, duration
                    )
                except IndexError as e:
                    logger.exception(f"kepid: {kepid}")
                    print()
                    continue
                # write to a file
                np.savetxt(fname, binned_flux, fmt=_float_fmt)
            # read from the file
            if label == '1':
                pc.append(np.loadtxt(fname).reshape(1, -1))
            else:
                test += 1
                non_pc.append(np.loadtxt(fname).reshape(1, -1))
        count += 1
    if merge:
        pc = np.concatenate(pc) if len(pc) != 0 else None
        non_pc = np.concatenate(non_pc) if len(non_pc) != 0 else None
    return pc, non_pc,


def get_local_binned_normalized_PC_flux(num=1,
                                        merge=True,
                                        shuffle=False,
                                        overwrite=False,
                                        scramble_id=None,
                                        return_kepids=False,
                                        nth=1):
    """
    if num == np.inf, return all flux (default to 1)
    return binned, normalized local view of candidate PC flux
    """
    if num == np.inf and os.path.exists(local_all_pc_flux_filename) \
            and not overwrite:
        with load_ctx(local_all_pc_flux_filename):
            res = np.loadtxt(local_all_pc_flux_filename)
            return res

    if shuffle:
        if nth == 1:
            print("shuffle used")
        else:
            print("shuffle ignored")
            shuffle = False
    all_pcs = get_PC_IDs(num=num, shuffle=shuffle, nth=nth)
    # print(all_pcs)
    pcs, others = get_binned_local_view_by_IDs(
        all_pcs, merge=merge, overwrite=overwrite,
        scramble_id=scramble_id
    )

    if num == np.inf and (not os.path.exists(local_all_pc_flux_filename)
                          or overwrite):
        # write all data to a file
        # filename is import from config.py
        with save_ctx(local_all_pc_flux_filename):
            np.savetxt(local_all_pc_flux_filename,
                       pcs, fmt=_float_fmt)

    if return_kepids:
        return pcs, all_pcs
    return pcs


def get_local_binned_normalized_Non_PC_flux(num=1,
                                            merge=True,
                                            shuffle=False,
                                            overwrite=False,
                                            scramble_id=None,
                                            return_kepids=False,
                                            nth=1):
    """
    if num == np.inf, return all flux (default to 1)
    return binned, normalized local view of Non-PC flux
    """
    if num == np.inf and os.path.exists(local_all_non_pc_flux_filename) \
            and not overwrite:
        with load_ctx(local_all_non_pc_flux_filename):
            res = np.loadtxt(local_all_non_pc_flux_filename)
        return res

    if shuffle:
        if nth == 1:
            print("shuffle used")
        else:
            print("shuffle ignored")
            shuffle = False
    all_non_pcs = get_NonPC_IDs(num=num, shuffle=shuffle, nth=nth)
    # print(all_pcs)
    pcs, others = get_binned_local_view_by_IDs(
        all_non_pcs, merge=merge, overwrite=overwrite,
        scramble_id=scramble_id)

    if pcs is not None:
        # raise ValueError(f'{pcs} should be None')
        pass

    if num == np.inf and (not os.path.exists(local_all_non_pc_flux_filename)
                          or overwrite):
        # write all data to a file
        # filename is import from config.py
        with save_ctx(local_all_non_pc_flux_filename):
            np.savetxt(local_all_non_pc_flux_filename,
                       others, fmt=_float_fmt)
    if return_kepids:
        return others, all_non_pcs
    return others


def get_summary_by_IDs(kepid):
    """
    print the "period", "tce duration", "tce epoch" of the kepler id \n
    "tce duration" is in days

    ONLY 1 kepid should be passed
    return None
    """
    norm_kepid = f'{int(kepid):09d}'
    period = get_period_by_IDs(kepid)[norm_kepid]
    duration = get_tce_duration_by_IDs(kepid)[norm_kepid]
    first_epoch = get_tce_epochs_by_IDs(kepid)[norm_kepid]
    print(json.dumps(
        dict(zip(['period', 'tce duration', 'tce epoch'],
                 [period, duration, first_epoch])), indent='  '))


def get_unknown_kepids():
    global df
    if df is None:
        fname = os.path.join(csv_folder, csv_name)
        df = pd.read_csv(fname, comment='#')
    col_name = 'av_training_set'
    unknown = 'UNK'
    kepid = 'kepid'
    return list(df[df[col_name] == unknown][kepid].values)


def get_features_by_ID(kepid, dr24=True):
    global __norm_features, __df_clean
    # generate test_feature

    fname = None
    if dr24:
        fname = os.path.join(csv_folder, csv_name_drop_unk)
    else:
        fname = os.path.join(csv_folder, csv_name_25)

    if __norm_features is None:
        all_features = get_more_features(dr24=dr24)
        __norm_features = norm_features(all_features.values)
    if __df_clean is None:
        __df_clean = pd.read_csv(fname, comment='#')
        __df_clean = sort_df(__df_clean)

    idx = __df_clean[__df_clean['kepid'] == int(kepid)].index[0]
    features = __norm_features[idx]
    return features


def __write_file(queue: mp.Queue):
    sentinel = '0' * 9
    while True:
        next_item = queue.get(block=True, timeout=None)
        kepid = norm_kepid(next_item.kepid)
        if kepid == sentinel:
            break
        label = str(next_item.label)
        plnt_num = str(next_item.plnt_num)
        data = next_item.data
        is_global = next_item.is_global
        filename = f'flux_{plnt_num}.txt' if is_global else f'local_flux_{plnt_num}.txt'

        next_file = os.path.join(
            train_root_dir, 'flux', label, kepid[:4],
            kepid, filename
        )
        if not os.path.exists(os.path.dirname(next_file)):
            print("making directory:", os.path.dirname(next_file))
            os.makedirs(os.path.dirname(next_file))

        np.savetxt(fname=next_file, X=data, fmt=_float_fmt)


def __process_global_local(kepler_info: str,
                           label,
                           time,
                           flux,
                           period,
                           t0,
                           duration,
                           queue: mp.Queue):
    logger = get_logger('process_global_local', 'kepler_io.log')
    try:
        kepid, plnt_num = kepler_info.split('-')
        args = (time, flux, period, t0, duration)
        local_binned_flux = process_local(*args)
        global_binned_flux = process_global(*args)

        local_binned_flux = local_binned_flux.reshape(1, *local_binned_flux.shape)
        global_binned_flux = global_binned_flux.reshape(1, *global_binned_flux.shape)

        queue.put(Item(kepid, plnt_num, label, True, global_binned_flux))
        queue.put(Item(kepid, plnt_num, label, False, local_binned_flux))

        return label, global_binned_flux, local_binned_flux
    except:
        logger.exception(kepler_info)
        return None, None, None
        # 3 Nones for unpack purpose


def write_global_and_local_PC(processes=32):
    pool = mp.Pool(processes=processes)
    write_queue = mp.Manager().Queue()
    write_file_process = mp.Process(
        target=__write_file, args=(write_queue,)
    )
    write_file_process.start()
    # start the writing process now

    fname = os.path.join(csv_folder, csv_name_drop_unk)
    df_clean = pd.read_csv(fname, comment='#')

    # hours to days
    df_clean['tce_duration'] = df_clean['tce_duration'] / 24.0

    sort_df(df_clean)

    result = []
    kepids = list(set(df_clean['kepid'].values))

    # first sort by kepler ids
    kepids = sorted(kepids, key=norm_kepid)

    count = 1
    f = open('./failed.txt', 'w')
    for kepid in kepids:
        try:
            time, flux = get_time_flux_by_ID(kepid)
            targets = df_clean[df_clean['kepid'] == kepid]

            plnt_nums, labels, periods, t0s, durations = \
                targets[['tce_plnt_num', 'av_training_set',
                        'tce_period', 'tce_time0bk', 'tce_duration']] \
                    .values.T

            labels = \
                list(map(lambda label: 1 if label == 'PC' else 0, labels))
            # transfer labels to 0/1 (integer)

            # remove_indices = [i for i, label in enumerate(labels)
            #                   if label == 1]

            remove_indices = [i for i, label in enumerate(labels)]

            remove_periods = [periods[i] for i in remove_indices]
            remove_t0s = [t0s[i] for i in remove_indices]
            remove_durations = [durations[i] for i in remove_indices]
            # only remove the confirmed PCs' transit

            for plnt_num, label, period, t0, duration in zip(
                    plnt_nums, labels, periods, t0s, durations):
                print(f'{count}/{len(df_clean)} ({kepid} - {plnt_num})')

                cur_time, cur_flux = remove_points_other_tce(
                    time, flux, period,
                    period_list=remove_periods,
                    t0_list=remove_t0s,
                    duration_list=remove_durations
                )
                result.append(pool.apply_async(
                    func=__process_global_local,
                    args=(f'{kepid}-{plnt_num}', label,
                        cur_time, cur_flux, period,
                        t0, duration, write_queue)
                ))
                count += 1
        except Exception as e:
            print('failed', kepid)
            f.write(str(kepid))
            f.write('\n')
    f.close()
    all_flux = [p.get() for p in result]
    pool.close()

    write_queue.put(Item('0' * 9, 0, 0, False, None))
    # send the sentinel value to the write_queue

    all_global_pc, all_local_pc = [], []
    all_global_non_pc, all_local_non_pc = [], []

    for label, g, l in all_flux:
        if label is None:
            # exception happens when generating "local views"
            continue
        if int(label) == 1:
            all_global_pc.append(g)
            all_local_pc.append(l)
        elif int(label) == 0:
            all_global_non_pc.append(g)
            all_local_non_pc.append(l)
        else:
            raise ValueError(f"label can only be 1 or 0 ({label})")

    all_global_pc = np.concatenate(all_global_pc)
    all_global_non_pc = np.concatenate(all_global_non_pc)
    all_local_pc = np.concatenate(all_local_pc)
    all_local_non_pc = np.concatenate(all_local_non_pc)

    np.savetxt(all_pc_flux_filename, all_global_pc, fmt=_float_fmt)
    np.savetxt(all_non_pc_flux_filename, all_global_non_pc, fmt=_float_fmt)
    np.savetxt(local_all_pc_flux_filename, all_local_pc, fmt=_float_fmt)
    np.savetxt(local_all_non_pc_flux_filename, all_local_non_pc, fmt=_float_fmt)

    pool.join()
    write_file_process.join()
