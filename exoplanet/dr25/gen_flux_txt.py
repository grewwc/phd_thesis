from config import *
from preprocess.get_more_feathres import get_more_features

sys.path.append(root_dir)

from clean_utils.normalization import norm_features

df = None
df24 = None

__norm_features = None

from preprocess.kepler_io import *


def get_info_by_ID(kepid=None, get_planet=False, dr24=False):
    global df
    global df24
    df = None
    if not dr24:
        if df is None:
            df = pd.read_csv(
                os.path.join(csv_folder, csv_name_25),
                comment='#'
            )
        df = df
    else:
        if df24 is None:
            df24 = pd.read_csv(
                os.path.join(csv_folder, csv_name_drop_unk),
                comment='#'
            )
        df = df24

    if kepid is not None:
        assert not isinstance(kepid, list)
        kepids = [kepid]
    else:
        fname = csv_name if dr24 else csv_name_25
        kepids = get_kepler_ids_from_csv(fname)

    info = {}
    for kepid in kepids:
        kepid = norm_kepid(kepid)
        info[kepid] = {}
        target_info = df[df['kepid'] == int(kepid)]
        info[kepid]['tce_period'] = \
            list(target_info['tce_period'].values)
        info[kepid]['tce_time0bk'] = \
            list(target_info['tce_time0bk'].values)
        info[kepid]['tce_duration'] = \
            list(target_info['tce_duration'].values)

        if get_planet:
            info[kepid]['tce_plnt_num'] = \
                list(target_info['tce_plnt_num'].values)

    if len(kepids) == 1:
        return info[norm_kepid(kepids[0])]
    return info


def write_dr25():
    kepids = get_kepler_ids_from_csv(csv_name_25)
    infos = get_info_by_ID()
    global_flux, local_flux = [], []

    count = 0
    total = len(kepids)
    for kepid in kepids:
        kepid = norm_kepid(kepid)
        info = infos[kepid]

        period_list = info['tce_period']
        t0_list = info['tce_time0bk']
        duration_list = [d / 24.0 for d in info['tce_duration']]

        count += 1
        print(f"{count}/{total}")
        time, flux = get_time_flux_by_ID(kepid)

        for period, t0, duration in \
                zip(period_list, t0_list, duration_list):
            removed_time, removed_flux = \
                remove_points_other_tce(
                    time, flux, period, period_list,
                    t0_list, duration_list
                )

            try:
                local_binned_flux = process_local(
                    removed_time, removed_flux, period, t0, duration
                )
            except:
                print(f'kepid: {kepid}, period: {period}, duration: {duration}')
            else:
                global_binned_flux = process_global(
                    removed_time, removed_flux, period, t0, duration
                )

                global_flux.append(global_binned_flux.reshape(1, -1))
                local_flux.append(local_binned_flux.reshape(1, -1))

    global_flux = np.concatenate(global_flux)
    local_flux = np.concatenate(local_flux)

    if not os.path.exists(os.path.dirname(dr25_global_flux_filename)):
        os.makedirs(os.path.dirname(dr25_global_flux_filename))

    try:
        np.savetxt(dr25_global_flux_filename, global_flux, fmt='%.6f')
        np.savetxt(dr25_local_flux_filename, local_flux, fmt='%.6f')
    except Exception as e:
        print(e)
        np.savetxt('./temp_global.txt', global_flux, fmt='%.6f')
        np.savetxt('./temp_local.txt', local_flux, fmt='%.6f')


def test_kepid(model, kepid, params=None, verbose=False,
               dr24=False, test_feature=None):
    """
    if params is not None, duration is in Hours
    """

    time, flux = get_time_flux_by_ID(kepid)
    info = get_info_by_ID(kepid, get_planet=True, dr24=dr24)

    if test_feature is None:
        test_feature = get_features_by_ID(kepid, dr24=dr24)

    period_list = info['tce_period']
    t0_list = info['tce_time0bk']
    duration_list = [d / 24.0 for d in info['tce_duration']]
    planet_nums = info['tce_plnt_num']
    summary = {}
    total = len(period_list)
    if params is None:
        for i, (period, t0, duration, planet_num) in \
                enumerate(zip(
                    period_list,
                    t0_list,
                    duration_list,
                    planet_nums
                )):
            if verbose:
                write_info(f'loading {i + 1}/{total}')

            time, flux = remove_points_other_tce(
                time, flux, period, period_list,
                t0_list, duration_list
            )

            global_flux = process_global(
                time, flux, period, t0, duration
            )

            local_flux = process_local(
                time, flux, period, t0, duration
            )
            # reshape flux
            global_flux = global_flux.reshape(1, *global_flux.shape, 1)
            local_flux = local_flux.reshape(1, *local_flux.shape, 1)

            pred = model.predict([global_flux,
                                  local_flux,
                                  test_feature.reshape(1, *test_feature.shape)])

            summary[planet_num] = pred[0][1]

    else:
        period, t0, duration = [float(x) for x in params]
        duration /= 24.0
        cur_time, cur_flux = remove_points_other_tce(
            time, flux, period, period_list,
            t0_list, duration_list
        )

        global_flux = process_global(
            cur_time, cur_flux, period, t0, duration
        )

        local_flux = process_local(
            cur_time, cur_flux, period, t0, duration
        )
        # reshape flux
        global_flux = global_flux.reshape(1, *global_flux.shape, 1)
        local_flux = local_flux.reshape(1, *local_flux.shape, 1)

        pred = model.predict([global_flux, local_flux])
        summary[0] = pred[0][0]
    return summary


def find_very_short_period(threshold=1e-4):
    threshold *= 24.0  # hours to day
    df = pd.read_csv(os.path.join(csv_folder, csv_name_25), comment='#')
    df = df[['kepid', 'tce_period', 'tce_duration']]

    def func(row): return float(row['tce_duration']) / \
                          float(row['tce_period']) < threshold

    short = df.apply(func, axis=1)
    return df[short]
