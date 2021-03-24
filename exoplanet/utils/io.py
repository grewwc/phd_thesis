import pandas as pd
import os


kepid = 'kepid'
csv_folder = "C:/Users/User/dev/data"
csv_name = 'dr24_tce.csv'

kepid_filename = "./exchange/kepids.txt"


def get_kepler_id_from_csv():
    filename = os.path.join(csv_folder, csv_name)
    try:
        data = pd.read_csv(filename, comment="#")
        return data[kepid].values
    except IOError as e:
        print(e)
        return None


def write_kepler_id(kepids):
    with open(kepid_filename, 'w') as f:
        for kepid in kepids:
            kepid = '{:09d}'.format(kepid)
            f.write(str(kepid) + '\n')
