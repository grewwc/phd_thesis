from os import path
import os
import sys
import platform
from tools.helpers import GlobalVars
import logging

# the root dir of the project
root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

__platform = platform.system()


csv_folder = None
classification_dir = None
train_root_dir = None
go_download_filename = os.path.join(
    "go_src", "download_target", "main.go"
)

if __platform == 'Windows':
    csv_folder = "C:/Users/User/dev/data"
    classification_dir = "C:/Users/User/dev/data/classification_info"
    train_root_dir = "C:/Users/User/dev/data/train"
    test_root_dir = "C:/Users/User/dev/data/test"
elif __platform == 'Linux':
    csv_folder = "/home/chao/dev/data"
    classification_dir = "/home/chao/dev/data/classification_info"
    train_root_dir = "/home/chao/dev/data/train"
    test_root_dir = "/home/chao/dev/data/test"

else:
    print(f"{__platform} is not supported")
    sys.exit(-1)

csv_name = 'q1_q17_dr24_tce.csv'
csv_name_25 = 'q1_q17_dr25_tce.csv'
csv_name_drop_unk = 'q1_q17_dr24_tce_clean.csv'
csv_name_drop_unk_25 = 'q1_q17_dr25_tce_clean.csv'

kepid_filename = "./exchange/kepids.txt"
kepid_filename_25 = "./exchange/kepids_25.txt"

num_bins = 2001
num_local_bins = 201
bin_width_factor = 0.16  # from google ai code

yes_pickle, no_pickle = 'yes.pkl', 'no.pkl'


all_pc_flux_filename = path.join(train_root_dir, "flux", "all_PCs_flux.txt")
all_non_pc_flux_filename = path.join(
    train_root_dir, "flux", "all_Non_PCs_flux.txt")

local_all_pc_flux_filename = path.join(
    train_root_dir, "flux", "local_all_PCs_flux.txt")
local_all_non_pc_flux_filename = path.join(
    train_root_dir, "flux", "local_all_Non_PCs_flux.txt")

dr25_global_flux_filename = path.join(
    test_root_dir, 'global_dr25.txt'
)

dr25_local_flux_filename = path.join(
    test_root_dir, 'local_dr25.txt'
)


# settings for logging
log_dir = path.join(root_dir, 'log')
if not path.exists(log_dir):
    os.makedirs(log_dir)
simple_formatter = logging.Formatter('msg [%(levelname)s]: %(message)s\n')


from tools.helpers import GlobalVars
