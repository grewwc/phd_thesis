import argparse
import os
import sys

sys.path.append(os.getcwd())
# append the root dir to pythonpath

from dr25.gen_flux_txt import *
from dr25.compare_with_robo import compare
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--download",
                    dest="download",
                    action="store_true",
                    help=f"download light curves from '{os.path.join(csv_folder, csv_name)}'"
                    )

parser.add_argument("--download-dr25",
                    dest="download_dr25",
                    action="store_true",
                    help=f"download light curves from '{os.path.join(csv_folder, csv_name_25)}'"
                    )

parser.add_argument("--gen-pc-flux",
                    dest="gen_pc_flux",
                    nargs='*',
                    help=f"generate normalized PC flux and write to {all_pc_flux_filename}"
                    )

parser.add_argument("--gen-non-pc-flux",
                    dest="gen_other_flux",
                    nargs='*',
                    help=f"generate normalized Non-PC flux and write to {all_non_pc_flux_filename}"
                    )

parser.add_argument("--gen-local-pc-flux",
                    dest="gen_local_pc_flux",
                    nargs='*',
                    help=f"generate normalized local view of PC flux and write to {local_all_pc_flux_filename}"
                    )

parser.add_argument("--gen-local-non-pc-flux",
                    dest="gen_local_other_flux",
                    nargs='*',
                    help=f"generate normalized local view of Non-PC flux and write to {local_all_non_pc_flux_filename}" +
                         "takes 1 additional argument indicating if overwrite"
                    )

parser.add_argument("--gen-all-dr24",
                    dest="gen_all_dr24",
                    action='store_true',
                    help='generate pc, non-pc, local-pc, local-non-pc at the same time')

parser.add_argument("--gen-dr25-flux",
                    dest='gen_dr25_flux',
                    action='store_true',
                    help=f'generate normalized global & local view of dr25 flux and write to {dr25_global_flux_filename}'
                    )

parser.add_argument('--compare',
                    dest='compare',
                    nargs='*',
                    help='compare the deep learning model with the robovetter model')


def main():
    parsed_args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        warnings.warn(f'unknown auguments: {unknown}')
    # only download files
    # after download, quit the function
    if parsed_args.download:
        download_target()
        return

    if parsed_args.download_dr25:
        download_dr25_target()
        return

    if parsed_args.gen_pc_flux is not None:
        assert len(parsed_args.gen_pc_flux) <= 1
        overwrite = True
        if len(parsed_args.gen_pc_flux) > 0:
            overwrite = int(parsed_args.gen_pc_flux[0])
        get_binned_normalized_PC_flux(
            num=np.inf, overwrite=overwrite)

    if parsed_args.gen_other_flux is not None:
        assert len(parsed_args.gen_other_flux) <= 1
        overwrite = True
        if len(parsed_args.gen_other_flux) > 0:
            overwrite = int(parsed_args.gen_other_flux[0])
        get_binned_normalized_Non_PC_flux(
            num=np.inf, overwrite=overwrite)

    if parsed_args.gen_local_pc_flux is not None:
        assert len(parsed_args.gen_local_pc_flux) <= 1
        overwrite = True
        if len(parsed_args.gen_local_pc_flux) > 0:
            overwrite = int(parsed_args.gen_local_pc_flux[0])
        get_local_binned_normalized_PC_flux(
            num=np.inf, overwrite=overwrite)

    if parsed_args.gen_local_other_flux is not None:
        assert len(parsed_args.gen_local_other_flux) <= 1
        overwrite = True
        if len(parsed_args.gen_local_other_flux) > 0:
            overwrite = int(parsed_args.gen_local_other_flux[0])
        get_local_binned_normalized_Non_PC_flux(
            num=np.inf, overwrite=overwrite)

    if parsed_args.gen_all_dr24:
        write_global_and_local_PC()

    if parsed_args.gen_dr25_flux:
        write_dr25()

    if parsed_args.compare is not None:
        threashold = None
        if len(parsed_args.compare) > 0:
            threashold = float(parsed_args.compare[0])
        if threashold is None:
            compare()
        else:
            compare(threashold)


if __name__ == "__main__":
    main()
