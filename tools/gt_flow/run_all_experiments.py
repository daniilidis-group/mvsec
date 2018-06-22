import compute_flow

import downloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mvsec_dir',
                    type=str,
                    help="Path to MVSEC directory.",
                    required=True)
args = parser.parse_args()

downloader.set_tmp(args.mvsec_dir)

#exp_data = compute_flow.experiment_flow("indoor_flying", 1, save_movie=False)

for experiment, n_runs in zip(downloader.experiments, downloader.number_of_runs):
    for i in range(n_runs):
        run_number = i+1
        print "Running ", experiment, run_number
        compute_flow.experiment_flow(experiment, run_number, save_movie=False)
