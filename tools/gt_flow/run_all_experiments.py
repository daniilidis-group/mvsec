import compute_flow

import downloader
downloader.set_tmp("/NAS/data/mvsec")

for experiment, n_runs in zip(downloader.experiments, downloader.number_of_runs):
    for i in range(n_runs):
        run_number = i+1
        print "Running ", experiment, run_number
        compute_flow.experiment_flow(experiment, run_number)
