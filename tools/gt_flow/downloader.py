from __future__ import print_function

import os
try:
    import wget
    HAS_WGET=True
except ImportError:
    import subprocess
    HAS_WGET=False

MVSEC_URL="http://visiondata.cis.upenn.edu/mvsec"

TMP_FOLDER=""

def set_tmp(new_tmp):
    global TMP_FOLDER
    TMP_FOLDER = new_tmp
    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)
    else:
        assert os.path.isdir(TMP_FOLDER)

def get_tmp():
    return TMP_FOLDER

set_tmp(os.path.expanduser('~/mvsec_data'))

experiments = ["indoor_flying", "outdoor_day",
               "outdoor_night", "motorcycle"]
number_of_runs = [4, 2, 3, 1]

def print_experiments():
    print(zip(experiments, number_of_runs))

def download(urls, filenames, overwrite=False):
    for url, fn in zip(urls, filenames):
        if not os.path.exists(fn) or overwrite:
            if not os.path.exists(os.path.dirname(fn)):
                os.makedirs(os.path.dirname(fn))
            if HAS_WGET:
                wget.download(url, fn)
            else:
                subprocess.call(['wget',
                    '--no-check-certificate',
                    url,
                    '-O', fn])

def get_calibration(experiment, overwrite=False):
    assert experiment in experiments
    full_url = [os.path.join(MVSEC_URL, experiment, experiment+"_calib.zip")]
    full_local = [os.path.join(TMP_FOLDER, experiment, experiment+"_calib.zip")]

    download(full_url, full_local, overwrite)

    return full_local[0]

def get_data(experiment_name, experiment_numbers=None,overwrite=False):
    assert experiment_name in experiments
    if type(experiment_numbers)==int:
        experiment_numbers=[experiment_numbers]
    elif type(experiment_numbers)==list:
        pass
    elif experiment_numbers is None:
        experiment_numbers = range(0, number_of_runs[experiments.index(experiment_name)])
    else:
        raise TypeError("Unsupported type "+type(experiment_numbers))

    base_url = os.path.join(MVSEC_URL, experiment_name, experiment_name)
    full_urls = [base_url+str(n)+"_data.bag" for n in experiment_numbers]

    base_path = os.path.join(TMP_FOLDER, experiment_name, experiment_name)
    full_paths = [base_path+str(n)+"_data.bag" for n in experiment_numbers]

    download(full_urls, full_paths, overwrite)

    return full_paths

def get_ground_truth(experiment_name, experiment_numbers=None,overwrite=False):
    assert experiment_name in experiments
    if type(experiment_numbers)==int:
        experiment_numbers=[experiment_numbers]
    elif type(experiment_numbers)==list:
        pass
    elif experiment_numbers is None:
        experiment_numbers = range(0, number_of_runs[experiments.index(experiment_name)])
    else:
        raise TypeError("Unsupported type "+type(experiment_numbers))

    base_url = os.path.join(MVSEC_URL, experiment_name, experiment_name)
    full_urls = [base_url+str(n)+"_gt.bag" for n in experiment_numbers]

    base_path = os.path.join(TMP_FOLDER, experiment_name, experiment_name)
    full_paths = [base_path+str(n)+"_gt.bag" for n in experiment_numbers]

    download(full_urls, full_paths, overwrite)

    return full_paths
