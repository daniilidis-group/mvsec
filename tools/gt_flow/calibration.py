""" Handles loading the calbiration for the mvsec rig
"""

import downloader
import numpy as np
import zipfile
import yaml

class Calibration(object):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.calib_zip_fn = downloader.get_calibration(experiment_name)
        with zipfile.ZipFile(self.calib_zip_fn) as calib_zip:
            self.left_map = np.stack(self.load_rectification_map(calib_zip, "left"), axis=2)
            self.right_map = np.stack(self.load_rectification_map(calib_zip, "right"), axis=2)
            self.intrinsic_extrinsic = self.load_yaml(calib_zip)

    def load_rectification_map(self, calib_zip, direction="left"):
        assert direction in ["left", "right"]

        x_name=self.experiment_name+"_"+direction+"_x_map.txt"
        with calib_zip.open(x_name) as f:
            x_map = np.loadtxt(f)

        y_name=self.experiment_name+"_"+direction+"_y_map.txt"
        with calib_zip.open(y_name) as f:
            y_map = np.loadtxt(f)

        return x_map, y_map

    def load_yaml(self, calib_zip):
        yaml_name = "camchain-imucam-"+self.experiment_name+".yaml"
        with calib_zip.open(yaml_name) as yaml_file:
            intrinsic_extrinsic = yaml.load(yaml_file)
        return intrinsic_extrinsic
