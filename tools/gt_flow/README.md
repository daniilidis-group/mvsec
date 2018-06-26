# Ground Truth Optical Flow Generation
This repo contains the code to generate the ground truth optical flow presented in "EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras". As the optical flow is entirely generated from the MVSEC dataset, and due to the large space requirements, we do not provide the pre-computed data directly. Instead, the code in this folder will allow you to generate the ground truth flow locally, given the ROS bags from MVSEC.

Note that this folder will generate files which take up a lot of storage. For example, the indoor_flying sequences consume >25G of data. Please make sure your machine has enough storage before running the script.

## Computing Flow
The code expects the MVSEC ground truth bags to be in the following folder structure, with each group of sequences in its own folder:

    ├── indoor_flying
    │   ├── indoor_flying1_gt.bag
    │   ├── indoor_flying2_gt.bag
    │   ...
    ├── outdoor_day
    │   ├── outdoor_day1_gt.bag
    │   ...
    ...

To extract ground truth optical flow from a single bag, run:
`python extract_single_bag.py --mvsec_dir $MVSEC_DIR --sequence $SEQUENCE --sequence_num $SEQUENCE --save_numpy`
where `$MVSEC_DIR` points to the top level of your MVSEC directory, `$SEQUENCE$` is the name of the desired sequence (e.g. indoor_flying), and `$SEQUENCE_NUM` is the number of the desired sequence, e.g. 1. Additional options are: `--save_movie` to save a video file with the output flow for the entire sequence (this may take some time), and `--start_ind` and `--stop_ind` to set the start and stop indicies within the ground truth pose/depth frames.

To extract ground truth optical flow from all available bags (currently: indoor_flying, outdoor_day, outdoor_night), run:
`python run_all_experiments.py --mvsec_dir $MVSEC_DIR`. Note that this will attempt to download the bag if it is not found. Please make sure the bag is present, with the directory structure above, if you do not want this to happen.

## Data Format
The code will generate three files for each bag. For example, for `indoor_flying1`, it will generate: 
- `indoor_flying1_gt_bag_index.npy`, which is used internally to index each bag for rapid reading from the bag in python in future runs. 
- `indoor_flying1_gt_flow.npz`, which contains the ground truth flow for the distorted images. The keys for this numpy file are:
  - `timestamps` - The timestamp for each ground truth frame.
  - `x_flow_dist` - The optical flow in the x direction for the distorted (raw) camera image.
  - `y_flow_dist` - The optical flow in the y direction for the distorted (raw) camera image.
- `indoor_flying1_odom.npz`, which contains the odometry camera poses and computed camera velocities. The keys for this numpy file are:
  - `timestamps` - The timestamp for each odometry value.
  - `pos` - The position of the camera at each time. Note that this is from the odometry, and so is prone to drift over time.
  - `quat` - The rotation of the camera at each time in quaternion form (w,x,y,z). Note that this is also from odometry.
  - `lin_vel` - The computed linear velocity of the camera at each time.
  - `ang_vel` - The computed angular velocity of the camera at each time.

Note that this will only produce optical flow for the raw (distorted) image at this time. We plan to support flow in the rectified image in the future.

## Citations
If you use this dataset, please cite the following paper:

Zhu, A. Z., Yuan, L., Chaney, K., Daniilidis, K. "EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras." Robotics: Science and Systems (2018).
