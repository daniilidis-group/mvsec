"""
Writes a bag to h5py format
"""

import mvsec_reader as MR
import h5py
import numpy as np
from tqdm import tqdm

def create_side_group(hdf5_file, camera_type, side):
    if camera_type in hdf5_file.keys():
        camera_group = hdf5_file[camera_type]
    else:
        camera_group = hdf5_file.create_group(camera_type)

    if side is None:
        return camera_group

    if side in camera_group.keys():
        side_group = camera_group[side]
    else:
        side_group = camera_group.create_group(side)

    return side_group

def convert_events(hdf5_file, bag_path, side='left'):
    print("============EVENTS %s=============" % (side))
    mvsec_event_reader = MR.EventReader(bag_path, side)
    camera_type = 'davis'

    side_group = create_side_group(hdf5_file, camera_type, side)

    events_dataset = side_group.create_dataset("events",
            data=mvsec_event_reader.events)

    return mvsec_event_reader

def convert_images(hdf5_file, bag_path,
                   camera_type='davis', side='left', name='image_raw',
                   ts_match=None):
    print("============IMAGES %s %s %s=============" % (camera_type, side, name))
    mvsec_image_reader = MR.ImageReader(bag_path, camera_type, side, name)
    nimages = len(mvsec_image_reader)-1

    side_group = create_side_group(hdf5_file, camera_type, side)

    sample_img = mvsec_image_reader[0][0]
    images_dataset = side_group.create_dataset(name,
            shape=tuple([nimages]+list(sample_img.shape)),
            dtype=sample_img.dtype,
            compression="lzf",
            chunks=tuple([1]+list(sample_img.shape)))

    image_times_dataset = side_group.create_dataset(name+"_ts",
            shape=(nimages,),
            dtype=np.float)

    if ts_match is not None:
        event_to_image_inds = side_group.create_dataset(name+"_event_inds",
                shape=(nimages,),
                dtype=np.int64)

    for i in tqdm(range(nimages)):
        image, time = mvsec_image_reader[i]
        images_dataset[i,...] = image
        image_times_dataset[i] = time
        if ts_match is not None:
            event_to_image_inds[i] = np.searchsorted(ts_match, time)-1

    return mvsec_image_reader

def convert_flow(hdf5_file, npy_path):
    print("============FLOW=============")
    flow_reader = MR.FlowReader(npy_path)
    side_group = create_side_group(hdf5_file, 'davis', 'left')

    sample_img = flow_reader[0][0]
    nimages = len(flow_reader)

    images = side_group.create_dataset("flow_dist",
            shape=(nimages,
                   sample_img.shape[0],
                   sample_img.shape[1],
                   sample_img.shape[2]),
            dtype=sample_img.dtype,
            compression="lzf",
            chunks=(1,sample_img.shape[0],
                      sample_img.shape[1],
                      sample_img.shape[2]))
    times = side_group.create_dataset("flow_dist_ts",
            shape=(nimages,),
            dtype=np.float)

    for i in tqdm(range(nimages)):
        flow, ts = flow_reader[i]
        images[i,...] = flow
        times[i] = ts

    return flow_reader

def convert_odom(hdf5_file, bag_path, topic='odometry'):
    print("============ODOM %s=============" % (topic))
    odom_reader = MR.OdomReader(bag_path, topic)
    side_group = create_side_group(hdf5_file, 'davis', 'left')

    n_samples = len(odom_reader)
    odom = side_group.create_dataset(topic,
                                shape=(n_samples,4,4),
                                dtype=np.float)
    times = side_group.create_dataset(topic+"_ts",
                                 shape=(n_samples,),
                                 dtype=np.float)
    for i in tqdm(range(n_samples)):
        o, t = odom_reader[i]
        odom[i,...] = o
        times[i] = t

def convert_imu(hdf5_file, bag_path, camera, side):
    print("============IMU %s %s=============" % (camera, side))
    if side is None:
        topic = "/"+camera+"/imu"
    else:
        topic = "/"+camera+"/"+side+"/imu"

    imu_reader = MR.IMUReader(bag_path, topic)
    side_group = create_side_group(hdf5_file, camera, side)

    n_samples = len(imu_reader)
    imu = side_group.create_dataset('imu',
                                shape=(n_samples,6),
                                dtype=np.float)
    times = side_group.create_dataset("imu_ts",
                                 shape=(n_samples,),
                                 dtype=np.float)
    for i in tqdm(range(n_samples)):
        aw, t = imu_reader[i]
        imu[i,...] = aw
        times[i] = t

def convert_velodyne(hdf5_file, bag_path):
    print("============VELODYNE===============")
    velo_reader = MR.VelodyneReader(bag_path)

    n_samples = len(velo_reader)
    max_scan_size = velo_reader.largest_scan_size()

    velo_group = hdf5_file.create_group("velodyne")

    velo_dataset = velo_group.create_dataset("scans",
            shape=(n_samples, max_scan_size, 4),
            dtype=np.float32)
    times_dataset = velo_group.create_dataset("scans_ts",
            shape=(n_samples,),
            dtype=np.float)

    for i in tqdm(range(n_samples)):
        padded_scan = np.empty((max_scan_size, 4),
                               dtype=np.float32)
        padded_scan[...] = np.nan

        scan, ts = velo_reader[i]
        padded_scan[:scan.shape[0],:] = scan

        velo_dataset[i,...] = padded_scan
        times_dataset[i] = ts

def convert_data(path):
    has_visensor = "outdoor_" in path or "motorcycle" in path
    has_right_images = not "outdoor_day" in path
    has_ground_truth = not "motorcycle" in path

    data_bag_path = path+'_data.bag'

    data_file = h5py.File(path+"_data.hdf5", 'w')

    davis_group = data_file.create_group("davis")
    davis_left_group = davis_group.create_group("left")

    left_events = convert_events(data_file, data_bag_path, 'left')
    convert_imu(data_file, data_bag_path, "davis", "left")
    convert_images(data_file, data_bag_path,
                   "davis", "left", "image_raw",
                   ts_match=left_events.events[:,2])
    left_events = None

    right_events = convert_events(data_file, data_bag_path, 'right')
    convert_imu(data_file, data_bag_path, "davis", "right")
    if has_right_images:
        convert_images(data_file, data_bag_path,
                       "davis", "right", "image_raw",
                       ts_match=right_events.events[:,2])
    right_events = None

    if has_visensor:
        convert_images(data_file, data_bag_path,
                       "visensor", "right", "image_raw")
        convert_images(data_file, data_bag_path,
                       "visensor", "left", "image_raw")
        convert_imu(data_file, data_bag_path, "visensor", None)

    if has_ground_truth:
        convert_velodyne(data_file, data_bag_path)

    data_file.flush()
    data_file.close()

def convert_gt(path):
    has_visensor = "outdoor_" in path or "motorcycle" in path
    has_right_images = not "outdoor_day" in path
    has_ground_truth = not "motorcycle" in path

    if not has_ground_truth:
        return

    gt_bag_path = path+'_gt.bag'
    gt_file = h5py.File(path+"_gt.hdf5", 'w')
    convert_images(gt_file, gt_bag_path,
                   "davis", "left", "blended_image_rect")
    convert_images(gt_file, gt_bag_path,
                   "davis", "left", "depth_image_raw")
    convert_images(gt_file, gt_bag_path,
                   "davis", "left", "depth_image_rect")

    if has_right_images:
        convert_images(gt_file, gt_bag_path,
                       "davis", "right", "blended_image_rect")
        convert_images(gt_file, gt_bag_path,
                       "davis", "right", "depth_image_raw")
        convert_images(gt_file, gt_bag_path,
                       "davis", "right", "depth_image_rect")

    convert_odom(gt_file, gt_bag_path, 'odometry')
    convert_odom(gt_file, gt_bag_path, 'pose')

    flow_path = path+'_gt_flow_dist.npz'
    convert_flow(gt_file, flow_path)

    gt_file.flush()
    gt_file.close()

if __name__ == "__main__":
    dataset_list = [
            # '/home/ken/datasets/mvsec/indoor_flying/indoor_flying1',
            # '/home/ken/datasets/mvsec/indoor_flying/indoor_flying2',
            # '/home/ken/datasets/mvsec/indoor_flying/indoor_flying3',
            # '/home/ken/datasets/mvsec/indoor_flying/indoor_flying4',
            # '/home/ken/datasets/mvsec/outdoor_day/outdoor_day1',
            # '/home/ken/datasets/mvsec/outdoor_day/outdoor_day2',
            '/home/ken/datasets/mvsec/outdoor_night/outdoor_night1',
            # '/home/ken/datasets/mvsec/outdoor_night/outdoor_night2',
            # '/home/ken/datasets/mvsec/outdoor_night/outdoor_night3',
            # '/home/ken/datasets/mvsec/motorcycle/motorcycle1',
            ]
    for seq in dataset_list:
        convert_data(seq)
        convert_gt(seq)
