"""
Simple readers that pull data from the rosbags.

Caching (this takes a while the first time performed):
    - the index of messages in the rosbag as a whole
    - the event stream of all events from topic

How to use?

    $ source /opt/ros/melodic/setup.bash
    $ ipython

In [1]: import mvsec_reader as MR
In [2]: davis_left_events = MR.EventReader('/home/ken/datasets/mvsec/outdoor_day/outdoor_day1_data.bag')
In [3]: davis_left_imgs = MR.ImageReader('/home/ken/datasets/mvsec/outdoor_day/outdoor_day1_data.bag')
"""

from __future__ import print_function
import os
import numpy as np
from bag_indexer import get_bag_indexer
import transformations as tf
from tqdm import tqdm

def load_bag(bag_path):
    """ Loads the bag located at bag_path

        - Applies the proper synchronizations on the topics
          depending on the type of bag
    """

    HAS_LEFT_IMAGES = True
    HAS_RIGHT_IMAGES = not 'outdoor_day' in bag_path
    HAS_VI_SENSOR = not 'indoor_flying' in bag_path
    HAS_SYNCNED_GT = not 'motorcycle' in bag_path

    sync_topics = []
    if '_data.bag' in bag_path:
        if HAS_LEFT_IMAGES:
            sync_topics.append(["/davis/left/image_raw",
                                "/davis/left/camera_info"])
        if HAS_RIGHT_IMAGES:
            sync_topics.append(["/davis/right/image_raw",
                                "/davis/right/camera_info"])
        if HAS_VI_SENSOR:
            sync_topics.append(["/visensor/right/image_raw",
                                "/visensor/right/camera_info",  
                                "/visensor/left/image_raw",
                                "/visensor/left/camera_info"])
    elif '_gt.bag' in bag_path:
        if HAS_SYNCNED_GT:
            sync_topics.append(["/davis/left/blended_image_rect",
                                 "/davis/left/depth_image_raw",
                                 "/davis/left/depth_image_rect",
                                 "/davis/left/odometry",
                                 "/davis/left/camera_info"])
            sync_topics.append(["/davis/right/blended_image_rect",
                                 "/davis/right/depth_image_raw",
                                 "/davis/right/depth_image_rect",
                                 "/davis/right/camera_info"])
    else:
        raise NotImplementedError("The MVSEC datasets only end in\
                _data.bag or _gt.bag")

    sync_topics = [ (t, [0.01]*len(t)) for t in sync_topics ]

    return get_bag_indexer(bag_path, sync_topics)

class EventReader(object):
    """ Pulls all events from given davis to numpy array
        Bag path is to the raw data bag

        Useful attributes:
        - events - numpy array of all events in the choosen
                   dvs stream
    """
    def __init__(self, bag_path, camera_side='left',
            chunk_size=5000, start_time=0., end_time=None):
        self.bag_path = bag_path
        self.bag = None
        self.camera_side = camera_side
        self.chunk_size = chunk_size

        self.events = self.load_events(camera_side)

        self.set_start_time(start_time)
        self.set_end_time(end_time)

        self.cur_ind = 0

        self.chunk_size = chunk_size
        self.total_events = self.events.shape[0]

    def load_events(self, camera_side):
        """ Loads all events from camera_side
        - Caches the events if a cache doesn't yet exist
        """
        npy_cache_path = self.bag_path+"_"+camera_side+"_events.npy"

        if os.path.exists(npy_cache_path):
            events = np.load(npy_cache_path)
        else:
            print("Extracting "+camera_side+" events from bag")
            self.bag = load_bag(self.bag_path)

            event_reader = self.bag.get_topic_reader('/davis/'
                    +camera_side+'/events')

            event_list = []
            for i in tqdm(range(len(event_reader))):
                event_list.append( self.event_msg_to_numpy(event_reader[i][1]) )

            events = np.row_stack( event_list )
            print("Saving cached events", events.shape)
            np.save(npy_cache_path, events)

        return events

    def set_start_time(self, start_time=None):
        """ Set the start time for the iterator
        """
        if start_time is None:
            self.start_ind = 0
        else:
            self.start_ind = np.searchsorted(self.events[:,2],
                    start_time, 'right')

        self.start_time = self.events[self.start_ind, 2]

    def set_end_time(self, end_time=None):
        """ Set the end time for the iterator
        """
        if end_time is None:
            self.end_ind = self.events.shape[0]
        else:
            self.end_ind = np.searchsorted(self.events[:,2],
                    end_time, 'right')

        self.end_time = self.events[self.end_ind-1, 2]

    def get_events_between_times(self, start_time=None, end_time=None):
        """ Gets all events between two times
        """
        if start_time is None:
            start_ind = 0
        else:
            start_ind = np.searchsorted(self.events[:,2], start_time, 'right')

        if end_time is None:
            end_ind = self.events.shape[0]
        else:
            end_ind = np.searchsorted(self.events[:,2], end_time, 'right')-1

        return self.events[start_ind:end_ind, :]

    def get_events_for_timestamp(self, timestamp):
        """ Gets the chunk of events centered around the timestamp
        """
        ind = np.argmin(np.abs(self.events[:,2] - timestamp))
        lind = int(ind-self.chunk_size/2)
        rind = int(ind+self.chunk_size/2)
        return self.events[lind:rind, :]

    def __iter__(self):
        self.cur_ind = self.start_ind
        return self

    def __next__(self):
        """ Get the next chunk of events
        """
        if self.cur_ind >= self.end_ind:
            raise StopIteration

        start_ind = self.cur_ind
        self.cur_ind += self.chunk_size
        stop_ind = min(self.cur_ind, self.end_ind)

        # if the whole event chunk wouldn't be captured exit
        if self.cur_ind > self.end_ind:
            raise StopIteration

        event_chunk = self.events[start_ind:stop_ind, :]

        return event_chunk

    def __len__(self):
        """ Number of chunks in the event stream
        """
        return int(np.floor(float(self.end_ind - self.start_ind)/float(self.chunk_size)))

    next = __next__

    def event_msg_to_numpy(self, msg):
        """ Convert from ros message to numpy array
        """
        event_tuple = np.zeros((len(msg.events), 4))
        event_tuple[:, 0] = np.array([msg.events[i].x for i in range(event_tuple.shape[0])])
        event_tuple[:, 1] = np.array([msg.events[i].y for i in range(event_tuple.shape[0])])
        event_tuple[:, 2] = np.array([msg.events[i].ts.to_sec() for i in range(event_tuple.shape[0])])
        event_tuple[:, 3] = np.array([msg.events[i].polarity for i in range(event_tuple.shape[0])])
        event_tuple[:, 3] -= 0.5
        event_tuple[:, 3] *= 2.0

        return event_tuple


class ImageReader(object):
    """ Provides random access to the images from the rosbag
        Bag path is to the raw data bag

        Useful attributes:
        - timestamps - numpy array of message timestamps
    """
    def __init__(self, bag_path, camera_type='davis',
            camera_side='left', image_topic_suffix='image_raw',
            start_time=None, end_time=None):
        camera = "/"+camera_type+"/"+camera_side

        self.image_topic = str(os.path.join(camera, image_topic_suffix))
        self.info_topic = str(os.path.join(camera, "camera_info"))

        self.bag = load_bag(bag_path)

        self.readers = {}
        for t in [self.image_topic, self.info_topic]:
            if 'image' in t:
                self.readers[t] = self.bag.get_image_topic_reader(t)
            else:
                self.readers[t] = self.bag.get_topic_reader(t)

        safe_suffix = ''.join([c for c in camera if c not in ['/']])
        npy_cache_path = bag_path+"_"+safe_suffix+"_timestamps.npy"

        if os.path.exists(npy_cache_path):
            self.timestamps = np.load(npy_cache_path)
        else:
            self.timestamps = np.zeros((len(self.readers[self.info_topic]),), np.float)

            for i in range(self.timestamps.shape[0]):
                info_msg = self.readers[self.info_topic][i]
                self.timestamps[i] = info_msg[2].to_sec()

            np.save(npy_cache_path, self.timestamps)

        if start_time is None:
            self.start_ind = 0
        else:
            self.start_ind = np.argmin(np.abs( self.timestamps - start_time ))

        if end_time is None:
            self.end_ind = self.timestamps.shape[0]
        else:
            self.end_ind = np.argmin(np.abs( self.timestamps - end_time ))

        self.cur_ind = 0
        self.start_time = self.timestamps[self.start_ind]
        self.end_time = self.timestamps[self.end_ind-1]

    def __iter__(self):
        self.cur_ind = self.start_ind
        return self

    def __next__(self):
        if self.cur_ind >= self.end_ind:
            raise StopIteration

        image = self.get_image(self.cur_ind)
        time = self.timestamps[self.cur_ind]

        self.cur_ind += 1

        return image, time

    def __len__(self):
        return self.end_ind - self.start_ind + 1

    def __getitem__(self, ind):
        return self.get_image(self.start_ind+ind), self.timestamps[self.start_ind+ind]

    next = __next__

    def get_timestamp(self, ind):
        """Gets the timestamp of the image at ind"""
        return self.timestamps[ind]

    def get_closest_ind(self, t):
        """Gets the closest index to a timestamp"""
        return np.argmin(np.abs(self.timestamps - t))

    def get_closest_image(self, t):
        """Gets the closest image to a timestamp"""
        return self.get_image(self.get_closest_ind(t))

    def get_image(self, ind):
        """Gets the image at ind"""
        i = self.readers[self.image_topic][ind]
        i.acquire()
        return i.img


class OdomReader(object):
    def __init__(self, bag_path, topic='odometry',
            start_time=None, end_time=None):

        assert '_gt.bag' in bag_path

        self.bag = load_bag(bag_path)
        full_topic = '/davis/left/'+topic
        self.full_topic = full_topic
        self.odom_reader = self.bag.get_topic_reader(full_topic)

        self.start_ind = 0
        self.end_ind = len(self)

    def __iter__(self):
        self.cur_ind = self.start_ind
        return self

    def __next__(self):
        if self.cur_ind >= self.end_ind:
            raise StopIteration

        self.cur_ind += 1

        return self[self.cur_ind]

    def get_pq(self, msg):
        if hasattr(msg[1], 'pose'):
            pose = msg[1].pose
        else:
            pose = msg[1]

        p = np.array([pose.position.x,
                      pose.position.y,
                      pose.position.z])
        q = np.array([pose.orientation.w,
                      pose.orientation.x,
                      pose.orientation.y,
                      pose.orientation.z])

        return p, q

    def __getitem__(self, ind):
        msg = self.odom_reader[ind]
        p, q = self.get_pq(msg)

        T_w_c = tf.quaternion_matrix(q)
        T_w_c[:3,3] = p

        t = msg[1].header.stamp.to_sec()

        return T_w_c, t

    def __len__(self):
        return len(self.odom_reader)

    next = __next__

class FlowReader(object):
    def __init__(self, numpy_path):
        flow_npz = np.load(numpy_path)
        self.timestamps = flow_npz["timestamps"]
        self.x_flow = flow_npz["x_flow_dist"]
        self.y_flow = flow_npz["y_flow_dist"]
        self.start_ind = 0
        self.end_ind = len(self)

    def __len__(self):
        return self.timestamps.shape[0]

    def __getitem__(self, ind):
        ts = self.timestamps[ind]
        if ind == 0:
            ind = 1
        xf = self.x_flow[ind,:,:]
        yf = self.y_flow[ind,:,:]
        return np.stack([xf, yf]), ts

    def __iter__(self):
        self.cur_ind = self.start_ind
        return self

    def __next__(self):
        if self.cur_ind >= self.end_ind:
            raise StopIteration

        self.cur_ind += 1

        return self[self.cur_ind]


class VelodyneReader(object):
    def __init__(self, bag_path):
        self.bag = load_bag(bag_path)
        self.reader = self.bag.get_topic_reader('/velodyne_point_cloud')

        self.scans = []
        for i in tqdm(range(len(self.reader))):
            self.scans.append(self.msg_to_numpy(self.reader[i]))

    def msg_to_numpy(self, msg):
        pc_msg = msg[1]
        pc = np.frombuffer(msg[1].data, dtype=np.float32)
        pc = pc.reshape(pc.shape[0]/4, 4)
        ts = msg[1].header.stamp.to_sec()
        return pc, ts

    def display(self, ind):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pc, _ = self[ind]

        ax.scatter(pc[::10,0], pc[::10,1], pc[::10,2])

    def largest_scan_size(self):
        return np.max([s[0].shape[0] for s in self.scans])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, ind):
        return self.scans[ind]


class IMUReader(object):
    def __init__(self, bag_path, topic):
        self.bag = load_bag(bag_path)
        self.reader = self.bag.get_topic_reader(topic)

    def msg_to_numpy(self, msg):
        msg = msg[1]
        aw = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z,
                       msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z])

        ts = msg.header.stamp.to_sec()
        return aw, ts

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, ind):
        return self.msg_to_numpy(self.reader[ind])
