# Author Cody Phillips

from __future__ import print_function
from __future__ import division

import os
import topic_reader
import bisect
try:
    import rosbag
    import cv_bridge
    bridge = cv_bridge.CvBridge()
except ImportError:
    pass
import numpy as np

class Stamp:
    def __init__(self, stamp=None):
        if stamp:
            self.secs = stamp.secs
            self.nsecs = stamp.nsecs
            self._secs = stamp.to_sec()
        else:
            self.secs = None
            self.nsecs = None
            self._secs = None

    def to_sec(self):
        return self._secs


class Header:
    def __init__(self, info=None):
        if info:
            self.seq = info.seq
            self.frame_id = info.frame_id
            self.stamp = Stamp(info.stamp)
        else:
            self.seq = None
            self.frame_id = None
            self.stamp = Stamp(None)


class BagInfo:
    def __init__(self):
        self.synced_image_count = dict()
        self.image_count = dict()


def load_bag_info(filename):
    info_filename = filename + "_info.npy"
    try:
        info = np.load(info_filename).ravel()[0]
        return info
    except Exception:
        return BagInfo()


def save_bag_info(filename, info):
    info_filename = filename + "_info.npy"
    np.save(info_filename, info)


def save_indexer(filename, indexer):
    index_filename = filename + "_index.npy"
    np.save(index_filename, indexer)


def load_indexer(filename):
    index_filename = filename + "_index.npy"
    return np.load(index_filename).ravel()[0]


def get_bag_indexer(filename, sync_topic_list=[], topic_filters=[]):
    info = load_bag_info(filename)
    try:
        print("Loading indexer",filename)
        indexer = load_indexer(filename)
        indexer.set_bag(filename=filename)
        # indexer.sync_positions = dict()
        # indexer.compute_approximate_sync_positions(sync_topics)
        # np.save(index_filename, indexer)
    except Exception as ex:
        print(ex)
        print("Creating indexer")
        indexer = BagIndexer()
        indexer.set_bag(filename=filename)
        indexer.create_index()
        save_indexer(filename, indexer)
    updated = False
    for src_topic, dst_topic, filter_fun in topic_filters:
        if dst_topic in indexer.topics:
            continue
        print("Creating filtered topic")
        indexer.create_filtered_topic(src_topic, dst_topic, filter_fun)
        updated = True
    if updated:
        print("Saving updated indexer, filtered_topics")
        save_indexer(filename, indexer)
    updated = False
    for sync_topics, max_deltas in sync_topic_list:
        sync_info = indexer.get_sync_info(sync_topics)
        if sync_info is None:
            print("Sync info missing", sync_topics)
        else:
            print("Sync max deltas %s == %s" % (sync_info[1], max_deltas))
        if sync_info and not isinstance(sync_info[1], list):
            print("Old style max delta encounters, not updating")
            continue
        if sync_info and sync_info[1] == max_deltas:
            continue
        updated = True
        print("Computing approximate sync positions")
        indexer.compute_approximate_sync_positions(sync_topics, max_deltas)
    if updated:
        print("Saving updated indexer, synced_topics")
        save_indexer(filename, indexer)
    updated = False
    for sync_topics, max_deltas in sync_topic_list:
        if not np.all(["image" in topic for topic in sync_topics]):
            continue
        for topic in sync_topics:
            count = len(indexer.get_sync_positions(topic, sync_topics))
            updated = updated or topic not in info.synced_image_count
            print("Update sync count", topic, count)
            info.synced_image_count[topic] = count
    if updated:
        print("Saving bag info", filename)
        save_bag_info(filename, info)
    indexer.info = info
    return indexer


class BagIndexer:
    def __init__(self, bag=None, filename=None):
        self.bag = None
        self.filename = None
        self.set_bag(bag=bag, filename=filename)
        self.sync_info = []
        
    def create_filtered_topic(self, src_topic, dst_topic, filter_fun):
        self.ensure_open()
        reader = self.get_topic_reader("/tf")
        use = np.array([filter_fun(msg) for msg in reader])
        timestamps = self.timestamps_of_topic[src_topic]
        positions = self.positions_of_topic[src_topic]
        self.timestamps_of_topic[dst_topic] = timestamps[use]
        self.positions_of_topic[dst_topic] = positions[use]
        self.topics.append(dst_topic)

    def set_bag(self, bag=None, filename=None):
        if bag is not None:
            self.bag = bag
        if filename is not None:
            self.filename = filename

    def ensure_open(self):
        if self.bag is None and self.filename is None:
            assert(False)
        if self.bag is None:
            self.open(self.filename)

    def open(self, filename):
        print("Opening bag", filename)
        bag = rosbag.Bag(filename)
        self.set_bag(bag=bag, filename=filename)

    def __getstate__(self):
        keys = ["topics", "positions_of_topic",
                "timestamps_of_topic", "sync_info"]
        state = dict([(k, self.__dict__[k]) for k in keys])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.bag = None

    def create_index(self):
        self.ensure_open()
        self.positions_of_topic = dict()
        self.timestamps_of_topic = dict()
        self.topics = set()
        for key, indices in self.bag._connection_indexes.items():
            topic = self.bag._connections[key].topic
            print("Indexing:", topic)
            self.topics.add(topic)
            positions = []
            timestamps = []
            for index in indices:
                position = index.position
                msg = self.bag._read_message(position).message
                if topic == "/tf":
                    msg = msg.transforms[0]
                if not hasattr(msg, "header"):
                    continue
                timestamp = msg.header.stamp.to_nsec()
                timestamps.append(timestamp)
                positions.append(position)
            if len(positions) == 0:
                continue
            if topic in self.positions_of_topic:
                old_positions = self.positions_of_topic[topic]
                old_timestamps = self.timestamps_of_topic[topic]
                assert(len(positions) == len(timestamps))
                assert(len(old_positions) == len(old_timestamps))
                positions = old_positions + positions
                timestamps = old_timestamps + timestamps
            self.positions_of_topic[topic] = positions
            self.timestamps_of_topic[topic] = timestamps
        for key, value in self.positions_of_topic.items():
            self.positions_of_topic[key] = np.array(value)
        for key, value in self.timestamps_of_topic.items():
            self.timestamps_of_topic[key] = np.array(value)
        self.topics = list(self.topics)

    def compute_approximate_sync_positions(self, topics, max_milli_sec_diffs):
        compute_approximate_sync_positions(self, topics, max_milli_sec_diffs)
        topics = tuple(topics)

    def compute_exact_sync_positions(self, topics):
        compute_exact_sync_positions(self, topics)

    def set_sync_info(self, sync_topics, max_deltas, sync_positions, sync_timestamps, indicator):
        sync_index = self.get_sync_index(sync_topics)
        info = (sync_topics, max_deltas, sync_positions, sync_timestamps, indicator)
        if sync_index:
            self.sync_info[sync_index] = info
        else:
            self.sync_info.append(info)

    def get_sync_info(self, sync_topics):
        sync_index = self.get_sync_index(sync_topics)
        if sync_index is None:
            return None
        else:
            return self.sync_info[sync_index]

    def get_sync_index(self, sync_topics):
        if len(self.sync_info) == 0:
            return None
        sync_topic_list = zip(*self.sync_info)[0]
        sync_topic_list = [sorted(stl) for stl in sync_topic_list]
        sync_topics = sorted(sync_topics)
        if sync_topics not in sync_topic_list:
            return None
        return sync_topic_list.index(sync_topics)

    def get_sync_positions(self, topic, sync_topics):
        sync_index = self.get_sync_index(sync_topics)
        sync_topics, _, sync_positions, _, _ = self.sync_info[sync_index]
        top_index = sync_topics.index(topic)
        return sync_positions[top_index]

    def get_sync_timestamps(self, topic, sync_topics):
        sync_index = self.get_sync_index(sync_topics)
        sync_topics, _, _, sync_timestamps, _ = self.sync_info[sync_index]
        top_index = sync_topics.index(topic)
        return sync_timestamps[top_index]

    def read_message(self, position):
        self.ensure_open()
        return self.bag._read_message(position)

    def get_topic_reader(self, topic):
        self.ensure_open()
        return topic_reader.TopicReader(self, topic)

    def get_image_topic_reader(self, topic, sync_topics=None):
        self.ensure_open()
        if sync_topics is not None:
            positions = self.get_sync_positions(topic, sync_topics)
        else:
            positions = None
        return topic_reader.ImageTopicReader(self, topic, positions)

    def get_synced_messages(
            self,
            topics,
            slc=slice(None),
            formatter_fun=lambda msgs: msgs):
        self.ensure_open()
        ts_positions = self.get_sync_positions(topics)
        for ts, positions in ts_positions[slc]:
            msgs = [self.read_message(position) for position in positions]
            yield formatter_fun(msgs)

def compute_closest_sync_pair(Ti, Tj, closest_index, max_milli_sec_diff):
    inf = max_milli_sec_diff * 2
    def abs_milli_diff(ti, tj):
        return int(abs(ti - tj) // 1e6)
    nj = len(Tj)
    for i, ti in enumerate(Ti):
        ti = Ti[i]
        j_le = bisect.bisect_right(Tj, ti)
        j_gt = bisect.bisect_right(Tj, ti)
        if j_le:
            j_le -= 1
            tj_le = Tj[j_le]
            abs_le = abs_milli_diff(ti, tj_le)
        else: 
            abs_le = inf
        if j_gt != nj: 
            tj_gt = Tj[j_gt]
            abs_gt = abs_milli_diff(ti, tj_gt)
        else: 
            abs_gt = inf
        if abs_gt >= inf and abs_le >= inf: 
            continue
        closest_index[i] = j_le if abs_le < abs_gt else j_gt 
        
def compute_approximate_sync_pair(Ti, Tj, closest_index, max_milli_sec_diff):
    inf = max_milli_sec_diff * 2
    def abs_milli_diff(ti, tj):
        return int(abs(ti - tj) // 1e6)
    i = 0
    j = 0
    closest_delta = np.full(len(Ti), inf, dtype=np.int64)
    ni = len(Ti)
    nj = len(Tj)
    while i < ni and j < nj:
        ti = Ti[i]
        tj = Tj[j]
        diff = abs_milli_diff(ti, tj)
        if diff > max_milli_sec_diff:
            if ti < tj:
                i += 1
            else:
                j += 1
        else:
            if diff < closest_delta[i]:  # This must be true until its not
                closest_delta[i] = diff
                closest_index[i] = j
                j += 1
            else:
                i += 1


def compute_most_recent_sync_pair(Ti, Tj, closest_index):
    # i is matched with j such that tj <= ti
    i = 0
    j = 0
    ni = len(Ti)
    nj = len(Tj)
    for i in xrange(ni):
        ti = Ti[i]
        while j < nj:
            tj = Tj[j]
            if tj > ti: 
                break
            j += 1
        j -= 1
        closest_index[i] = j

def compute_approximate_sync_positions(self, topics, max_milli_sec_diffs):
    topic_timestamps = []
    for i, topic in enumerate(topics):
        timestamps = self.timestamps_of_topic[topic]
        topic_timestamps.append(timestamps)
    ref_timestamps = topic_timestamps[0]
    closest_indices_of_ts = np.full(
        (len(ref_timestamps), len(topic_timestamps) - 1), -1, dtype=np.int64)
    sync_data = enumerate(zip(topic_timestamps[1:], max_milli_sec_diffs))
    for j, (timestamps, max_milli_sec_diff) in sync_data:
        closest_index = closest_indices_of_ts.T[j]
        if max_milli_sec_diff == -1:
            compute_most_recent_sync_pair(ref_timestamps, timestamps,
                                          closest_index)
        else:
            compute_closest_sync_pair(ref_timestamps, timestamps,
                                          closest_index, max_milli_sec_diff)
            #compute_approximate_sync_pair(ref_timestamps, timestamps,
            #                              closest_index, max_milli_sec_diff)
        print((closest_index!=-1).sum())
    G = (closest_indices_of_ts != -1).all(1)
    self.orig_indices = np.hstack(
        (np.arange(len(G))[G][:, np.newaxis], closest_indices_of_ts[G]))

    positions = np.zeros((len(topics), G.sum(), 2), dtype=np.int64)
    timestamps = np.zeros((len(topics), G.sum()), dtype=np.int64)
    positions[0] = self.positions_of_topic[topics[0]][G]
    timestamps[0] = self.timestamps_of_topic[topics[0]][G]
    for i, topic in enumerate(topics[1:]):
        indices = closest_indices_of_ts.T[i][G]
        positions[i + 1] = self.positions_of_topic[topic][indices]
        timestamps[i + 1] = self.timestamps_of_topic[topic][indices]
    print([len(ts) for ts in topic_timestamps], positions.shape)
    self.set_sync_info(topics, max_milli_sec_diff, positions, timestamps, G)


def compute_exact_sync_positions(self, topics): 
    assert(False)
    ts_positions_dict = dict()
    n_topics = len(topics)
    for i, topic in enumerate(topics):
        timestamps = self.timestamps_of_topic[topic]
        positions = self.positions_of_topic[topic]
        for j, ts in enumerate(timestamps):
            ts_positions = ts_positions_dict[ts] if ts in ts_positions_dict else [
                None] * n_topics
            ts_positions[i] = positions[j]
            ts_positions_dict[ts] = ts_positions
    ts_positions = sorted(ts_positions_dict.items())
    ts_positions = [ps for ps in ts_positions if None not in ps]
    self.set_sync_info(topics, 0, ts_positions, None)
