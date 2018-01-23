""" Allows accessing of topic elements by index 

Author Cody Phillips
"""

import numpy as np
import image_reader_view_collection as IRVC
import image_view as IV


class TopicReader:
    """ Allows index access of topic msgs """

    def __init__(self, indexer, topic, positions=None, format_fun=None):
        self.topic = topic
        if positions is None:
            positions = indexer.positions_of_topic[topic]
        self.positions = positions
        self.indexer = indexer
        self.format_fun = format_fun

    def __getitem__(self, index):
        """ Get message at specified index """
        position = self.positions[index]
        msg = self.indexer.read_message(position)
        if self.format_fun:
            msg = self.format_fun(msg)
        return msg

    def __len__(self):
        """ Get message count for topic """
        return len(self.positions)


class ImageTopicReaderView(IV.ImageView):
    """ Image view from topic reader """

    def __init__(self, reader, key):
        IV.ImageView.__init__(self)
        self.reader = reader
        self.key = key
        self.header = None
        self.ts = None

    def get_timestamp(self):
        if self.ts is None:
            header = self.get_header()
            self.ts = header.stamp.to_nsec()
        return self.ts

    def get_header(self):
        if self.header is None:
            self.acquire()
        return self.header

    def acquire(self):
        if self.img is None:
            self.img, self.header = self.reader(self.key)
            self.get_timestamp()


class ImageTopicReader(TopicReader):
    """ Allows accessing image array and header by index """

    def __init__(self, indexer, topic, positions=None):
        TopicReader.__init__(self, indexer, topic, positions)
        self.compressed = "compressed" in topic
        self.depth = "depth" in topic
        self.use_raw = False
        import cv_bridge
        self.bridge = cv_bridge.CvBridge()

    def set_use_raw(self, use_raw):
        self.use_raw = use_raw

    def __call__(self, index):
        """ Get image array and header at index """
        bag_msg = TopicReader.__getitem__(self, index)
        if self.use_raw:
            return bag_msg.message, bag_msg.message.header
        if self.compressed:
            img = np.array(self.bridge.compressed_imgmsg_to_cv2(
                bag_msg.message))
            img = img[..., [2, 1, 0]]
        elif self.depth:
            img = np.array(self.bridge.imgmsg_to_cv2(bag_msg.message))
        else:
            img = np.array(self.bridge.imgmsg_to_cv2(bag_msg.message))
            img = img[..., [2, 1, 0]]
        return img, bag_msg.message.header

    def __getitem__(self, index):
        """ Get image view at index """
        return ImageTopicReaderView(self, index)

    def __contains__(self, key):
        """ Return if the key can be read """
        return (0 <= key and key < len(self))

    def get_image_views(self, indices=None):
        if indices is None:
            indices = np.arange(len(self))
        return IRVC.ImageReaderViewCollection(self, indices)
