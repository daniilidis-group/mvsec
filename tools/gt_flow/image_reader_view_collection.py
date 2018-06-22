""" Use an image reader to implement a view collection """
import image_view_collection as IVC


class ImageReaderViewCollection(IVC.ImageViewCollection):
    """ Collection of image views obtained from a reader  """

    def __init__(self, reader, keys):
        IVC.ImageViewCollection.__init__(self, keys)
        self.reader = reader

    def get_key_slice(self, slc):
        return ImageReaderViewCollection(self.reader, self.key_list[slc])

    def get_key(self, key):
        return self.reader[key]
# Author Cody Phillips
