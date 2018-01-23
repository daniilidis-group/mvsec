""" Implements a collection of view using a key list """

import key_list_collection as KLC


class ImageViewCollection(KLC.KeyListCollection):
    """ Abstract class for a image view collection

    Child must implement

    get_key_slice(slc)
    get_key(key)
    """

    def __init__(self, key_list):
        KLC.KeyListCollection.__init__(self, key_list)
        self.shape = None

    def set_shape(self, shape):
        self.shape = shape

    def __call__(self, key):
        """ Overides base behavior to set shape """
        item = KLC.KeyListCollection.__call__(self, key)
        if self.shape:
            item.set_shape(self.shape)
        return item

    def __getitem__(self, slc):
        """ Overides base behavior to set shape """
        item = KLC.KeyListCollection.__getitem__(self, slc)
        if self.shape:
            item.set_shape(self.shape)
        return item
# Author Cody Phillips
