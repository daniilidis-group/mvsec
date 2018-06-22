""" Implement abstract class for getting elements using an ordered key list """
import numpy as np


class KeyListCollection(object):
    """ Abstract class for a key list collection

    Child must implement

    get_key_slice(slc)
    get_key(key)
    """

    def __init__(self, key_list):
        self.key_list = key_list

    def __contains__(self, key):
        """ Check if key in key list """
        return key in self.key_list

    def __len__(self):
        """ Return len of key list """
        return len(self.key_list)

    def get_key_list(self):
        return self.key_list

    def __call__(self, key):
        return self.get_key(key)

    def __getitem__(self, slc):
        """ Operates over key_list """
        if isinstance(slc, slice) or (isinstance(slc, np.ndarray)
                                      and slc.ndim == 1):
            return self.get_key_slice(slc)
        else:
            return self.get_key(self.key_list[slc])

    def __iter__(self):
        """ Iterate through all elements of the key list """
        return (self(key) for key in self.key_list)

    def get_key_slice(self, slc):
        raise NotImplemented

    def get_key(self, key):
        raise NotImplemented
# Author Cody Phillips
