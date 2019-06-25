import numpy as np
import h5py

class MVSECSequence(object):
    def __init__(self, path):
        super().__init__()

        self.path = args.train_file

        self.load()

    def load(self):
        """ Opens the HDF5 file and associated datasets """

        self.sequence = h5py.File(self.path,'r')

        # Select the left davis camera
        self.davis_cam = self.sequence['davis']['left']

        # dense array of raw indexes NxHxW
        self.images = self.davis_cam['image_raw']
        self.raw_image_size = self.images.shape[1:]

        # dense array of all events Nx4 (x,y,t,p)
        self.events = self.davis_cam['events']

        # list of ending event inds for each image Nx1
        self.image_to_event = self.davis_cam['image_raw_event_inds']

        # list of image times for each image Nx1
        self.images_ts = self.davis_cam['image_raw_ts']

        # start at the second frame so you can get the events from 0-1
        self.start_frame = 1

        self.num_images = self.images.shape[0]

        self.loaded = True

    def close(self):
        """ Close the HDF5 file and associated datasets """

        self.images = None
        self.davis_cam = None
        self.images_to_events = None
        self.images_ts = None

        self.sequence.close()
        self.sequence = None
        self.loaded = False

    def get_image(self, ind):
        """ Returns the image and image time at ind

        Arguments:
        - ind - The index of the image relative to start_frame

        Returns:
        - image - The full image at ind
        - image_ts - The timestamp of image
        """

        image = self.images[ind + self.start_frame]
        image_ts = self.images_ts[ind + self.start_frame]

        return image, image_ts

    def get_events(self, pind, cind):
        """ Returns the events between two images

        Arguments:
        - pind - The index of the first image relative to start_frame
        - cind - The index of the second image relative to start_frame

        Returns:
        - events - The events that occured between the two images
        """

        peind = self.image_to_event[pind + self.start_frame]
        ceind = self.image_to_event[cind + self.start_frame]

        events = self.events[peind:ceind,:]
        return events

    def __len__(self):
        """ Returns the number of frames after start_frame """
        return self.num_images-self.start_frame

    def __getitem__(self, ind):
        """ Returns all data associated with the ind and the prior frame

        Arguments:
        - ind - The index of the current frame relative to start_frame

        Returns:
        - dict : events - all events between the prior image and current
                 prev_image - the previous image
                 prev_image_ts - the time the previous image was taken
                 cur_image - the current image
                 cur_image_ts - the time the current image was taken
        """

        assert self.loaded

        prev_ind, cur_ind = ind, ind+1

        prev_image, prev_image_ts = self.get_image(prev_ind)
        cur_image, cur_image_ts = self.get_image(cur_ind)

        events = self.get_events(pind, cind)

        return {"events": events,
                "prev_image": prev_image,
                "prev_image_ts": prev_image_ts,
                "cur_image": cur_image,
                "cur_image_ts": cur_image_ts}
