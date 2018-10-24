""" Handles the reading of the ros ground truth bags
"""

import downloader
import bag_indexer

class GroundTruth:
    def __init__(self, experiment_name, run_number):
        self.bag_path = downloader.get_ground_truth(experiment_name, run_number)[0]

        left_sync_topics = (['/davis/left/odometry','/davis/left/depth_image_raw','/davis/left/depth_image_rect'],[0.05,0.05,0.05])

        self.bag = bag_indexer.get_bag_indexer(self.bag_path, [left_sync_topics])

        self.left_cam_readers = {}

        for t in left_sync_topics[0]:
            if 'image' in t:
                self.left_cam_readers[t] = self.bag.get_image_topic_reader(t)
            else:
                self.left_cam_readers[t] = self.bag.get_topic_reader(t)

    def play_image(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure()
        left_cam = self.left_cam_readers['/davis/left/depth_image_raw']
        
        first_view = left_cam[0]
        first_view.acquire()
        im = plt.imshow(first_view.img, animated=True)
        #first_view.release()
        
        def updatefig(frame_num, *args):
            view = left_cam[frame_num]
            view.acquire()
            im.set_data(view.img)
            return im,
        
        ani = animation.FuncAnimation(fig, updatefig, frames=len(left_cam), blit=True)
        ani.save("test.mp4")
        plt.show()
