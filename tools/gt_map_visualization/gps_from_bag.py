#!/usr/bin/env python
#
# example usage:
#


import rosbag, rospy
import glob
import argparse
import numpy as np
import tf
import tf.transformations
import utm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract trajectories from bag file')
    parser.add_argument('--bag', '-b', action='store', default=None, required=True,
                        help='name of bag file')

    topics = ['/gps/fix']
              
    args = parser.parse_args()

    fileNames = ['gps_traj.txt']
    
    gpsFile = open(fileNames[0], 'w')

    for bagfile in [args.bag]:
        bag = rosbag.Bag(bagfile, 'r')
        it = bag.read_messages(topics=topics)
        for (topic, msg, tros) in it:
            if (topic == '/gps/fix'):  # gps long/lat
                utm_pos = utm.from_latlon(msg.latitude, msg.longitude)
                a = msg.altitude # ignored here
                T = tf.transformations.identity_matrix()
                T[0:3,3] = np.array([utm_pos[0], utm_pos[1], 0])
                gpsFile.write("%.6f %s\n" % (msg.header.stamp.to_nsec()/1e9, ' '.join(map(str, T.ravel()[0:12].tolist()))))
                
    gpsFile.close()
    print "wrote to files: %s" % ", ".join(fileNames)
    
