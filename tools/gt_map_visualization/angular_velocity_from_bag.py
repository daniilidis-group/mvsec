#!/usr/bin/env python

#
#

import rosbag, rospy, numpy as np
import sys, os, glob
import argparse
import numpy.linalg
import math
import tf
import tf.transformations



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get angular velocity from bag.')
    parser.add_argument('--start', '-s', action='store', default=rospy.Time(0), type=rospy.Time,
                        help='where to start in the bag.')
    parser.add_argument('--end', '-e', action='store', default=1e10, type=float, help='where to stop in the bag.')
    parser.add_argument('bagfile')

    args = parser.parse_args()


    t0 = -1
    integration_time = 1.0 # in seconds
    w_avg = np.array([0, 0, 0])
    w_max = 0
    cnt  =0
    for bagfile in glob.glob(args.bagfile):
        bag = rosbag.Bag(bagfile, 'r')
        topics = ["/davis/right/imu"]
#       topics = ["/davis/left/imu"]
#        topics = ["/imu0"]
        iterator = bag.read_messages(topics=topics, start_time=args.start)
        for (topic, msg, time) in iterator:
            tstamp = msg.header.stamp.to_sec()
            if (t0 < 0):
                t0 = tstamp
            #print msg
            w = msg.angular_velocity
            w_vec = np.array([w.x, w.y, w.z])
            t = tstamp - t0
            if (t > args.end):
                break;
            w_len = np.linalg.norm(w_vec)
            if w_len > w_max:
                w_max = w_len
            w_avg = w_avg + w_vec
            cnt  = cnt + 1
#            print "%f %f %f %f %f" % (t, w_vec[0], w_vec[1], w_vec[2], w_len)

    print "avg ang velocity: "  + str(w_avg / cnt)
    print "max ang velocity: %f"  % (w_max)
    
                
                    
