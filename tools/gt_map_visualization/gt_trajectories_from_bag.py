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


def pose_to_tf(msg):
    q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(q)
    T[0:3,3] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    return (T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract trajectories from bag file')
    parser.add_argument('--bag', '-b', action='store', default=None, required=True,
                        help='name of bag file')

    topics = ['/gps/fix',  '/davis/left/odometry', '/davis/left/pose']
              
    args = parser.parse_args()

    fileNames = ['gps_traj.txt', 'loam_traj.txt', 'cartographer_traj.txt']
    
    gpsFile = open(fileNames[0], 'w')
    loamFile = open(fileNames[1], 'w')
    cartographerFile = open(fileNames[2], 'w')

    for bagfile in [args.bag]:
        bag = rosbag.Bag(bagfile, 'r')
        it = bag.read_messages(topics=topics)
        for (topic, msg, tros) in it:
            if (topic == '/davis/left/odometry'):  # loam pose
                T = pose_to_tf(msg)
                loamFile.write("%.6f %s\n" % (msg.header.stamp.to_nsec()/1e9, ' '.join(map(str, T.ravel()[0:12].tolist()))))
            elif (topic == '/davis/left/pose'):  # cartographer pose
                T = pose_to_tf(msg)
                cartographerFile.write("%.6f %s\n" % (msg.header.stamp.to_nsec()/1e9, ' '.join(map(str, T.ravel()[0:12].tolist()))))
            elif (topic == '/gps/fix'):  # gps long/lat
                utm_pos = utm.from_latlon(msg.latitude, msg.longitude)
                a = msg.altitude # ignored here
                T = tf.transformations.identity_matrix()
                T[0:3,3] = np.array([utm_pos[0], utm_pos[1], 0])
                gpsFile.write("%.6f %s\n" % (msg.header.stamp.to_nsec()/1e9, ' '.join(map(str, T.ravel()[0:12].tolist()))))
                
    gpsFile.close()
    loamFile.close()
    cartographerFile.close()
    print "wrote to files: %s" % ", ".join(fileNames)
    
