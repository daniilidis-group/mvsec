#!/usr/bin/env python
#
#  script to turn trajectory files into tracks file for upload to gps visualizer
#
#


import rosbag, rospy
import glob
import argparse
import time
import utm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert trajectories to gpsvisualizer format')
    parser.add_argument('--files', '-f', action='store', default=None, required=True,
                        help='comma-separated list of trajectory files')
    parser.add_argument('--legend', '-l', action='store', default=None, required=True,
                        help='comma-separated list of legend names')
    parser.add_argument('--colors', '-c', action='store', default=None, required=True,
                        help='comma-separated list of colors')
    parser.add_argument('--linewidth', '-w', action='store', default=2, required=False,
                        help='line width')
    args = parser.parse_args()

    
    legends = args.legend.split(",")
    colors  = args.colors.split(",")
    files   = args.files.split(",")
    width   = args.linewidth

    if (len(legends) != len(files)):
        print "error: number of files %d must match number of legend items %d" % (len(files), len(legends))
        exit(-1)
    if (len(colors) != len(files)):
        print "error: number of files %d must match number of colors %d" % (len(files), len(colors))
        exit(-1)

    zone_number = 18  # UTM zone number
    zone_letter = 'S' # UTM zone letter

    tpn = 0 # track point number
    for f,n,c in zip(files, legends, colors):
        print "%s,%s,%s,%s,%s,%s,%s,%s" % ("trackpoint", "time", "latitude",
                                        "longitude", "alt", "name", "color", "width")
        traj = np.loadtxt(f)
        for row in traj:
            t        =  row[0]
            easting  =  row[4]
            northing =  row[8]
            alt      = row[12]
            lat_long = utm.to_latlon(easting, northing, zone_number, zone_letter)
            tstr = time.strftime("%H:%M:%S", time.gmtime(t))
            print "T,%s,%.6f,%.6f,%.3f,%s,%s,%d" % (tstr, lat_long[0], lat_long[1], alt, n, c, width)
            tpn = tpn + 1
    
