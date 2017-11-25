#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse

def add_to_plot(ax, traj, col, l):
    return ax.plot(traj[:,4], traj[:,8], color=col, marker='.', label=l)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert trajectories to gpsvisualizer format')
    parser.add_argument('--files', '-f', action='store', default=None, required=True,
                        help='comma-separated list of trajectory files')
    parser.add_argument('--legend', '-l', action='store', default=None, required=True,
                        help='comma-separated list of legend names')
    parser.add_argument('--colors', '-c', action='store', default=None, required=True,
                        help='comma-separated list of colors')
    args = parser.parse_args()

    
    legends = args.legend.split(",")
    colors  = args.colors.split(",")
    files   = args.files.split(",")

    if (len(legends) != len(files)):
        print "error: number of files %d must match number of legend items %d" % (len(files), len(legends))
        exit(-1)
    if (len(colors) != len(files)):
        print "error: number of files %d must match number of colors %d" % (len(files), len(colors))
        exit(-1)

    fig  = plt.figure()
    ax   = fig.add_subplot(111)

    for f,n,c in zip(files, legends, colors):
        traj = np.loadtxt(f)
        h = add_to_plot(ax, traj, c, n)

    ax.legend(loc='upper center', shadow = True)
    plt.axis('equal')
    plt.show()
