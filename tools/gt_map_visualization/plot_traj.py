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
    parser.add_argument('--tstart', '-t', action='store', default=0, required=False,
                        help='start time')
    parser.add_argument('--tend', '-T', action='store', default=1e9, required=False,
                        help='end time')
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

    tstart = float(args.tstart)
    tend   = float(args.tend)
    for f,n,c in zip(files, legends, colors):
        traj = np.loadtxt(f)
        t  = traj[:,0] - traj[0,0]
        h = add_to_plot(ax, traj[(t >= tstart) & (t <= tend),:], c, n)

    ax.legend(loc='upper center', shadow = True)
    plt.axis('equal')
    plt.show()
