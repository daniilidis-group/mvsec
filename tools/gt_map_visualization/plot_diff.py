#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse

def add_to_plot(ax, t, x, col, l):
    return ax.plot(t, x, color=col, marker='.', label=l)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert trajectories to gpsvisualizer format')
    parser.add_argument('--files', '-f', action='store', default=None, required=True,
                        help='comma-separated list of trajectory files')
    parser.add_argument('--legend', '-l', action='store', default=None, required=True,
                        help='comma-separated list of legend names')
    parser.add_argument('--ymin', '-n', type=float, action='store', default=0, required=False,
                        help='min y value')
    parser.add_argument('--ymax', '-x', type=float, action='store', default=100, required=False,
                        help='max y value')
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
    ref  = np.loadtxt(files[0])
    t0   = ref[0,0]
    t    = ref[:,0]-t0

    for f,n,c,i in zip(files, legends, colors, range(0,len(files))):
        if i > 0:
            traj = np.loadtxt(f)
            xi = np.interp(ref[:,0], traj[:,0], traj[:,4])
            yi = np.interp(ref[:,0], traj[:,0], traj[:,8])
            d  = np.sqrt(np.square(xi-ref[:,4]) + np.square(yi - ref[:, 8]))
            h  = add_to_plot(ax, t, d, c, n)

    axes = plt.gca()
    axes.set_ylim([args.ymin, args.ymax])
    ax.legend(loc='upper center', shadow = True)
    plt.show()
