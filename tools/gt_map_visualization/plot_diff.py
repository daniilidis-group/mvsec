#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse

def add_to_plot(ax, t, x, col, lab, w):
    return ax.plot(t, x, color=col, marker='.', label=lab, linewidth=w)
    #return ax.plot(t, x, color=col, marker='.', linewidth=w)
    

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
    parser.add_argument('--fatstart', '-s', action='store', default=1e9, required=False,
                        help='start of fat line')
    parser.add_argument('--fatend', '-e', action='store', default=0, required=False,
                        help='end of fat line')
    parser.add_argument('--title', '-t', action='store', default=None, required=False,
                        help='title')
    args = parser.parse_args()

    
    legends = args.legend.split(",")
    colors  = args.colors.split(",")
    files   = args.files.split(",")
    width   = 2
    tfatstart = float(args.fatstart)
    tfatend   = float(args.fatend)
    fatwidth = width * 4

    if (len(legends) != len(files)):
        print "error: number of files %d must match number of legend items %d" % (len(files), len(legends))
        exit(-1)
    if (len(colors) != len(files)):
        print "error: number of files %d must match number of colors %d" % (len(files), len(colors))
        exit(-1)

    plt.rcParams.update({'font.size': 48})
    fig  = plt.figure(figsize=(40,50))
    ax   = fig.add_subplot(111)
    ax.set_aspect(17)
    ref  = np.loadtxt(files[0])
    t0   = ref[0,0]
    t    = ref[:,0]-t0
    
    for f,n,c,i in zip(files, legends, colors, range(0,len(files))):
        if i > 0:
            traj = np.loadtxt(f)
            xi = np.interp(ref[:,0], traj[:,0], traj[:,4])
            yi = np.interp(ref[:,0], traj[:,0], traj[:,8])
            d  = np.sqrt(np.square(xi-ref[:,4]) + np.square(yi - ref[:, 8]))
            imax = np.argmax(d)
            print "%s largest error: %f at index: %d, time: %f" % (n, d[imax], imax, t[imax])
            h  = add_to_plot(ax, t, d, c, n, width)
            h  = add_to_plot(ax, t[(t>= tfatstart) & (t<=tfatend)], d[(t>= tfatstart) & (t <=tfatend)], c, None, fatwidth)
    #        h  = add_to_plot(ax, t[t>= tfatend], d[t>= tfatend], c, None, width)

    axes = plt.gca()
    axes.set_ylim([args.ymin, args.ymax])
    axes.set_xlim([0, ref[-1,0]-t0])
    if args.legend:
        ax.legend(loc='upper left', shadow = True)
    if args.title:
        plt.title(args.title)
    plt.xlabel('time [s]')
    plt.ylabel('distance [m]')
    plt.show()
    pp = PdfPages('plot.pdf')
    pp.savefig(fig)
    pp.close()
