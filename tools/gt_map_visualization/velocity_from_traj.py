#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse

def grad(y, x):
    dy = np.diff(y)
    dx = np.diff(x)
    dydx = dy / dx
    return np.hstack([dydx, 0])
    
def moving_average(a, n=3) :
    ret  = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute velocity from trajectory')
    parser.add_argument('--avgbins', '-a', type=int, action='store', default=5, required=False,
                        help='number of bins over which to average velocity')
    parser.add_argument('trajectory_file')
    args = parser.parse_args()

    a = int(args.avgbins)

    traj = np.loadtxt(args.trajectory_file)

    x = traj[:,4]
    y = traj[:,8]
    
    t = traj[:, 0] - traj[0, 0]
    vx = grad(x, t)
    vy = grad(y, t)

    
    vxa   = np.hstack([np.zeros(a-1), moving_average(vx,a)])
    vya   = np.hstack([np.zeros(a-1), moving_average(vy,a)])
    vabs  = np.sqrt(np.square(vx) + np.square(vy))
    vabsa = np.sqrt(np.square(vxa) + np.square(vya))

    a = np.transpose(np.vstack([t, vabs, vabsa]))

    imax = np.argmax(vabsa)
    print "max avg velocity: %f m/s = %f km/h = %f mph at time: %f" % (vabsa[imax], vabsa[imax]*3.6, vabsa[imax]*2.23694, t[imax])

#    for row in a:
#        print "%f  %f %f" % (row[0], row[1], row[2])
