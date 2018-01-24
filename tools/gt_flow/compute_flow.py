""" Computes optical flow from two poses and depth images """

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import logm
try:
    from quaternion import quaternion
except ImportError:
    class quaternion:
        def __init__(self,w,x,y,z):
            self.w = w
            self.x = x
            self.y = y
            self.z = z
    
        def norm(self):
            return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
        def inverse(self):
            qnorm = self.norm()
            return quaternion(self.w/qnorm,
                              -self.x/qnorm,
                              -self.y/qnorm,
                              -self.z/qnorm)
    
        def __mul__(q1, q2):
            r = quaternion(q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
                           q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
                           q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
                           q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w)
            return r
    
        def __rmul__(q1, s):
            return quaternion(q1.w*s, q1.x*s, q1.y*s, q1.z*s)
    
        def __sub__(q1, q2):
            r = quaternion(q1.w-q2.w,
                           q1.x-q2.x,
                           q1.y-q2.y,
                           q1.z-q2.z)
            return r
    
        def __div__(q1, s):
            return quaternion(q1.w/s, q1.x/s, q1.y/s, q1.z/s)

class Flow:
    """
    - parameters
        - calibration :: a Calibration object from calibration.py
    """
    def __init__(self, calibration):
        self.cal = calibration
        self.Pfx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][0]
        self.Ppx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][2]
        self.Pfy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][1]
        self.Ppy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][2]

        self.Kfx = self.cal.intrinsic_extrinsic['cam0']['intrinsics'][0]
        self.Kfy = self.cal.intrinsic_extrinsic['cam0']['intrinsics'][1]

        # number of pixels in the camera
        x_map = (self.cal.left_map[:,:,0]-self.Ppx)/self.Pfx
        y_map = (self.cal.left_map[:,:,1]-self.Ppy)/self.Pfy
        self.flat_x_map = x_map.ravel()
        self.flat_y_map = y_map.ravel()

        N = self.flat_x_map.shape[0]

        self.omega_mat = np.zeros((N,2,3))

        self.omega_mat[:,0,0] = self.flat_x_map * self.flat_y_map
        self.omega_mat[:,1,0] = 1+ np.square(self.flat_y_map)

        self.omega_mat[:,0,1] = -(1+np.square(self.flat_x_map))
        self.omega_mat[:,1,1] = -(self.flat_x_map*self.flat_y_map)

        self.omega_mat[:,0,2] = self.flat_y_map
        self.omega_mat[:,1,2] = -self.flat_x_map

        self.hsv_buffer = None

    def compute_flow_single_frame(self, V, Omega, depth_image, dt):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            depth_image : [m,n]
        """
        flat_depth = depth_image.ravel()
        flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]
        mask = np.isfinite(flat_depth)

        fdm = 1./flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask,:,:]

        x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_x_flow_out = x_flow_out.reshape((-1))
        flat_x_flow_out[mask] = fdm * (fxm*V[2]-V[0])
        flat_x_flow_out[mask] +=  np.squeeze(np.dot(omm[:,0,:], Omega))

        y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_y_flow_out = y_flow_out.reshape((-1))
        flat_y_flow_out[mask] = fdm * (fym*V[2]-V[1])
        flat_y_flow_out[mask] +=  np.squeeze(np.dot(omm[:,1,:], Omega))

        x_flow_out = (x_flow_out*self.Kfx)*dt
        y_flow_out = (y_flow_out*self.Kfx)*dt

        return x_flow_out, y_flow_out
    
    def rot_mat_from_quaternion(self, q):
        R = np.array([[1-2*q.y**2-2*q.z**2, 2*q.x*q.y+2*q.w*q.z, 2*q.x*q.z-2*q.w*q.y],
                      [2*q.x*q.y-2*q.w*q.z, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z+2*q.w*q.x],
                      [2*q.x*q.z+2*q.w*q.y, 2*q.y*q.z-2*q.w*q.x, 1-2*q.x**2-2*q.y**2]])
        return R

    def p_q_t_from_msg(self, msg):
        p = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        q = quaternion(msg.pose.orientation.w, msg.pose.orientation.x,
                            msg.pose.orientation.y, msg.pose.orientation.z)
        t = msg.header.stamp.to_sec()
        return p, q, t

    def compute_velocity_from_msg(self, P0, P1):
        p0, q0, t0 = self.p_q_t_from_msg(P0)
        p1, q1, t1 = self.p_q_t_from_msg(P1)

        # Not sure why we need to transpose
        R0 = self.rot_mat_from_quaternion(q0).T
        R1 = self.rot_mat_from_quaternion(q1).T

        dt = t1 - t0

        # compute H0^-1
        H0_inv_p = np.dot(R0,-p0)
        H0_inv_R = R0.T

        # set H1 to H0^-1 * H1
        H1_p = np.dot(H0_inv_R, p1-p0)
        H1_R = np.dot(H0_inv_R, R1)

        V = H1_p/dt

        w_hat = logm(np.dot(R0.T, R1)) / dt
        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega, dt

    def compute_velocity(self, p0, q0, p1, q1, dt):
        V = (p1-p0)/dt

        R_dot = ( self.rot_mat_from_quaternion(q1) - self.rot_mat_from_quaternion(q0) )/dt
        w_hat = np.dot(R_dot, self.rot_mat_from_quaternion(q1).T)

        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega

    def colorize_image(self, flow_x, flow_y):
        if self.hsv_buffer is None:
            self.hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1],3))
            self.hsv_buffer[:,:,1] = 1.0
        self.hsv_buffer[:,:,0] = (np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

        self.hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

        self.hsv_buffer[:,:,2] = np.log(1.+self.hsv_buffer[:,:,2]) # hopefully better overall dynamic range in final video

        flat = self.hsv_buffer[:,:,2].reshape((-1))
        m = np.nanmax(flat[np.isfinite(flat)])
        if not np.isclose(m, 0.0):
            self.hsv_buffer[:,:,2] /= m

        return colors.hsv_to_rgb(self.hsv_buffer)

    def visualize_flow(self, flow_x, flow_y, fig):
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow( self.colorize_image(flow_x, flow_y) )


def experiment_flow(experiment_name, experiment_num, save_movie=True, save_numpy=True, start_ind=None, stop_ind=None):
    if experiment_name == "motorcycle":
        print "The motorcycle doesn't have lidar and we can't compute flow without it"
        return

    import time
    import calibration
    cal = calibration.Calibration(experiment_name)
    import ground_truth
    gt = ground_truth.GroundTruth(experiment_name, experiment_num)

    flow = Flow(cal)
    P0 = None

    nframes = len(gt.left_cam_readers['/davis/left/depth_image_raw'])
    if stop_ind is not None:
        stop_ind = min(nframes, stop_ind)
    else:
        stop_ind = nframes

    if start_ind is not None:
        start_ind = max(0, start_ind)
    else:
        start_ind = 0

    nframes = stop_ind - start_ind


    depth_image, _ = gt.left_cam_readers['/davis/left/depth_image_raw'](0)
    flow_shape = (nframes, depth_image.shape[0], depth_image.shape[1])
    x_flow_tensor = np.zeros(flow_shape, dtype=np.float)
    y_flow_tensor = np.zeros(flow_shape, dtype=np.float)
    timestamps = np.zeros((nframes,), dtype=np.float)
    Vs = np.zeros((nframes,3), dtype=np.float)
    Omegas = np.zeros((nframes,3), dtype=np.float)
    dTs = np.zeros((nframes,), dtype=np.float)

    ps = np.zeros((nframes,3), dtype=np.float)
    qs = np.zeros((nframes,4), dtype=np.float)

    sOmega = np.zeros((3,))
    sV = np.zeros((3,))

    print "Extracting velocity"
    for frame_num in range(nframes):
        P1 = gt.left_cam_readers['/davis/left/odometry'][frame_num+start_ind].message

        if P0 is not None:
            V, Omega, dt = flow.compute_velocity_from_msg(P0, P1)

            Vs[frame_num, :] = V
            Omegas[frame_num, :] = Omega
            dTs[frame_num] = dt

        timestamps[frame_num] = P1.header.stamp.to_sec()

        tmp_p, tmp_q, _ = flow.p_q_t_from_msg(P1)
        ps[frame_num, :] = tmp_p
        qs[frame_num, 0] = tmp_q.w
        qs[frame_num, 0] = tmp_q.x
        qs[frame_num, 0] = tmp_q.y
        qs[frame_num, 0] = tmp_q.z

        P0 = P1

    filter_size = 10

    print "Computing flow"
    for frame_num in range(nframes):
        depth_image = gt.left_cam_readers['/davis/left/depth_image_raw'][frame_num+start_ind]
        depth_image.acquire()

        if frame_num-filter_size < 0:
            V = np.mean(Vs[0:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[0:frame_num+filter_size+1,:], axis=0)
        elif frame_num+filter_size >= nframes:
            V = np.mean(Vs[frame_num-filter_size:nframes,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:nframes,:], axis=0)
        else:
            V = np.mean(Vs[frame_num-filter_size:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:frame_num+filter_size+1,:], axis=0)
        dt = dTs[frame_num]

        flow_x, flow_y = flow.compute_flow_single_frame(V, Omega, depth_image.img, dt)
        x_flow_tensor[frame_num,:,:] = flow_x
        y_flow_tensor[frame_num,:,:] = flow_y

        depth_image.release()

    import downloader
    import os
    base_name = os.path.join(downloader.get_tmp(), experiment_name, experiment_name+str(experiment_num))

    if save_numpy:
        print "Saving numpy"
        numpy_name = base_name+"_gt_flow.npz"
        np.savez(numpy_name, ts=timestamps, x_flow_tensor=x_flow_tensor, y_flow_tensor=y_flow_tensor,
                             Vs=Vs, Omegas=Omegas, ps=ps, qs=qs)

    if save_movie:
        print "Saving movie"
        import matplotlib.animation as animation
        plt.close('all')
   
        fig = plt.figure()
        first_img = flow.colorize_image(x_flow_tensor[0], y_flow_tensor[0])
        im = plt.imshow(first_img, animated=True)
        
        def updatefig(frame_num, *args):
            im.set_data(flow.colorize_image(x_flow_tensor[frame_num], y_flow_tensor[frame_num]))
            return im,

        ani = animation.FuncAnimation(fig, updatefig, frames=len(x_flow_tensor))
        movie_path = base_name+"_gt_flow.mp4"
        ani.save(movie_path)
        plt.show()

    return x_flow_tensor, y_flow_tensor, timestamps, Vs, Omegas

def test_gt_flow():
    import calibration

    plt.close('all')

    cal = calibration.Calibration("indoor_flying")
    gtf = Flow(cal)
    
    p0 = np.array([0.,0.,0.])
    q0 = quaternion(1.0,0.0,0.0,0.0)

    depth = 10.*np.ones((cal.left_map.shape[0],cal.left_map.shape[1]))

    V, Omega = gtf.compute_velocity(p0,q0,p0,q0,0.1)
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)

    p1 = np.array([0.,0.25,0.5])
    q1 = quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)

    p1 = np.array([0.,-0.25,0.5])
    q1 = quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)
