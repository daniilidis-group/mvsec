""" Computes optical flow from two poses and depth images """

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
        self.fx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][0]
        self.px = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][2]
        self.fy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][1]
        self.py = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][2]

        # number of pixels in the camera
        x_map = (self.cal.left_map[:,:,0]-self.px)/self.fx
        y_map = (self.cal.left_map[:,:,1]-self.py)/self.fy
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
        mask = np.isfinite(flat_depth)

        fdm = flat_depth[mask]
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

        x_flow_out = (x_flow_out*self.fx)*dt
        y_flow_out = (y_flow_out*self.fx)*dt

        return x_flow_out, y_flow_out
    
    def rot_mat_from_quaternion(self, q):
        R = np.array([[1-2*q.y**2-2*q.z**2, 2*q.x*q.y+2*q.w*q.z, 2*q.x*q.z-2*q.w*q.y],
                      [2*q.x*q.y-2*q.w*q.z, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z+2*q.w*q.x],
                      [2*q.x*q.z+2*q.w*q.y, 2*q.y*q.z-2*q.w*q.x, 1-2*q.x**2-2*q.y**2]])
        return R

    def compute_velocity_from_msg(self, P0, P1):
        t0 = P0.header.stamp.to_sec()
        t1 = P1.header.stamp.to_sec()
        dt = t1 - t0
        p0 = np.array([P0.pose.position.x,P0.pose.position.y,P0.pose.position.z])
        q0 = quaternion(P0.pose.orientation.w, P0.pose.orientation.x,
                             P0.pose.orientation.y, P0.pose.orientation.z)

        p1 = np.array([P1.pose.position.x,P1.pose.position.y,P1.pose.position.z])
        q1 = quaternion(P1.pose.orientation.w, P1.pose.orientation.x,
                             P1.pose.orientation.y, P1.pose.orientation.z)

        # compute H0^-1
        inv = q0.inverse()
        p0 = np.dot(self.rot_mat_from_quaternion(q0),-p0)
        q0 = inv

        # set H1 to H0^-1 * H1
        p1 = p0 + np.dot(self.rot_mat_from_quaternion(q0),p1)
        q1 = q0 * q1

        # Set H0 to I
        p0 = np.zeros(p0.shape)
        q0 = quaternion(1.0, 0.0, 0.0, 0.0)

        V, Omega = self.compute_velocity(p0, q0, p1, q1, dt)
        return V, Omega, dt

    def compute_velocity(self, p0, q0, p1, q1, dt):
        V = (p1-p0)/dt

        dqdt = (q1-q0)/dt

        Omega = 2.*dqdt*q1.inverse()
        Omega = np.array([Omega.x, Omega.y, Omega.z])

        return V, Omega

    def colorize_image(self, flow_x, flow_y):
        if self.hsv_buffer is None:
            self.hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1],3))
            self.hsv_buffer[:,:,1] = 1.0
        self.hsv_buffer[:,:,0] = (np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

        self.hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

        flat = self.hsv_buffer[:,:,2].reshape((-1))
        flat[flat>20.] = 20.

        m = np.nanmax(self.hsv_buffer[:,:,2])
        if not np.isclose(m, 0.0):
            self.hsv_buffer[:,:,2] /= m
        return colors.hsv_to_rgb(self.hsv_buffer)

    def visualize_flow(self, flow_x, flow_y, fig):
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow( self.colorize_image(flow_x, flow_y) )


def experiment_flow(experiment_name, experiment_num, save_movie=True, save_numpy=True):
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

    nframes = len(gt.left_cam_readers['/davis/left/depth_image_rect'])

    depth_image, _ = gt.left_cam_readers['/davis/left/depth_image_rect'](0)
    flow_shape = (nframes, depth_image.shape[0], depth_image.shape[1])
    x_flow_tensor = np.zeros(flow_shape, dtype=np.float)
    y_flow_tensor = np.zeros(flow_shape, dtype=np.float)
    timestamps = np.zeros((nframes,), dtype=np.float)
    Vs = np.zeros((nframes,3))
    Omegas = np.zeros((nframes,3))

    sOmega = np.zeros((3,))
    sV = np.zeros((3,))
    alpha = 0.1

    print "Computing depth"
    for frame_num in range(len(gt.left_cam_readers['/davis/left/depth_image_rect'])):
        depth_image = gt.left_cam_readers['/davis/left/depth_image_rect'][frame_num]
        depth_image.acquire()
        P1 = gt.left_cam_readers['/davis/left/odometry'][frame_num].message

        if P0 is not None:
            V, Omega, dt = flow.compute_velocity_from_msg(P0, P1)

            sOmega = alpha * Omega + (1-alpha)*sOmega
            sV = alpha * V + (1-alpha)*sV

            flow_x, flow_y = flow.compute_flow_single_frame(sV, sOmega, depth_image.img, dt)
            x_flow_tensor[frame_num,:,:] = flow_x
            y_flow_tensor[frame_num,:,:] = flow_y
            timestamps[frame_num] = P1.header.stamp.to_sec()
            Vs[frame_num, :] = sV
            Omegas[frame_num, :] = sOmega
        else:
            timestamps[frame_num] = P1.header.stamp.to_sec()

        depth_image.release()
        P0 = P1

    import downloader
    import os
    base_name = os.path.join(downloader.get_tmp(), experiment_name, experiment_name+str(experiment_num))

    if save_numpy:
        print "Saving numpy"
        numpy_name = base_name+"_gt_flow.npz"
        np.savez(numpy_name, ts=timestamps, x_flow_tensor=x_flow_tensor, y_flow_tensor=y_flow_tensor, Vs=Vs, Omegas=Omegas)

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

    p1 = np.array([0.,1.,0.])
    q1 = quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)
    plt.figure()
    plt.imshow(x)
    plt.figure()
    plt.imshow(y)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)
