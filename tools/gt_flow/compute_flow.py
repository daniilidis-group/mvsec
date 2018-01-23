""" Computes optical flow from two poses and depth images """

import numpy as np
import quaternion as quat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

    def compute_flow_single_frame(self, V, Omega, depth_image):
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

    def compute_velocity_from_msg(self, P0, P1):
        t0 = P0.header.stamp.to_sec()
        t1 = P1.header.stamp.to_sec()
        dt = t1 - t0
        p0 = np.array([P0.pose.position.x,P0.pose.position.y,P0.pose.position.z])
        p1 = np.array([P1.pose.position.x,P1.pose.position.y,P1.pose.position.z])

        q0 = quat.quaternion(P0.pose.orientation.w, P0.pose.orientation.x,
                             P0.pose.orientation.y, P0.pose.orientation.z)
        q1 = quat.quaternion(P1.pose.orientation.w, P1.pose.orientation.x,
                             P1.pose.orientation.y, P1.pose.orientation.z)

        return self.compute_velocity(p0, q0, p1, q1, dt)

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
        m = np.nanmax(self.hsv_buffer[:,:,2])
        if not np.isclose(m, 0.0):
            self.hsv_buffer[:,:,2] /= m
        return colors.hsv_to_rgb(self.hsv_buffer)

    def visualize_flow(self, flow_x, flow_y, fig):
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow( self.colorize_image(flow_x, flow_y) )


def experiment_flow(experiment_name, experiment_num, save_movie=True, save_numpy=True):
    import time
    import calibration
    cal = calibration.Calibration(experiment_name)
    import ground_truth
    gt = ground_truth.GroundTruth(experiment_name, experiment_num)

    flow = Flow(cal)
    P0 = None

    x_flow_list = []
    y_flow_list = []

    print "Computing depth"
    for frame_num in range(len(gt.left_cam_readers['/davis/left/depth_image_rect'])):
        depth_image = gt.left_cam_readers['/davis/left/depth_image_rect'][frame_num]
        depth_image.acquire()
        P1 = gt.left_cam_readers['/davis/left/odometry'][frame_num].message

        if P0 is not None:
            V, Omega = flow.compute_velocity_from_msg(P0, P1)
            flow_x, flow_y = flow.compute_flow_single_frame(V, Omega, depth_image.img)
            x_flow_list.append(flow_x)
            y_flow_list.append(flow_y)
        else:
            x_flow_list.append(np.zeros(depth_image.shape))
            y_flow_list.append(np.zeros(depth_image.shape))

        depth_image.release()
        P0 = P1

    import downloader
    import os
    base_name = os.path.join(downloader.get_tmp(), experiment_name, experiment_name+str(experiment_num))

    if save_numpy:
        print "Saving numpy"
        numpy_name = base_name+"flow.npz"
        np.savez(numpy_name, x_flow_list=x_flow_list, y_flow_list=y_flow_list)

    if save_movie:
        print "Saving movie"
        import matplotlib.animation as animation
        plt.close('all')
   
        fig = plt.figure()
        first_img = flow.colorize_image(x_flow_list[0], y_flow_list[0])
        im = plt.imshow(first_img, animated=True)
        
        def updatefig(frame_num, *args):
            im.set_data(flow.colorize_image(x_flow_list[frame_num], y_flow_list[frame_num]))
            return im,

        ani = animation.FuncAnimation(fig, updatefig, frames=len(x_flow_list))
        movie_path = base_name+"flow.mp4"
        ani.save(movie_path)
        plt.show()

def test_gt_flow():
    import calibration

    plt.close('all')

    cal = calibration.Calibration("indoor_flying")
    gtf = Flow(cal)
    
    p0 = np.array([0.,0.,0.])
    q0 = quat.quaternion(1.0,0.0,0.0,0.0)

    depth = 10.*np.ones((cal.left_map.shape[0],cal.left_map.shape[1]))

    V, Omega = gtf.compute_velocity(p0,q0,p0,q0,0.1)
    x,y = gtf.compute_flow_single_frame(V, Omega, depth)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)

    p1 = np.array([0.,1.,0.])
    q1 = quat.quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth)
    plt.figure()
    plt.imshow(x)
    plt.figure()
    plt.imshow(y)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)
