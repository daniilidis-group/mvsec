
# Creating track maps from the ground truth bags

Here is how to create visualizations of the outdoor sequences from the ground truth ROS bags:

1. Create trajectories in UTM coordinate frame from bag file, in this case the first day time outdoor sequence:

    	./gt_trajectories_from_bag.py --bag /big_data/mvsec/west_philly_day1_gt.bag

   This will create the trajectory files `gps_traj.txt`, `loam_traj.txt`, and `cartographer_traj.txt`.

2. Align loam and cartographer trajectories with the gps trajectory, using the "align_trajectory" code
   from the [PennCOSYVIO data set](https://github.com/daniilidis-group/penncosyvio/tree/master/tools/cpp).

    	./align_trajectory  -r gps_traj.txt -t loam_traj.txt -o loam_traj_align.txt
        ./align_trajectory  -r gps_traj.txt -t cartographer_traj.txt -o cartographer_traj_align.txt

   The alignment is done using Horn's method, i.e. rotating and translating the data, but no scaling is applied


3. Now convert trajectories to file format suitable for GPS visualizer:

    	./traj_to_gpsvisualizer.py  --files gps_traj.txt,loam_traj_align.txt,cartographer_traj_align.txt --legend GPS,LOAM,Cartographer --colors cyan,blue,red > gps_visualizer.txt

   This file can be directly used for upload at [GPS Visualizer](http://www.gpsvisualizer.com/).

