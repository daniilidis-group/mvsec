import compute_flow; reload(compute_flow)

import downloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mvsec_dir',
                    type=str,
                    help="Path to MVSEC directory.",
                    required=True)
parser.add_argument('--sequence',
                    type=str,
                    help="Specific sequence group to extract, e.g. indoor_flying.",
                    required=True)
parser.add_argument('--sequence_number',
                    type=int,
                    help="Specific sequence within the group to extract, e.g. 1.",
                    required=True)
parser.add_argument('--save_movie',
                    action='store_true',
                    help="If set, will save a movie of the estimated flow for visualization.")
parser.add_argument('--save_numpy',
                    action='store_true',
                    help="If set, will save the results to a numpy file.")
parser.add_argument('--start_ind',
                    type=int,
                    help="Index of the first ground truth pose/depth frame to process.",
                    default=None)
parser.add_argument('--stop_ind',
                    type=int,
                    help="Index of the last ground truth pose/depth frame to process.",
                    default=None)
args = parser.parse_args()

downloader.set_tmp(args.mvsec_dir)

compute_flow.experiment_flow(args.sequence, 
                             args.sequence_number, 
                             save_movie=args.save_movie, 
                             save_numpy=args.save_numpy, 
                             start_ind=args.start_ind, 
                             stop_ind=args.stop_ind)
