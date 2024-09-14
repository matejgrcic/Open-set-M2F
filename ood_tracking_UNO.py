import os
import argparse

from uno_sam_tracker.preprocessing import run_uno, normalize_anomaly_scores

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['sos', 'wos'], type=str, default='sos', help='Dataset.')
parser.add_argument('--input_folder', type=str, default='/mnt/sdb1/datasets/street_obstacle_sequences', help='Path to folder with images to be run.')

args = parser.parse_args()

if args == 'wos':
    args.extension = 'png'
    args.threshold = 0.0965028

elif args == 'sos':
    args.extension = 'jpg'
    args.threshold = 0.52


args.raw_anomaly_score_folder = os.path.join(args.input_folder, 'ood_score_no_norm')
args.anomaly_score_folder = os.path.join(args.input_folder, 'ood_score')
args.video_dir = os.path.join(args.input_folder, 'raw_data_tmp')
args.res_dir = os.path.join(args.input_folder, 'ood_prediction_tracked')
args.res_video_dir = os.path.join(args.input_folder, 'result_videos')

# 1. Calculate anomaly scores & street masks
run_uno(args)

# 2. Normalize anomaly scores
normalize_anomaly_scores(args)

