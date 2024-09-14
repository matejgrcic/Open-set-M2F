import os
import argparse
from PIL import Image
from mask2former import add_maskformer2_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
import numpy as np
import torch
import glob
from tqdm import tqdm


def calculate_anomaly_scores(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # SWIN-L
    cfg.merge_from_file('./configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_150k_kplus2_evaluate.yaml')
    cfg.MODEL.WEIGHTS = './checkpoints/model_0001999_uno.pth'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True

    predictor = DefaultPredictor(cfg)
    images = glob.glob(f"{args.input_folder}/raw_data/**/*.{args.extension}", recursive=True)

    for image_name in tqdm(images):
        image = Image.open(image_name)
        image = np.asarray(image)
        # Convert RGB to BGR
        x = image[:, :, ::-1].copy()
        with torch.no_grad():
            output = predictor(x)
        mask_pred = output['mask_pred']
        mask_cls_ = output['mask_cls']
        mask_pred = mask_pred.sigmoid()
        mask_cls = mask_cls_.softmax(-1)[..., :-2]
        
        p_c_x = mask_cls_.softmax(-1)

        s_x = -p_c_x[..., -2] + mask_cls.max(1)[0]

        v = (mask_pred * s_x.view(-1, 1, 1)).sum(0)
        ood_score = - v

        out_path = image_name.replace('raw_data', 'ood_score_no_norm').replace(args.extension, 'npy').replace('_ood_score', '')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, ood_score.detach().cpu().numpy())
        


def run_uno(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # SWIN-L
    cfg.merge_from_file('./configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_150k_kplus2_evaluate.yaml')
    cfg.MODEL.WEIGHTS = './checkpoints/model_0001999_uno.pth'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True

    predictor = DefaultPredictor(cfg)
    images = glob.glob(f"{args.input_folder}/raw_data/**/*.{args.extension}", recursive=True)

    for image_name in tqdm(images):
        image = Image.open(image_name)
        image = np.asarray(image)
        # Convert RGB to BGR
        x = image[:, :, ::-1].copy()
        with torch.no_grad():
            output = predictor(x)

        # get road mask
        sem_seg = (output["sem_seg"].argmax(0).to("cpu")).numpy()
        road_mask = sem_seg == 0
        road_mask_path = image_name.replace('raw_data', 'street_masks').replace(args.extension, 'npy').replace('_raw_image', '')
        os.makedirs(os.path.dirname(road_mask_path), exist_ok=True)
        np.save(road_mask_path, road_mask)
        
        
        # get anomaly score
        mask_pred = output['mask_pred']
        mask_cls_ = output['mask_cls']
        mask_pred = mask_pred.sigmoid()
        mask_cls = mask_cls_.softmax(-1)[..., :-2]
        
        p_c_x = mask_cls_.softmax(-1)

        s_x = -p_c_x[..., -2] + mask_cls.max(1)[0]

        v = (mask_pred * s_x.view(-1, 1, 1)).sum(0)
        ood_score = - v

        anomaly_score_path = image_name.replace('raw_data', 'ood_score_no_norm').replace(args.extension, 'npy').replace('_ood_score', '')
        os.makedirs(os.path.dirname(anomaly_score_path), exist_ok=True)
        np.save(anomaly_score_path, ood_score.detach().cpu().numpy())
        


def normalize_anomaly_scores(args):
    scores = glob.glob(f"{args.input_folder}/ood_score_no_norm/**/*.npy", recursive=True)
    
    MIN = np.inf
    MAX = - np.inf
    for score_file in tqdm(scores):

        ood_score = np.load(score_file)
        
        if ood_score.min() < MIN:
            MIN = ood_score.min()
        if ood_score.max() > MAX:
            MAX = ood_score.max()

    print(MIN, MAX)


    for score_file in tqdm(scores):

        ood_score = np.load(score_file)
        
        s_norm = (ood_score-MIN)/(MAX-MIN)

        out_path = score_file.replace('ood_score_no_norm', 'ood_score')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, s_norm)


def copy_and_rename_files_in_subdir(new_video_dir):
    for root, dirs, files in tqdm(os.walk(new_video_dir)):
        for file in files:
            # Construct the full file path
            old_file_path = os.path.join(root, file)
            
            new_file_name = file.replace("_raw_data", "")
            new_file_path = os.path.join(root, new_file_name)
            
            os.rename(old_file_path, new_file_path)
            print(f"Copied and renamed: {old_file_path} -> {new_file_path}")



def copy_and_rename(args):
    new_video_dir = args.input_folder.replace('raw_data', 'raw_data_tmp')

    # We assume that there exists a dir named raw_data_tmp with content copied from raw_data
    if not os.path.exists(new_video_dir):
        # Try running the following line,
        # shutil.copytree(video_dir, new_video_dir, dirs_exist_ok=True)
        # or better, copy raw_files from terminal:
        # cp -r raw_data raw_data_tmp
        raise FileNotFoundError(f"The directory {args.input_folder} does not exist.")
    
    copy_and_rename_files_in_subdir(new_video_dir)