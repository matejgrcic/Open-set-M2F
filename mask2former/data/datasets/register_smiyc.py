# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

SMIYC_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 0, 0],
        "instances": True,
        "readable": "Inliers",
        "name": "inliers",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": True,
        "readable": "Outlier",
        "name": "outlier",
        "evaluate": True,
    }
]

def _get_smiyc_meta():
    stuff_classes = [k["readable"] for k in SMIYC_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in SMIYC_SEM_SEG_CATEGORIES if k["evaluate"]]
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_smiyc_anomaly_track(root, name='validation'):

    image_files = list(sorted(glob.glob(f"{root}/dataset_AnomalyTrack/images/{name}*.jpg")))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'labels_masks').replace('.jpg', '_labels_semantic.png'),
            "height": 720,
            "width": 1280,
        })
    return examples

def load_test_smiyc_anomaly_track(root):
    from PIL import Image
    import numpy as np

    image_files = list(sorted(glob.glob(f"{root}/dataset_AnomalyTrack/images/*.jpg")))
    examples = []

    for im_file in image_files:
        W, H = Image.open(im_file).size
        fake_lbl = f"./tmp/{H}_{W}.jpg"
        Image.fromarray(np.zeros((H, W))).convert('L').save(fake_lbl)
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": fake_lbl,
            "height": H,
            "width": W,
        })
    return examples

def load_lostandfound(root, split='test'):
    assert split in ['train', 'test']
    image_files = list(sorted(glob.glob(f"{root}/dataset_LostAndFound/leftImg8bit/{split}/*/*.png")))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('_leftImg8bit.png', '_gtCoarse_labelTrainIds.png').replace('leftImg8bit', 'gtCoarse'),
            "height": 1024,
            "width": 2048,
        })
    return examples


def load_smiyc_obstacle_track(root, name='validation'):

    image_files = list(sorted(glob.glob(f"{root}/dataset_ObstacleTrack/images_jpg/{name}*.JPEG")))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images_jpg', 'labels_masks').replace('.JPEG', '_labels_semantic.png'),
            "height": 1080,
            "width": 1920,
        })
    return examples

def load_test_smiyc_obstacle_track(root):

    image_files = list(sorted(glob.glob(f"{root}/dataset_ObstacleTrack/images_jpg/*.JPEG")))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": f"{root}/dataset_ObstacleTrack/labels_masks/validation_18_labels_semantic.png",
            "height": 1080,
            "width": 1920,
        })
    return examples

def register_all_smiyc(root):
    root = os.path.join(root, "smiyc")
    meta = _get_smiyc_meta()

    DatasetCatalog.register(
        "smiyc_anomalytrack_val", lambda x=root: load_smiyc_anomaly_track(x)
    )

    DatasetCatalog.register(
        "smiyc_anomalytrack_test", lambda x=root: load_test_smiyc_anomaly_track(x)
    )

    DatasetCatalog.register(
        "smiyc_obstacletrack_val", lambda x=root: load_smiyc_obstacle_track(x)
    )

    DatasetCatalog.register(
        "smiyc_obstacletrack_test", lambda x=root: load_test_smiyc_obstacle_track(x)
    )

    DatasetCatalog.register(
        "laf_test", lambda x=root: load_lostandfound(x, 'test')
    )

    MetadataCatalog.get("smiyc_anomalytrack_val").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )

    MetadataCatalog.get("smiyc_anomalytrack_test").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )

    MetadataCatalog.get("smiyc_obstacletrack_val").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )

    MetadataCatalog.get("smiyc_obstacletrack_test").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )

    MetadataCatalog.get("laf_test").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_smiyc(_root)
