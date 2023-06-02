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

def _get_ra_meta():
    stuff_classes = [k["readable"] for k in SMIYC_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in SMIYC_SEM_SEG_CATEGORIES if k["evaluate"]]
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_road_anomaly(root):

    image_files = list(sorted(glob.glob(f"{root}/frames/*.jpg")))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('.jpg', '.labels/labels_semantic.png'),
            "height": 720,
            "width": 1280,
        })
    return examples
def register_road_anomaly(root):
    root = os.path.join(root, "road_anomaly")
    meta = _get_ra_meta()

    DatasetCatalog.register(
        "road_anomaly", lambda x=root: load_road_anomaly(x)
    )

    MetadataCatalog.get("road_anomaly").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_road_anomaly(_root)
