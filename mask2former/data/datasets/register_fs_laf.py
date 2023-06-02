# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FS_LAF_SEM_SEG_CATEGORIES = [
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

def _get_fs_laf_meta():
    stuff_classes = [k["readable"] for k in FS_LAF_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in FS_LAF_SEM_SEG_CATEGORIES if k["evaluate"]]
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_fs_laf_val(root):
    image_files = list(sorted(glob.glob(root + '/validation/leftImg8bit/*.png')))
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('_leftImg8bit.png', '_labels.png').replace('leftImg8bit', 'gtFine'),
            "height": 1024,
            "width": 2048,
        })
    return examples


def register_all_fs_laf(root):
    root = os.path.join(root, "fs_lost_found")
    meta = _get_fs_laf_meta()

    DatasetCatalog.register(
        "fs_laf_val", lambda x=root: load_fs_laf_val(x)
    )
    MetadataCatalog.get("fs_laf_val").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_fs_laf(_root)
