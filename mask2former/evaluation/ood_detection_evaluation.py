# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve, auc
import h5py

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator
from torchvision.utils import save_image
import torch.nn.functional as F
# from mask2former.utils import colorize_labels
import matplotlib.pyplot as plt

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


def calculate_stat(conf, gt):
    fpr, tpr, threshold = roc_curve(gt, conf)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    threshold = np.array(threshold)

    roc_auc = auc(fpr, tpr)
    fpr_best = fpr[tpr >= 0.95][0]
    tau = threshold[tpr >= 0.95][0]
    return roc_auc, fpr_best, tau

class DenseOODDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate OOD detection metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )



    def reset(self):
        self._gt = []
        self._ood_score = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        for input, output in zip(inputs, outputs):
            mask_pred = output['mask_pred']
            mask_cls_ = output['mask_cls']
            mask_pred = mask_pred.sigmoid()
            mask_cls = mask_cls_.softmax(-1)[..., :-1]

            max_p = mask_cls.max(1)[0]
            s_x = max_p

            v = (mask_pred * s_x.view(-1, 1, 1)).sum(0)
            ood_score = 1 - v.to(self._cpu_device)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int64)
            gt_vec = torch.from_numpy(gt).view(-1)



            if self._dataset_name == 'road_anomaly':
                gt_vec[gt_vec == 2] = 1

            self._ood_score += [ood_score.view(-1)[gt_vec != self._ignore_label]]
            self._gt += [gt_vec[gt_vec != self._ignore_label]]

    def evaluate(self):
        if self._distributed:
            raise Exception('Not implemented.')

        gt = torch.cat(self._gt, 0)
        score = torch.cat(self._ood_score, 0)

        AUROC, FPR, _ = calculate_stat(score, gt)
        AP = average_precision_score(gt, score)


        res = {}
        res["AP"] = 100 * AP
        res["AUROC"] = 100 * AUROC
        res["FPR@TPR95"] = 100 * FPR


        results = OrderedDict({"ood_detection": res})
        self._logger.info(results)
        return results
