# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetMapperTraffic"]

from cityscapesscripts.helpers.labels import labels as _labels

import json
import os
import torch

_wd_to_city_mappings = [
( 0 , 255 ),
( 1 , 255 ),
( 2 ,  255),
( 3 , 255 ),
( 4 , 255 ),
( 5 , 255 ),
( 6 , 255 ),
( 7 , 0 ),
( 8 , 1 ),
( 9 , 255 ),
( 10 , 255 ),
( 11 , 2 ),
( 12 , 3 ),
( 13 , 4 ),
( 14 , 255 ),
( 15 , 255 ),
( 16 , 255 ),
( 17 , 5 ),
( 18 , 255 ),
( 19 , 6 ),
( 20 , 7 ),
( 21 , 8 ),
( 22 , 9 ),
( 23 , 10 ),
( 24 , 11 ),
( 25 , 12 ),
( 26 , 13 ),
( 27 , 14 ),
( 28 , 15 ),
( 29 , 255 ),
( 30 , 255 ),
( 31 , 16 ),
( 32 , 17 ),
( 33 , 18 ),
( 34 , 13 ),
( 35 , 13 ),
( 36 , 255 ),
( 37 , 255 ),
( 38 , 0 )
]

_vistas_to_cityscapes = {
    'construction--barrier--curb': 'sidewalk',
    'construction--barrier--fence': 'fence',
    'construction--barrier--guard-rail': 'fence',
    'construction--barrier--wall': 'wall',
    'construction--flat--bike-lane': 'road',
    'construction--flat--crosswalk-plain': 'road',
    'construction--flat--curb-cut': 'sidewalk',
    'construction--flat--parking': 'road',
    'construction--flat--pedestrian-area': 'sidewalk',
    'construction--flat--rail-track': 'road',
    'construction--flat--road': 'road',
    'construction--flat--service-lane': 'road',
    'construction--flat--sidewalk': 'sidewalk',
    'construction--structure--bridge': 'building',
    'construction--structure--building': 'building',
    'construction--structure--tunnel': 'building',
    'human--person': 'person',
    'human--rider--bicyclist': 'rider',
    'human--rider--motorcyclist': 'rider',
    'human--rider--other-rider': 'rider',
    'marking--crosswalk-zebra': 'road',
    'marking--general': 'road',
    'nature--sand': 'terrain',
    'nature--sky': 'sky',
    'nature--snow': 'terrain',
    'nature--terrain': 'terrain',
    'nature--vegetation': 'vegetation',
    'object--support--pole': 'pole',
    'object--support--traffic-sign-frame': 'traffic sign',
    'object--support--utility-pole': 'pole',
    'object--traffic-light': 'traffic light',
    'object--traffic-sign--front': 'traffic sign',
    'object--vehicle--bicycle': 'bicycle',
    'object--vehicle--bus': 'bus',
    'object--vehicle--car': 'car',
    'object--vehicle--motorcycle': 'motorcycle',
    'object--vehicle--on-rails': 'train',
    'object--vehicle--truck': 'truck',
}

def create_id_to_name():
    id_to_name = {}
    for lbl in _labels:
        if lbl[2] != 255:
            id_to_name[lbl[2]] = lbl[0]
    del id_to_name[-1]
    id_to_name[255] = 'ignore'
    return id_to_name

def create_name_to_id():
    name_to_id = {}
    for lbl in _labels:
        if lbl[2] != 255:
            name_to_id[lbl[0]] = lbl[2]
    del name_to_id['license plate']
    name_to_id['ignore'] = 255
    return name_to_id

cityscapes_name_to_id = create_name_to_id()
cityscapes_id_to_name = create_id_to_name()
cs_ignore_class = 255
n_classes = 66


def _parse_config(config_path):
    # read in config file
    with open(config_path) as config_file:
        config = json.load(config_file)

    labels = config['labels']

    class_names = []
    class_ids = []
    class_colors = []
    id_to_name = {}
    print("> There are {} labels in the config file".format(len(labels)))
    for label_id, label in enumerate(labels):
        class_names.append(label["readable"])
        class_ids.append(label_id)
        class_colors.append(label["color"])
        id_to_name[label_id] = label["name"]
    return class_names, class_ids, class_colors, id_to_name, labels


def _to_cityscapes_class(id, vistas_id_to_name):
    vistas_name = vistas_id_to_name[id]
    cityscapes_name = _vistas_to_cityscapes.get(vistas_name)
    if cityscapes_name == None:
        return cityscapes_name_to_id['ignore']
    else:
        return cityscapes_name_to_id[cityscapes_name]

def create_vistas_to_cityscapes_mapper(root):
    class_names, class_ids, class_colors, vistas_id_to_name, labels = _parse_config(os.path.join(root, 'config.json'))
    cityscapes_classes_mapper = torch.zeros(n_classes).long().fill_(cs_ignore_class)
    for i in range(len(labels)):
        cityscapes_classes_mapper[i] = _to_cityscapes_class(i, vistas_id_to_name)
    return cityscapes_classes_mapper

def create_wilddash_to_cityscapes_mapper():
    mapper = torch.ones(len(_wd_to_city_mappings)).long()
    for widl_class, city_class in _wd_to_city_mappings:
        mapper[widl_class] = city_class
    return mapper

class MaskFormerSemanticDatasetMapperTraffic:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        self.vistas_mapper = create_vistas_to_cityscapes_mapper('./datasets/mapillary-vistas')

        self.wilddash_mapper = create_wilddash_to_cityscapes_mapper()

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        sem_seg_file_name = dataset_dict["sem_seg_file_name"] if "sem_seg_file_name" in dataset_dict else None
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if 'vistas' in sem_seg_file_name:
            sem_seg_gt = self.vistas_mapper[sem_seg_gt.long()]
            dataset_dict["sem_seg"] = sem_seg_gt
        elif 'wilddash' in sem_seg_file_name:
            sem_seg_gt = self.wilddash_mapper[sem_seg_gt.long()]
            dataset_dict["sem_seg"] = sem_seg_gt


        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
