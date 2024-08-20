# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_semantic_dataset_mapper_traffic_oe import MaskFormerSemanticDatasetMapperTrafficWithOE
from .data.dataset_mappers.mask_former_semantic_dataset_mapper_traffic_uno import MaskFormerSemanticDatasetMapperWithUNO
from .data.dataset_mappers.mask_former_semantic_dataset_mapper_traffic import MaskFormerSemanticDatasetMapperTraffic

from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .maskformer_model_joint_flow import MaskFormerJointFlow
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.ood_detection_evaluation import DenseOODDetectionEvaluator
from .evaluation.ood_detection_evaluation_UNO import DenseOODDetectionEvaluatorUNO