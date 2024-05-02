# On Advantages of Mask-level Recognition for Outlier-aware Segmentation


[Matej Grcić]() [Josip Šarić]() [Siniša Šegvić]()

[[`arXiv`](https://arxiv.org/abs/2301.03407)]

## Features
* Extension of Mask2Former for outlier-aware scenarios
* Supports outlier-aware semantic segmentation

**TODO:**
* Support for outlier-aware panoptic segmentation

## Installation

Similar to Mask2Former repo, see [installation instructions](INSTALL.md).

## Evaluation

Currently supports evaluation on RoadAnomaly and Fishyscapes-val

```bash
python train_net.py --config-file  configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_115k_city+vistas.yaml --eval-only MODEL.WEIGHTS path_to_model DATASETS.TEST eval_dataset_name
```
eval_dataset_name can be one of the following:
- ("fs_static_val", "fs_laf_val", "road_anomaly",)

## Pretrained Models

Mask2Former with SWIN-L backbone fine-tuned with ADE20k negatives: [weights](https://drive.google.com/file/d/1u5s10ZhYNR50M5lqW4bjriHdfDMe1xH-/view?usp=sharing)


## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Citing Outlier-aware Mask2Former


```BibTeX
@inproceedings{grcic23cvprw,
  title={On Advantages of Mask-level Recognition for Outlier-aware Segmentation},
  author={Matej Grcic and Josip Šarić and Siniša Šegvić}
  journal={CVPR 2023 Workshop Visual Anomaly and Novelty Detection (VAND)},
  year={2023}
}
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

## Acknowledgement

Code is extension of Mask2Former (https://github.com/facebookresearch/Mask2Former).
