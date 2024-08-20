Open-set Semantic Segmentation Built  atop Mask-level Recognition
==========================================================

This repository provides the official implementation of our recent papers:

### :scroll: UNO: Outlier detection by ensembling uncertainty with negative objectness

[Anja Delić](https://github.com/adelic99), [Matej Grcić](https://matejgrcic.github.io/), [Siniša Šegvić](https://www.zemris.fer.hr/~ssegvic/index_en.html)

Published in BMVC 2024

[[`arXiv`](https://arxiv.org/abs/2402.15374)]

### :scroll: **EAM**: On Advantages of Mask-level Recognition for Outlier-aware Segmentation

[Matej Grcić](https://matejgrcic.github.io/), [Josip Šarić](https://jsaric.github.io/), [Siniša Šegvić](https://www.zemris.fer.hr/~ssegvic/index_en.html)

Published in CVPR workshop (VAND) 2023

[[`arXiv`](https://arxiv.org/abs/2301.03407)]


## Installation

Similar to Mask2Former repo, see [installation instructions](INSTALL.md).

## Training

### UNO
Finetuning a model with the additional negative objectness class with ADE20K negatives:
```bash
python finetune_UNO.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_2k_city+vistas_uno.yaml --num-gpus 3
```

Finetuning a model with the additional negative objectness class with **synthetic** negatives:
```bash
python finetune_UNO_synthetic.py --config-file  configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_2k_city+vistas_uno_synthetic.yaml --num-gpus 3
```

### EAM
Rejecting predictions in negative instances:
```bash
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs12_2k_city+vistas_oe.yaml
```


## Evaluation

### UNO
```bash
python train_net.py --config-file  configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_115k_city+_vistas_uno.yaml --eval-only MODEL.WEIGHTS path_to_model DATASETS.TEST eval_dataset_name
```

### EAM
```bash
python train_net.py --config-file  configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs18_115k_city+vistas.yaml --eval-only MODEL.WEIGHTS path_to_model DATASETS.TEST eval_dataset_name
```
`eval_dataset_name` can be one of the following:
 `("fs_static_val", "fs_laf_val", "road_anomaly",) `

## Checkpoints

### EAM
Mask2Former with SWIN-L backbone trained on Cityscapes (CS): [weights](https://drive.google.com/file/d/1AaBePz8MQe3NBxMa768yTNuW2I8Tpyv1/view?usp=sharing)

Mask2Former with SWIN-L backbone trained on Cityscapes and Vistas (CS&MV): [weights](https://drive.google.com/file/d/1Ebgr9wc-UivGGiqMPNnYwm1LvGzA3YkR/view?usp=sharing)

Mask2Former with SWIN-L backbone (CS&MV) fine-tuned with ADE20k negatives: [weights](https://drive.google.com/file/d/1u5s10ZhYNR50M5lqW4bjriHdfDMe1xH-/view?usp=sharing)

### UNO
Mask2Former with SWIN-L backbone (CS&MV) with K+2 classes fine-tuned with ADE20k negatives: [weights](https://drive.google.com/file/d/1ablD-t34MXcP-oSSzSq0-TNz0AxKtp_m/view?usp=sharing)

Mask2Former with SWIN-L backbone (CS&MV) with K+2 classes fine-tuned with synthetic negatives: [weights](https://drive.google.com/file/d/108CHRZFWTnDBonQv2yRjRL3JNj4_y47E/view?usp=sharing) <br>
DenseFlow pretrained on CS&MV: [weights](https://drive.google.com/file/d/1vS7K2irT2Gxh_8UQ9Aw1X5t5l6tG0Eol/view?usp=sharing)




## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Citing Outlier-aware Mask2Former


```BibTeX
@inproceedings{delic24bmvc,
  title={Outlier detection by ensembling uncertainty with negative objectness},
  author={Anja Delić and Matej Grcic and Siniša Šegvić}
  journal={BMVC 2024 British Machine Vision Conference},
  year={2024}
}
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
