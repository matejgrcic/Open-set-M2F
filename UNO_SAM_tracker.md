UNO-SAM Out-of-Distribution Tracker
==========================================================

Our UNO-SAM OOD detector relies on [UNO](https://arxiv.org/abs/2402.15374) for pixel-level OOD detection and then tracks detected OOD objects by leveraging the [SAM 2](https://arxiv.org/abs/2408.00714) model.


### A brief description of our algorithm:

1. OOD segmentation: use UNO and threshold at TPR=95%.

2. Instance detection: split segmentation masks using connected components.

3. SAM 2 tracking: prompt SAM 2 and propagate over the video.

4. Post-process refinement



### Run the code

Prerequisits:
- Install Mask2Former. More details [here](README.md).

- Install SAM 2 and download weights. More details [here](https://github.com/facebookresearch/segment-anything-2).

- Download UNO [weights](https://drive.google.com/file/d/1ablD-t34MXcP-oSSzSq0-TNz0AxKtp_m/view?usp=sharing).

- Download SOS and WOS datasets.

- Copy & rename the input images since SAM 2 functions require a specific naming convention:
```bash
cd uno_sam_tracker; python copy_and_rename.py
```

- Adjust paths in ood_tracking_*.py sctipts

OOD tracking:

1. Run UNO

```bash
python ood_tracking_UNO.py
```

2. Run SAM 2
```bash
python ood_tracking_SAM2_SOS.py
```
```bash
python ood_tracking_SAM2_WOS.py
```

