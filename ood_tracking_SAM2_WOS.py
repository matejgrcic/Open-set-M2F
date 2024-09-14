import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
import pickle
from collections import Counter, defaultdict
import random
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from scipy.ndimage import measurements


DATASET = 'wos' # or 'wos'


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def connected_components(binary_mask, closed_road):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    masks_filtered, stats_filtered, centroids_filtered = [], [], []
    for i in range(num_labels):
        if i == 0:
            continue # background
        if stats[i, 4] < 500:
            continue
        mask = labels_im==i
        if np.abs(np.sum(np.logical_and(mask, closed_road)) - np.sum(mask)) > 100:
            continue
        masks_filtered.append(mask)
        stats_filtered.append(stats[i])
        centroids_filtered.append(np.expand_dims(centroids[i], 0))
    return len(masks_filtered), masks_filtered, stats_filtered, centroids_filtered

colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (128, 0, 0),      # Maroon
    (128, 128, 0),    # Olive
    (0, 128, 0),      # Dark Green
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (0, 0, 128),      # Navy
    (255, 165, 0),    # Orange
    (255, 192, 203),  # Pink
    (255, 215, 0),    # Gold
    (173, 216, 230),  # Light Blue
    (50, 205, 50),    # Lime Green
    (139, 69, 19)  ,   # Brown
    (0, 0, 0)
]

def generate_random_rgb_colors(n):
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
    return colors

# Generate 100 random RGB colors
colors = generate_random_rgb_colors(100)
colors.append((0, 0, 0))

def colorize_labels(labels):
    # Create an empty RGB image
    colorized_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    
    # Ensure there are enough colors for the number of labels
    unique_labels = np.unique(labels)
    # if len(unique_labels) > len(colors):
    #     raise ValueError("Not enough colors to represent all labels")
    
    colorized_image[labels == -1] = colors[-1]
    # Assign each unique label a color from the list
    for label in unique_labels:
        
        if label == -1: continue
        # colorized_image[labels == 255] = colors[-1]
        try:
            colorized_image[labels == label] = colors[label]
        except:
            colorized_image[labels == label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    return colorized_image

sam2_checkpoint = "/home/adelic/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

if DATASET == 'sos':
    video_dir = "/home/adelic/m2f-anomaly/data/street_obstacle_sequences/raw_data_tmp"
    res_dir = './data/street_obstacle_sequences/ood_prediction_tracked/'
    result_video_dir = '/mnt/sdb1/datasets/street_obstacle_sequences/result_videos/'
    threshold = 0.52

elif DATASET == 'wos':
    video_dir = "/home/adelic/m2f-anomaly/data/wos/raw_data_tmp"
    res_dir = './data/wos/ood_prediction_tracked/'
    result_video_dir = '/mnt/sdb1/datasets/wos/result_videos/'
    threshold = 0.0965028


sequences = os.listdir(video_dir)
sequences.sort()

for sequence in sequences:
    print(sequence)
    seq_dir = os.path.join(video_dir, sequence)
    seq_res_dir = os.path.join(res_dir, sequence)


    frame_names = [
        p for p in os.listdir(seq_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    #init video inference
    inference_state = predictor.init_state(video_path=seq_dir)
    predictor.reset_state(inference_state)
    
    point_database = {}

    # step = len(frame_names) // 10
    start_frame = len(frame_names) - len(frame_names) // 3
    # start_frame = len(frame_names) // 2
    step = 10
    start = 0
    for frame_id in tqdm(range(start, len(frame_names), step)):

        # load ood score
        image_path = os.path.join(seq_dir, frame_names[frame_id])
        image = cv2.imread(image_path)
        masks = mask_generator.generate(image)


        score_path = image_path.replace('raw_data_tmp', 'ood_score_no_norm').replace('_ood_score', '').replace('jpg', 'npy')
        ood_score = np.load(score_path)
        binary_mask = (ood_score >= threshold).astype(np.uint8)


        # load road mask
        road_path = image_path.replace('raw_data_tmp', 'street_masks').replace('.jpg', '_street_masks.npy')
        road = np.load(road_path)

        kernel = np.ones((50, 50), np.uint8)  # Adjust the kernel size as needed
        # Apply morphological closing to fill holes
        closed_image = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # fill in big objects
        closed_image[-1, :] = 1 # street starts from the bottom of the image

        image_copy = closed_image.copy().astype(np.uint8)

        # Define the mask for floodFill (must be 2 pixels larger than the image)
        h, w = closed_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from the borders to mark the background
        for i in range(h):
            if image_copy[i, 0] == 0:
                cv2.floodFill(image_copy, mask, (0, i), 2)
            if image_copy[i, w - 1] == 0:
                cv2.floodFill(image_copy, mask, (w - 1, i), 2)
        for j in range(w):
            if image_copy[0, j] == 0:
                cv2.floodFill(image_copy, mask, (j, 0), 2)
            if image_copy[h - 1, j] == 0:
                cv2.floodFill(image_copy, mask, (j, h - 1), 2)

        if np.any(image_copy == 0):
          mask_to_fill = image_copy == 0
          closed_image[mask_to_fill] = 1

        # closed_image = np.ones_like(binary_mask)
        num_labels, component_masks, stats, centroids = connected_components(binary_mask, closed_image)

        refined_masks = []
        for sam_m in masks:
            sam_m = sam_m['segmentation']
            if np.sum(sam_m) >= (2073600//2):
                continue
            
            for j, uno_m in enumerate(component_masks):
                iou = np.sum(np.logical_and(sam_m, uno_m)) / np.sum(np.logical_or(sam_m, uno_m))

                if iou > 0.7:
                    refined_masks.append(j)

        if num_labels == 0:
            continue



        all_points = []
        ann_frame_idx = frame_id
        for i in range(num_labels):
            if i not in refined_masks:
                continue
            ann_obj_id = i
            points = centroids[i].astype(np.float32)
            # true_indices = np.argwhere(component_masks[i])
            # points = true_indices[np.random.choice(true_indices.shape[0], 3, replace=False)]
            # points = np.array([[y, x] for x, y in points], dtype=np.float32)
            all_points.append(points)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1] * points.shape[0], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )
            # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            #     inference_state=inference_state,
            #     frame_idx=ann_frame_idx,
            #     obj_id=ann_obj_id,
            #     mask=component_masks[i],
            #     # labels=labels,
            # )
        if len(all_points) > 0:
            point_database[frame_id] = np.concatenate(all_points).astype(np.int32)

    os.makedirs(seq_res_dir, exist_ok=True)

    if len(point_database) == 0:
        print('no points found')

        for frame_idx in tqdm(range(len(frame_names))):
        
            result = -1 * np.ones((1080, 1920)) # npixels = 2073600
        
            np.save(os.path.join(seq_res_dir, frame_names[frame_idx].replace('_raw_data', '').replace('jpg', 'npy')), result)

            # plot the result
            prediction = colorize_labels(result.astype(np.int64))

            image = Image.open(os.path.join(seq_dir, frame_names[frame_idx]))

            image = np.asarray(image)
            # Convert RGB to BGR
            image = image[:, :, ::-1].copy()

            alpha = 0.5
            result = alpha * image + (1 - alpha) * prediction

            os.makedirs(os.path.join(result_video_dir, sequence), exist_ok=True)
            cv2.imwrite(os.path.join(result_video_dir, sequence, frame_names[frame_idx]), result)

        continue

    tracking_database = {}
    tracked_ids = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame):        
        if len(out_obj_ids) == 0:
            continue        
        segment_masks = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            segment_mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
            if not segment_mask.any():
                continue


                
            segment_masks[out_obj_id] = segment_mask


        tracking_database[out_frame_idx] = segment_masks


    if 0 not in tracking_database:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame, reverse=True):        
            if len(out_obj_ids) == 0:
                continue        
            segment_masks = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                segment_mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                if not segment_mask.any():
                    continue

                    
                segment_masks[out_obj_id] = segment_mask


            tracking_database[out_frame_idx] = segment_masks

    tracking_database = dict(sorted(tracking_database.items()))
    tracking_database = list(tracking_database.values())

    
    refined_tracking_database = []

    tracked_ids = []
    histoy_mask = {}
    history_time = {}

    for t, frame_dict in enumerate(tracking_database):
        if t == 76:
            a = 5
        if tracked_ids == []:
            new_frame_dict = {}
            for i, (id, mask) in enumerate(frame_dict.items()):
                new_frame_dict[id] = mask
                histoy_mask[id] = mask
                history_time[id] = t
                tracked_ids.append(id)
            refined_tracking_database.append(new_frame_dict)
            continue
        
        new_frame_dict = {}
        for id, mask in frame_dict.items():

            c1 = measurements.center_of_mass(mask)

            closest = 99999
            closest_id = -1
            for idh, maskh in histoy_mask.items():
                c2 = measurements.center_of_mass(maskh)
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist < closest:
                    closest = dist
                    closest_id = idh

            if closest < 200:
                new_frame_dict[closest_id] = mask
                histoy_mask[closest_id] = mask
                history_time[closest_id] = t
            else:
                new_frame_dict[len(tracked_ids)] = mask
                tracked_ids.append(len(tracked_ids))
                histoy_mask[len(tracked_ids)] = mask
                history_time[len(tracked_ids)] = t

        

        refined_tracking_database.append(new_frame_dict)


    good_ids = set()
    consecutive_occurrences = defaultdict(int)
    
    for frame_dict in refined_tracking_database:
        
        for object_id in frame_dict.keys():
            consecutive_occurrences[object_id] += 1
            if consecutive_occurrences[object_id] >= 5:
                good_ids.add(object_id)

        # Reset counts for object IDs not in the current frame
        for object_id in list(consecutive_occurrences.keys()):
            if object_id not in frame_dict:
                consecutive_occurrences[object_id] = 0


    for frame_idx in tqdm(range(len(frame_names))):
        
        result = -1 * np.ones((1080, 1920)) # npixels = 2073600


        for id, segment_mask in refined_tracking_database[frame_idx].items():
            if id not in good_ids:
                continue
            result[segment_mask] = id
    
        np.save(os.path.join(seq_res_dir, frame_names[frame_idx].replace('_raw_data', '').replace('jpg', 'npy')), result)

        # plot the result
        prediction = colorize_labels(result.astype(np.int64))

        image = Image.open(os.path.join(seq_dir, frame_names[frame_idx]))

        image = np.asarray(image)
        # Convert RGB to BGR
        image = image[:, :, ::-1].copy()

        alpha = 0.5
        result = alpha * image + (1 - alpha) * prediction

        if frame_idx in point_database:
            points = point_database[frame_idx]
            for i in range(len(points)):
                point = points[i]
                cv2.drawMarker(result, (int(point[0]), int(point[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        os.makedirs(os.path.join(result_video_dir, sequence), exist_ok=True)
        cv2.imwrite(os.path.join(result_video_dir, sequence, frame_names[frame_idx]), result)





