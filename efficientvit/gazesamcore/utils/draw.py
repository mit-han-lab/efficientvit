import cv2
import numpy as np
import torch

from efficientvit.gazesamcore.gaze.helpers import find_edge_intersection
from efficientvit.gazesamcore.utils.consts import GREY, IOU_THRESH, TENSORRT, YELLOW

__all__ = ["annotate_frame", "annotate_blank_frame", "draw_mask"]


def annotate_frame(
    frame,
    gaze_head,
    gaze_tail,
    bboxes,
    masks,
    iou_preds,
    depth_mask,
    min_overlap_thresh=2,
):
    if len(masks) != len(bboxes):
        print(f"err: num bboxes {len(bboxes)} must match num masks {len(masks)}")
        return frame, None, None

    bbox_areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

    best_mask, best_bbox, largest_overlap = None, None, 0

    for i, (bbox, mask, iou_pred) in enumerate(zip(bboxes, masks, iou_preds)):
        mask = mask[0]

        # ignore segmentations covering the eyes
        if mask[gaze_head[1]][gaze_head[0]] == 1:
            continue

        if iou_pred < IOU_THRESH:
            continue

        # crop mask and depth mask to include only bbox region
        mask_crop = mask[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
        depth_crop = depth_mask[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]

        # number of pixels where the segmentation is in front of the eyes
        segmented_in_front_area = torch.count_nonzero(mask_crop & depth_crop)
        # number of segmented pixels
        segmented_area = mask_crop.count_nonzero()

        # ensure some % of the segmentation is in front of the eyes
        if segmented_in_front_area * min_overlap_thresh >= segmented_area:
            percentage_overlap = segmented_in_front_area / bbox_areas[i]
            # store largest overlap
            if percentage_overlap > largest_overlap:
                best_mask, best_bbox = mask, bbox
                largest_overlap = percentage_overlap

    # draw mask and bbox around the best segmentation
    if best_mask is not None:
        frame = draw_mask(frame, best_mask)
        # draw gaze line
        box_midpoint = (best_bbox[0] + best_bbox[2]) // 2, (best_bbox[1] + best_bbox[3]) // 2
        h, w, _ = frame.shape
        endpoint = find_edge_intersection(w, h, gaze_head, box_midpoint)
        cv2.line(frame, gaze_head, endpoint, GREY, 2)
    else:
        cv2.line(frame, gaze_head, gaze_tail, GREY, 2)
    return frame, best_mask, best_bbox


# when presented with no bboxes for this frame, use prev mask and bbox
def annotate_blank_frame(frame, gaze_head, gaze_tail, saved_mask, saved_bbox, runtime, webcam):
    # As PyTorch and ONNX modes are slower, webcam FPS will not be real time, thus saved masks
    # may not fit well
    if saved_mask is not None and (not webcam or runtime == TENSORRT):
        frame = draw_mask(frame, saved_mask)
        box_midpoint = (saved_bbox[0] + saved_bbox[2]) // 2, (saved_bbox[1] + saved_bbox[3]) // 2
        h, w, _ = frame.shape
        endpoint = find_edge_intersection(w, h, gaze_head, box_midpoint)
        cv2.line(frame, gaze_head, endpoint, GREY, 2)
    else:
        cv2.line(frame, gaze_head, gaze_tail, GREY, 2)

    return frame


def draw_mask(frame, mask, device="cuda"):
    torch_img = torch.tensor(frame, device=device)
    color_mask = torch.zeros_like(torch_img, dtype=torch.uint8, device=device)
    color_mask[mask == 1] = torch.tensor(YELLOW, dtype=torch.uint8, device=device)

    mix = color_mask * 0.5 + torch_img * 0.5
    expanded_mask = torch.unsqueeze(mask, axis=2)
    canvas = expanded_mask * mix + ~expanded_mask * torch_img
    frame = canvas.cpu().numpy().astype(np.uint8)

    return frame
