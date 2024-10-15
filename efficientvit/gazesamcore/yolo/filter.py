import numpy as np
import torch

__all__ = ["filter_bboxes_face", "filter_bboxes_gaze", "filter_bboxes_depth"]


def filter_bboxes_face(bboxes, face_bbox):
    mask = (
        (bboxes[:, 0] <= face_bbox[0])
        & (bboxes[:, 1] <= face_bbox[1])
        & (bboxes[:, 2] >= face_bbox[2])
        & (bboxes[:, 3] >= face_bbox[3])
    )
    keep_ixes = np.logical_not(mask)
    return bboxes[keep_ixes]


def filter_bboxes_gaze(bboxes, gaze_head, gaze_tail):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # check if line segment 'ab' intersects line segment 'cd'
    def intersect(a, b, c, d):
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    inds = np.zeros(len(bboxes), dtype=np.uint8)

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        p1, p2, p3, p4 = (x1, y1), (x2, y1), (x2, y2), (x1, y2)

        # determine if gaze line intersects any bbox edge
        combo1 = intersect(p1, p2, gaze_head, gaze_tail)
        combo2 = intersect(p2, p3, gaze_head, gaze_tail)
        combo3 = intersect(p4, p3, gaze_head, gaze_tail)
        combo4 = intersect(p1, p4, gaze_head, gaze_tail)
        if combo1 or combo2 or combo3 or combo4:
            inds[i] = 1

    return bboxes[inds == 1]


def filter_bboxes_depth(bboxes, depth_map, eye_pos, depth_margin=20, min_overlap_ratio=0.5):
    thresh = depth_map[int(eye_pos[1])][int(eye_pos[0])] - depth_margin
    depth_mask = (depth_map >= thresh).to(torch.int8)
    bbox_area = (bboxes[:, 3] - bboxes[:, 1] + 1) * (bboxes[:, 2] - bboxes[:, 0] + 1)

    foreground_inds = []
    for i, bb in enumerate(bboxes):
        in_front = depth_mask[bb[1] : bb[3] + 1, bb[0] : bb[2] + 1].count_nonzero()

        if in_front >= min_overlap_ratio * bbox_area[i]:
            foreground_inds.append(i)

    return bboxes[foreground_inds], depth_mask
