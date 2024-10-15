import cv2
import numpy as np
import torch
from torch.nn import functional as F

"""
    Some functions in this file are modified from https://github.com/SysCV/sam-hq/blob/main/train/utils/misc.py.
"""

__all__ = [
    "point_sample",
    "cat",
    "get_uncertain_point_coords_with_randomness",
    "dice_loss",
    "sigmoid_ce_loss",
    "calculate_uncertainty",
    "loss_masks",
    "mask_iou",
    "compute_iou",
    "mask_to_boundary",
    "boundary_iou",
    "compute_boundary_iou",
    "masks_sample_points",
    "mask_iou_batch",
]


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: list[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list.
    """

    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """

    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, mode: str):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if mode == "none":
        return loss
    else:
        return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, mode: str):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if mode == "none":
        return loss.mean(1)
    else:
        return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """

    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0, mode="mean"):
    """
    Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, mode)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks, mode)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


def mask_iou(pred_label, label):
    """
    calculate mask iou for pred_label and gt_label.
    """

    pred_label = (pred_label > 0)[0].int()
    label = (label > 128)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / (union + 1e-6)


def compute_iou(preds, target):
    if preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode="bilinear", align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + mask_iou(postprocess_preds[i], target[i])
    return iou / len(preds)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """

    device = gt.device
    dt = (dt > 0)[0].cpu().byte().numpy()
    gt = (gt > 128)[0].cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / (union + 1e-6)
    return torch.tensor(boundary_iou).float().to(device)


def compute_boundary_iou(preds, target):
    if preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode="bilinear", align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0, len(preds)):
        iou = iou + boundary_iou(target[i], postprocess_preds[i])
    return iou / len(preds)


def masks_sample_points(masks, k=10):
    """Sample points on mask"""

    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    # k = 10
    samples = []
    for b_i in range(len(masks)):
        select_mask = masks[b_i].bool()
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)

        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:, None], samples_y[:, None]), dim=1)
        samples.append(samples_xy)

    samples = torch.stack(samples)

    return samples


def mask_iou_batch(pred_label, label):
    """
    calculate mask iou for pred_label and gt_label.
    """

    pred_label = (pred_label > 0).int()
    label = (label > 128).int()

    intersection = ((label * pred_label) > 0).sum(dim=(-1, -2))
    union = ((label + pred_label) > 0).sum(dim=(-1, -2))
    return intersection / (union + 1e-6)
