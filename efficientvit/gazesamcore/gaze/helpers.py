import cv2
import numpy as np

__all__ = ["resize_and_pad", "preprocess_gaze", "get_point_along_gaze", "find_edge_intersection"]


def resize_and_pad(
    img: np.ndarray,
    size: tuple,
    pad_color=0,
):
    h, w = img.shape[:2]
    sw, sh = size
    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w / h
    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    # color image but only one color provided
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return scaled_img


def preprocess_gaze(imgs, swap=(2, 0, 1)):
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    resized_images = []

    for image in imgs:
        # Resize (Keep aspect ratio) + Normalization + BGR->RGB
        resized_image = resize_and_pad(
            img=image,
            size=(448, 448),
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = (resized_image - mean) / std
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        resized_images.append(resized_image)

    res = np.asarray(resized_images).astype(np.float32)

    return res


def get_point_along_gaze(face_bb=None, gaze_yawpitch=None):
    if gaze_yawpitch is None:
        return None, None
    else:
        LEN = 15
        dx = -LEN * np.sin(gaze_yawpitch[0]) * np.cos(gaze_yawpitch[1])
        dy = -LEN * np.sin(gaze_yawpitch[1])

        eye_pos = (
            (face_bb[0] + face_bb[2]) // 2,
            (face_bb[1] + face_bb[3]) // 2,
        )  # eye pos is the middle of the box

        arrow_head = eye_pos
        arrow_tail = (eye_pos[0] + dx, eye_pos[1] + dy)

        return arrow_head, arrow_tail


def find_edge_intersection(w, h, start_point, point_along_vec):
    start_x, start_y = start_point
    x, y = point_along_vec

    up = y <= start_y
    right = x >= start_x

    if abs(x) == abs(start_x):  # vertical case
        if y < start_y:
            return (x, 0)
        elif y == start_y:
            return start_point
        else:
            return (x, h - 1)

    m = (y - start_y) / (x - start_x)
    abs_m = abs(m)

    avg_slope = h / w

    if up and right:  # Q1
        new_x, new_y = w - 1, 0
    elif up and not right:  # Q2
        new_x, new_y = 0, 0
    elif not up and right:
        new_x, new_y = w - 1, h - 1
    elif not up and not right:
        new_x, new_y = 0, h - 1
    else:
        raise Exception("Cannot find edge")

    if abs_m < avg_slope:
        new_y = m * (new_x - x) + y
    if abs_m > avg_slope:
        new_x = 1 / m * (new_y - y) + x

    return int(new_x), int(new_y)
