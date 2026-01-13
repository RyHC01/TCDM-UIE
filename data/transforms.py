import cv2
import random
import numpy as np


def mod_crop(img, scale):
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_inputs, input_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_inputs, list):
        img_inputs = [img_inputs]

    h_inp, w_inp, _ = img_inputs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(input_patch_size * scale)

    if h_gt != h_inp * scale or w_gt != w_inp * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of INPUT ({h_inp}, {w_inp}).')
    if h_inp < input_patch_size or w_inp < input_patch_size:
        raise ValueError(f'INPUT ({h_inp}, {w_inp}) is smaller than patch size '
                         f'({input_patch_size}, {input_patch_size}). '
                         f'Please remove {gt_path}.')

    top = random.randint(0, h_inp - input_patch_size)
    left = random.randint(0, w_inp - input_patch_size)

    img_inputs = [
        v[top:top + input_patch_size, left:left + input_patch_size, ...]
        for v in img_inputs
    ]

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_inputs) == 1:
        img_inputs = img_inputs[0]
    return img_gts, img_inputs


def paired_random_crop_DP(img_inputLs, img_inputRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_inputLs, list):
        img_inputLs = [img_inputLs]
    if not isinstance(img_inputRs, list):
        img_inputRs = [img_inputRs]

    h_inp, w_inp, _ = img_inputLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    input_patch_size = gt_patch_size // scale

    if h_gt != h_inp * scale or w_gt != w_inp * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of INPUT ({h_inp}, {w_inp}).')
    if h_inp < input_patch_size or w_inp < input_patch_size:
        raise ValueError(f'INPUT ({h_inp}, {w_inp}) is smaller than patch size '
                         f'({input_patch_size}, {input_patch_size}). '
                         f'Please remove {gt_path}.')

    top = random.randint(0, h_inp - input_patch_size)
    left = random.randint(0, w_inp - input_patch_size)

    img_inputLs = [
        v[top:top + input_patch_size, left:left + input_patch_size, ...]
        for v in img_inputLs
    ]

    img_inputRs = [
        v[top:top + input_patch_size, left:left + input_patch_size, ...]
        for v in img_inputRs
    ]

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_inputLs) == 1:
        img_inputLs = img_inputLs[0]
    if len(img_inputRs) == 1:
        img_inputRs = img_inputRs[0]
    return img_inputLs, img_inputRs, img_gts


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.fliplr(image)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 1)  # (0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
