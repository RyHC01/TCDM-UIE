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


def paired_random_crop(img_gts, img_raws, raw_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_raws, list):
        img_raws = [img_raws]

    h_raw, w_raw, _ = img_raws[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(raw_patch_size * scale)

    if h_gt != h_raw * scale or w_gt != w_raw * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of raw ({h_raw}, {w_raw}).')
    if h_raw < raw_patch_size or w_raw < raw_patch_size:
        raise ValueError(f'raw ({h_raw}, {w_raw}) is smaller than patch size '
                         f'({raw_patch_size}, {raw_patch_size}). '
                         f'Please remove {gt_path}.')

    top = random.randint(0, h_raw - raw_patch_size)
    left = random.randint(0, w_raw - raw_patch_size)

    img_raws = [
        v[top:top + raw_patch_size, left:left + raw_patch_size, ...]
        for v in img_raws
    ]

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_raws) == 1:
        img_raws = img_raws[0]
    return img_gts, img_raws


def paired_random_crop_DP(img_rawLs, img_rawRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_rawLs, list):
        img_rawLs = [img_rawLs]
    if not isinstance(img_rawRs, list):
        img_rawRs = [img_rawRs]

    h_raw, w_raw, _ = img_rawLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    raw_patch_size = gt_patch_size // scale

    if h_gt != h_raw * scale or w_gt != w_raw * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of raw ({h_raw}, {w_raw}).')
    if h_raw < raw_patch_size or w_raw < raw_patch_size:
        raise ValueError(f'raw ({h_raw}, {w_raw}) is smaller than patch size '
                         f'({raw_patch_size}, {raw_patch_size}). '
                         f'Please remove {gt_path}.')

    top = random.randint(0, h_raw - raw_patch_size)
    left = random.randint(0, w_raw - raw_patch_size)

    img_rawLs = [
        v[top:top + raw_patch_size, left:left + raw_patch_size, ...]
        for v in img_rawLs
    ]

    img_rawRs = [
        v[top:top + raw_patch_size, left:left + raw_patch_size, ...]
        for v in img_rawRs
    ]

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_rawLs) == 1:
        img_rawLs = img_rawLs[0]
    if len(img_rawRs) == 1:
        img_rawRs = img_rawRs[0]
    return img_rawLs, img_rawRs, img_gts


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
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
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def data_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.fliplr(image)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 1)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
