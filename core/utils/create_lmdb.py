import argparse
from os import path as osp

from core.utils import scandir
from core.utils.lmdb_util import make_lmdb_from_imgs
import os
import tqdm
import cv2


def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys



