from torch.utils import data as data
from torchvision.transforms.functional import normalize

from core.data.data_util import (paired_paths_from_folder,
                                 paired_paths_from_lmdb,
                                 paired_paths_from_meta_info_file)
from core.data.transforms import random_augmentation
from core.utils import FileClient, imfrombytes, img2tensor, padding
from core.data.transforms import paired_random_crop
import random
import torch


class Dataset_PairedImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.raw_folder = opt['dataroot_gt'], opt['dataroot_raw']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.raw_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['raw', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.raw_folder, self.gt_folder], ['raw', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.raw_folder, self.gt_folder], ['raw', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.raw_folder, self.gt_folder], ['raw', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    @staticmethod
    def get_params(img, output_size, n):
        w, h, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return [0] * n, [0] * n, h, w
    
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img[y[i]: y[i] + w, x[i]: x[i] + h, :]
            crops.append(new_crop)
        return crops

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        raw_path = self.paths[index]['raw_path']
        img_bytes = self.file_client.get(raw_path, 'raw')
        try:
            img_raw = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("raw path {} not working".format(raw_path))

        gt_size = self.opt['gt_size']

        img_gt, img_raw = padding(img_gt, img_raw, gt_size)

        if self.opt['phase'] == 'train' and self.geometric_augs:
            i, j, h, w = self.get_params(img_gt, (gt_size, gt_size), 1)
            img_raw = img_raw[j[0]: j[0] + w, i[0]: i[0] + h, :]
            img_gt = img_gt[j[0]: j[0] + w, i[0]: i[0] + h, :]
            img_gt, img_raw = random_augmentation(img_gt, img_raw)
            img_gt, img_raw = img2tensor([img_gt, img_raw], bgr2rgb=True, float32=True)

        else:
            img_gt, img_raw = img2tensor([img_gt, img_raw], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_raw, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'raw': img_raw,
            'gt': img_gt,
            'raw_path': raw_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
