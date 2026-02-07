from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (paired_paths_from_folder,
                            paired_paths_from_lmdb,
                            paired_paths_from_meta_info_file)
from data.transforms import random_augmentation
from utils import FileClient, imfrombytes, img2tensor, padding
import random
import torch


class Dataset_PairedImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.input_folder = opt['dataroot_gt'], opt['dataroot_input']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.input_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['input', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.input_folder, self.gt_folder], ['input', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.input_folder, self.gt_folder], ['input', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.input_folder, self.gt_folder], ['input', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    @staticmethod
    def get_params(img, output_size, n):
        w, h, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            # 修改这里，返回列表而不是单个整数
            return [0] * n, [0] * n, h, w
    
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        # print(img.shape)
        for i in range(len(x)):
            new_crop = img[y[i]: y[i] + w, x[i]: x[i] + h, :]
            # print(new_crop.shape)
            # new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return crops

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        input_path = self.paths[index]['input_path']
        img_bytes = self.file_client.get(input_path, 'input')
        try:
            img_input = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("input path {} not working".format(input_path))

        # augmentation for training
        gt_size = self.opt['gt_size']
        patch_n = self.opt['patch_n']

        # padding
        img_gt, img_input = padding(img_gt, img_input, gt_size)

        # random_crops
        if self.opt['phase'] == 'train' and self.geometric_augs:
            i, j, h, w = self.get_params(img_gt, (gt_size, gt_size), patch_n)
            img_inputs = self.n_random_crops(img_input, i, j, h, w)
            img_gts = self.n_random_crops(img_gt, i, j, h, w)
            img_gts_with_inputs = [random_augmentation(img_gts[i], img_inputs[i]) for i in range(len(img_gts))]
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gts_with_inputs_tensor = [img2tensor([img_gt, img_input], bgr2rgb=True, float32=True) for img_gt, img_input in
                                       img_gts_with_inputs]
            img_gts = [x for x, _ in img_gts_with_inputs_tensor]
            img_inputs = [x for _, x in img_gts_with_inputs_tensor]
            img_gt = torch.stack(img_gts, dim=0)
            img_input = torch.stack(img_inputs, dim=0)

        else:
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_input = img2tensor([img_gt, img_input], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_input, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'input': img_input,
            'gt': img_gt,
            'input_path': input_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
