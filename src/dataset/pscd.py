import os

from os.path import join as pjoin, splitext as spt

import dataset.transforms as T 
from dataset.dataset import CDDataset, get_transforms

import dataset.path_config as Data_path

class PSCD(CDDataset):
    # all images are 512x512
    def __init__(self, root, rotation=True, transforms=None, revert_transforms=None):
        super(PSCD, self).__init__(root, transforms)
        self.root = root #'/home/main/datasets/PSCD'
        self.rotation = rotation
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for file in os.listdir(os.path.join(self.root, 'mask')):
            if self._check_validness(file):
                idx = int(file.split('.')[0].split('_')[-1])
                if self.rotation or idx == 0:
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file))
                    t1.append(pjoin(self.root, 't1', file))
        return gt, t0, t1



def get_PSCD(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = Data_path.get_dataset_path('PSCD')
    size_dict = {
        224: (224, 224),
        512: (512, 512),
        784: (784, 784),
        1024: (1024, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PSCD(os.path.join(raw_root, mode), 
        transforms=transforms, revert_transforms=revert_transforms)
    print("PSCD {}: {}".format(mode, len(dataset)))
    return dataset
        
class PSCD_Raw(CDDataset):
    # all images are 1024x768
    def __init__(self, root, transforms=None, revert_transforms=None):
        super(PSCD_Raw, self).__init__(root, transforms)
        self.root = root #'/home/main/datasets/PSCD'
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        sub_class = list(f for f in os.listdir(self.root) if os.path.isdir(pjoin(self.root, f)))
        for c in sub_class:
            img_root = pjoin(self.root, c, 'RGB')
            mask_root = pjoin(self.root, c, 'GT')
            for f in os.listdir(mask_root):
                if self._check_validness(f):
                    gt.append(pjoin(mask_root, f))
                    t0.append(pjoin(img_root, f.replace("gt", "1_")))
                    t1.append(pjoin(img_root, f.replace("gt", "2_")))
        return gt, t0, t1

    def get_raw(self, index):
        imgs, mask = super(PSCD_Raw, self).get_raw(index)
        # x == 255 is sky
        mask = mask.point(lambda x: int(0 < x < 255) * 255)
        return imgs, mask


def get_PSCD_Raw(args, train=True):
    mode = 'train' if train else 'test'
    raw_root = Data_path.get_dataset_path('PSCD')
    size_dict = {
        512: (512, 512),
        768: (768, 1024),
        1024: (1024, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PSCD_Raw(os.path.join(raw_root, mode), 
        transforms=transforms, revert_transforms=revert_transforms)
    print("PSCD_Raw {}: {}".format(mode, len(dataset)))
    return dataset
        

