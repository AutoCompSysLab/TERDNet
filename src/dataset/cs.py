import os

from os.path import join as pjoin, splitext as spt

import dataset.transforms as T 
from dataset.dataset import CDDataset, get_transforms

import dataset.path_config as Data_path

class CSim(CDDataset):
    # all images are 512x512
    def __init__(self, root, rotation=True, transforms=None, revert_transforms=None):
        super(CSim, self).__init__(root, transforms)
        self.root = root #'/home/main/datasets/CSim'
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

class CSim_test(CDDataset):
    # all images are 512x512
    def __init__(self, root, rotation=True, transforms=None, revert_transforms=None):
        super(CSim_test, self).__init__(root, transforms)
        self.root = root #'/home/main/datasets/CSim'
        self.rotation = rotation
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        for root_path in self.root:
            for file in os.listdir(os.path.join(root_path, 'mask')):
                if self._check_validness(file):
                    idx = int(file.split('.')[0].split('_')[-1])
                    if self.rotation or idx == 0:
                        gt.append(pjoin(root_path, 'mask', file))
                        t0.append(pjoin(root_path, 't0', file))
                        t1.append(pjoin(root_path, 't1', file))
        return gt, t0, t1

def get_CSim(args, train=True):
    mode = 'train' if train else 'test'
    #raw_root = Data_path.get_dataset_path('CSim')
    if mode == 'train':
        root_path = '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/train'
    elif mode == 'test':
        root_path = ['/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_0',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_1',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_2',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_3',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_4',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_5',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_6',
                    '/home/main/workspace/jaewoo1/ChangeWarp/data_path/ChangeSim/test_7']
    size_dict = {
        256: (256, 256),
        512: (512, 512),
        768: (768, 1024),
        1024: (1024, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    if mode == 'train':
        dataset = CSim(root_path, 
                        transforms=transforms, revert_transforms=revert_transforms)
    elif mode == 'test':
        dataset = CSim_test(root_path, 
                        transforms=transforms, revert_transforms=revert_transforms)
    
    print("CSim {}: {}".format(mode, len(dataset)))
    return dataset
        