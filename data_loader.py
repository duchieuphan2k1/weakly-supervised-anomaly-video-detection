import numpy as np
import torch
import torch.utils.data as data
import os
from utils import process_feat
torch.set_default_tensor_type('torch.FloatTensor')

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.datafolder = args.datafolder
        self.rgb_list_file = 'list/shanghai-swin-test-10crop-name.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(os.path.join(self.datafolder, self.list[index].strip('\n')), allow_pickle=True)
  
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame