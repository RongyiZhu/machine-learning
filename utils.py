import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets

import pdb

class ImageNet(Dataset):
    def __init__(self, data_dir, idx_to_class, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.idx_to_class = idx_to_class
        self.transform = transform
        
    def __len__(self):
        return len(self.idx_to_class)
    
    def __getitem__(self, idx):
        pass
    
    
if __name__ == '__main__':
    train = datasets.ImageNet(root='./data/imagenet', split='val')
    #pdb.set_trace()
    print(len(train))
    