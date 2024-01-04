import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets

import pdb
  
    
if __name__ == '__main__':
    train = datasets.ImageNet(root='./data/imagenet', split='val')
    #pdb.set_trace()
    print(len(train))
    