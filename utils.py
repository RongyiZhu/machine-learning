import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb


def imagenet_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train = datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
    train_loader = DataLoader(train, batch_size=2048, shuffle=True, num_workers=2)
    
    val = datasets.ImageNet(root='./data/imagenet', split='val', transform=transform)
    val_loader = DataLoader(val,  batch_size=512, shuffle=True, num_workers=2)
    
    return train_loader, val_loader

def cifar_loader():
    transform = transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    train = datasets.CIFAR100(root='./data/cifar-100', train=True, transform=transform)
    train_loader = DataLoader(train, batch_size=2048, shuffle=True, num_workers=2)
    
    val = datasets.CIFAR100(root='./data/cifar-100', train=False, transform=transform)
    val_loader = DataLoader(val,  batch_size=16, shuffle=True, num_workers=2)
    
    return train_loader, val_loader

if __name__ == '__main__':
    train, val = cifar_loader()
    pdb.set_trace()
    for i, (x, y) in enumerate(train):
        print(x.shape, y.shape)
        pdb.set_trace()
        break
    