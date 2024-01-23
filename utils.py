import os
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import einops
import torchvision


import pdb


def imagenet_loader(batch_size=2048):
    train_transform = transforms.Compose([
        transforms.RandAugment(2, 10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
    ])
    train = datasets.ImageNet(root='./data/imagenet', split='train', transform=train_transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val = datasets.ImageNet(root='./data/imagenet', split='val', transform=val_transform)
    val_loader = DataLoader(val,  batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader, val_loader

def cifar_loader(batch_size=2048):
    transform = transforms.Compose([
        transforms.RandAugment(2, 10),
        #RandAugment(2, 10),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    train = datasets.CIFAR100(root='./data/cifar-100', train=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val = datasets.CIFAR100(root='./data/cifar-100', train=False, transform=transform)
    val_loader = DataLoader(val,  batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader, val_loader

def get_patches(imgs, blocks=4):
    size = stride = int(imgs.shape[-1]/blocks)
    patches = imgs.unfold(2, size, stride).unfold(3, size, stride)
    return einops.rearrange(patches, 'b c h w h1 w1 -> b (h w) c h1 w1')
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val = cifar_loader(8192)
    #print(next(model.parameters()).device) 
    # model.load_state_dict(torch.load('./ckpt/cifar/MLP/MLP.pth'), strict=False)
    