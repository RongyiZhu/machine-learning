import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.multiprocessing
from torch import nn
import torch.optim as optim
import wandb
import torch
import tqdm
import pdb

from MLP import MLP

torch.multiprocessing.set_sharing_strategy('file_system')
wdb = wandb.init(project='machine-learning', 
                 config={
                     "learning_rate": 0.005,
                     "epochs": 100,
                 }
                 )

def imagenet_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train = datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
    train_loader = DataLoader(train, batch_size=1024, shuffle=True, num_workers=2)
    
    val = datasets.ImageNet(root='./data/imagenet', split='val', transform=transform)
    val_loader = DataLoader(val,  batch_size=16, shuffle=True, num_workers=2)
    
    return train_loader, val_loader
    
if __name__ == '__main__':
    train, val = imagenet_loader()
    model = MLP()
    
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        for i, (x, y) in tqdm.tqdm(enumerate(train)):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        if epoch % 2 == 0:
            #validation
            top1 = top5 = loss = 0
            for i, (x, y) in enumerate(val):
                output = model(x)
                loss += criterion(output, y)
                top1 += (output.argmax(dim=1) == y).sum().item()
                top5 += (output.topk(5, dim=1).indices == y.unsqueeze(dim=1)).sum().item()
            
            top1 /= len(val.dataset)
            top5 /= len(val.dataset)
            loss /= len(val.dataset)
            wandb.log({"top1": top1, "top5": top5, "loss": loss})
            
            torch.save(model.state_dict(), f"./ckpt/MLP/model_{epoch}.pth")
    
    