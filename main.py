import os
import torch.multiprocessing
from torch import nn
import torch.optim as optim
import wandb
import torch
import tqdm
import pdb

from MLP import MLP
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')
wdb = wandb.init(project='Linear-perceptron', 
                 config={
                    "learning_rate": 0.005,
                    "epochs": 500,
                    "batch_size": 2048,
                    "dataset": "cifar-100",
                    "model": "MLP-3",
                    "optimizer": "SGD",
                    "scheduler": "StepLR",
                 }
                 )

if __name__ == '__main__':
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    
    train, val = cifar_loader()
    model = MLP(image_size=32, num_classes=100)
    
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(500):
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
            
        torch.save(model.state_dict(), f"./ckpt/cifar/MLP/MLP_8.pth")
    
    