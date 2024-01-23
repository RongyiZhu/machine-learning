import os
import torch.multiprocessing
from torch import nn
import torch.optim as optim
import wandb
import torch
import tqdm
import pdb
from torchstat import stat
import argparse

from MLP import MLP
from CNN import VGG4
from VIT import ViT
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser(description='This is a implementation of some classic machine learning algorithms.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr_warmup_epochs', type=int, default=30, help='lr warmup epochs')
    parser.add_argument('--lr_warmup_decay', type=float, default=0.033, help='lr warmup decay')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    wdb = wandb.init(project='VIT', 
                 config={
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                 }
                 )
    
    train, val = imagenet_loader(batch_size=args.batch_size)
    model = ViT(
        image_size = 224,
        patch_size = 14,
        num_classes = 1000,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to('cuda')
    model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[args.lr_warmup_epochs])
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        for i, (x, y) in tqdm.tqdm(enumerate(train)):
            x, y = x.to('cuda'), y.to('cuda')
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        if epoch % 2 == 0:
            #validation
            model.eval()
            top1 = top5 = 0
            val_loss = []
            for i, (x, y) in enumerate(val):
                x, y = x.to('cuda'), y.to('cuda')
                output = model(x)
                val_loss.append(criterion(output, y).item())
                top1 += (output.argmax(dim=1) == y).sum().item()
                top5 += (output.topk(5, dim=1).indices == y.unsqueeze(dim=1)).sum().item()
            
            top1 /= len(val.dataset)
            top5 /= len(val.dataset)
            val_loss = sum(val_loss) / len(val_loss)
            wandb.log({"top1": top1, "top5": top5, "loss": val_loss})
            
        torch.save(model.state_dict(), f"./ckpt/imagenet/VIT/VIT_base.pth")
    
    