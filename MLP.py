import torch
import math
from torch import nn
import einops
from utils import get_patches
import torchstat
import pdb


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device='None'):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features), device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.kaiming_uniform(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        return torch.matmul(x, self.wight, bias=self.bias)
    
    
class MLP(nn.Module):
    def __init__(self, image_size=224, num_classes=100):
        super(MLP, self).__init__()
        self.patch_num = 16
        
        self.patch_linear = nn.ModuleList()
        for i in range(self.patch_num):
            self.patch_linear.append(nn.Sequential(nn.Linear(3*8*8, 64),
                                                   nn.BatchNorm1d(64),
                                                   nn.ReLU(),
                                                   nn.Linear(64, 8)))
            
        self.classifier = nn.Sequential(nn.Linear(8*self.patch_num, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Linear(512, num_classes))
            
    def forward(self, x):
        x = get_patches(x, blocks=4)
        x = einops.rearrange(x, 'b p c h w -> b p (c h w)')
        temp = torch.zeros((x.shape[0], self.patch_num, 8), device=x.device)
        for i in range(self.patch_num):
            for layer in self.patch_linear:
                temp[:, i, :] = layer(x[:, i, :])
        x = einops.rearrange(temp, 'b p c -> b (p c)')
        x = self.classifier(x)
        return x
    
class MLP3(nn.Module):
    def __init__(self, num_classes=100):
        super(MLP3, self).__init__()
        self.layer_dim = 4096
        
        self.classifier = nn.Sequential(nn.Linear(3*32*32, self.layer_dim),
                                        nn.BatchNorm1d(self.layer_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.layer_dim, self.layer_dim),
                                        nn.BatchNorm1d(self.layer_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.layer_dim, num_classes))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

        
        
if __name__ == '__main__':
    #input = torch.randn(16, 3, 32, 32).to('cuda')
    model = MLP()
    torchstat.stat(model, (3, 32, 32))
    
    
    