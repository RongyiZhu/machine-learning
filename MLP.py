import torch
import math
from torch import nn
import einops


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
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(3*224*224, 4096)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4096, 1000)
        
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
    