# To-Do

## Preparation
We use Cifar-100 as the training and testing dataset for reimplement the classic machine learning algorithms. We built our algorithms based on Torch.


## Statistics
Cifar-100
| Network |  #Params   | Top-1   | Top-5  | Loss   |
|---------|------------|---------|--------|--------|
| MLP 1   | --         | 0.11    | 0.30   | 0.25   |
| MLP 1.1 | --         | 18.03%  | 41.37% | 0.2223 |
| MLP 2   | --         | 19.66%  | 42.72% | 0.2192 |
| MLP 2.1 | --         | 25.76%  | 52.15% | 0.0015 |
| MLP 3   | 10,702,948 | 28.27%  | 55.06% | 0.0015 |
| MLP 3.1 | 10,702,948 | 28.22%  | 55.11% | 0.0015 |
| MLP 3.2 | 117,911,652| 30.04%  | 55.51% | 0.00149|
| MLP 3.3 | 10,702,948 | 10.59%  | 26.76% | 4.31e-4|
| MLP 3.4 | 10,702,948 | 15.54%  | 33.81% | 0.1195 |
| MLP 4   |   326,372  | 27.24%  | 54.24% | 3.158  |
| MLP 4.1 |       --   | 33.35%  | 61.59% | 2.976  |
|---------|------------|---------|--------|--------|
| VGG 1   | 8,568,164  | 38.83%  | 66.16% | 0.0027 |
| VGG 1.1 | --         | 32.65%  | 61.64% | 0.0006 |
| VGG 1.2 | --         | 41.11%  | 68.60% | 0.0114 |
| VGG 1.3 | --         | 38.46%  | 66.6%  | 6.79e-4|
| VGG 1.4 | --         | 12.49%  | 29.79% | 7.9e-4 |
| VGG 1.5 | --         | 48.22%  | 74.61% | 0.00727|
|---------|------------|---------|--------|--------|


ImageNet-1K
| Network |  #Params   | Top-1   | Top-5  | Loss   |
|---------|------------|---------|--------|--------|
|VIT_base | 86,450,560 |



Notice: MLP1,MLP2,MLP3,VGG1 loss calculation might be wrong

### Hyper-parameter

MLP 1
```
learning_rate: 0.005
Epoch: 100
Optimizer: SGD
Scheduler: None
Batch_size: 1024
Transformation: [MLP1.1: Normalization]
Structure: + Linear (3*32*32) -> 2048
           + ReLU
           + Linear 2048 -> 100
```

MLP 2
```
learning_rate: 0.005
Epoch: 100 [MLP2.1:500]
Optimizer: SGD
Scheduler: None [MLP2.1:StepLR(size=50, gamma=0.1)]
Batch_size: 1024
Transformation: Normalization
Structure: + Linear (3*32*32) -> 2048
           + Bn (2048)
           + ReLU
           + Linear 2048 -> 100 
```


MLP 3
```
learning_rate: 0.005 [MLP3.1:0.01]
Epoch: 500
Optimizer: SGD
Scheduler: StepLR(size=50, gamma=0.1) [MLP3.1:StepLR(size=50, gamma=0.5)]
Batch_size: 2048 [MLP3.3:16384]
Transformation: Normalization [MLP3.3:+RandAugment]
Structure: + Linear (3*32*32) -> 2048 [MLP3.2:16384]
           + Bn (2048) [MLP3.2:16384]
           + ReLU
           + Linear 2048 -> 2048 [MLP3.2:16384->4096]
           + Bn (2048) 
           + ReLU
           + Linear 2048 -> 100 [MLP3.2:4096->100]
```

MLP 4
```
learning_rate: 0.01
Epoch: 5000
Optimizer: SGD
Scheduler: StepLR(size=500, gamma=0.5) 
Batch_size: 8192
Transformation: Normalization + RandAugment
Structure: + 16 x Patch_Linear[(3*8*8, 64) -> bn(64) -> ReLU -> (64*8)]
           + Linear (8*16) -> 512
           + ReLU
           + Linear 512 -> 100
                  
```


VGG 1
```
learning_rate: 0.01
Epoch: 500 [VGG1.3:]
Optimizer: SGD [VGG1.4:Adam]
Scheduler: StepLR(size=100, gamma=0.5)
Batch_size: 2048 [VGG1.1:8192] [VGG1.2:512]
Transformation: Normalization
Structure: + Conv2d(3, 64, kernel_size=3, padding=1)
           + Bn (64)
           + ReLU
           + MaxPool2d(kernel_size=2, stride=2)
            
           + Conv2d(64, 128, kernel_size=3, padding=1)
           + Bn (128)
           + ReLU
           + MaxPool2d(kernel_size=2, stride=2)

           + Linear (128*8*8) -> 1024
           + Bn (4096)
           + ReLU
           + Linear 1024 -> 100 
  [MLP1.5] + Softmax()
```

