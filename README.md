# To-Do

## Preparation
We use Cifar-100 as the training and testing dataset for reimplement the classic machine learning algorithms. We built our algorithms based on Torch.


## Statistics
| Network | Transformation | Top-1   | Top-5  | Loss   |
|---------|----------------|---------|--------|--------|
| MLP 1   | --             | 0.11    | 0.30   | 0.25   |
| MLP 1   | Normalization  | 18.03%  | 41.37% | 0.2223 |
| MLP 2   | Normalization  | 19.66%  | 42.72% | 0.2192 |





### Hyper-parameter

MLP 1
```
learning_rate: 0.005
Epoch: 100
Optimizer: SGD
Scheduler: None
Batch_size: 1024
Structure: + Linear (3*32*32) -> 2048
           + ReLU
           + Linear 2048 -> 100
```
