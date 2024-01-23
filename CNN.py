from torch import nn
from torchstat import stat

class VGG4(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG4, self).__init__()
        # self.features = []     NOTICE: including layers in a list will cause error in moving model to GPU
        
        self.features1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        self.features2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = self.softmax(x)  # NOTICE: no softmax here because we use CrossEntropyLoss
        return x
    
if __name__ == '__main__':
    stat(VGG4(), (3, 32, 32))
    