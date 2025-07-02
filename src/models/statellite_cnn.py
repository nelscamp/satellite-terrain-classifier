import torch
import torch.nn as nn
import torch.nn.functional as F

class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(SatelliteCNN, self).__init__()
        # input is 3x256x256
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), # output is now 64x128x128
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # output is now 128x128x128
            nn.ReLU(),
            nn.MaxPool2d(2) # output is now 128x64x64
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # output is now 256x64x64
            nn.ReLU(),
            nn.MaxPool2d(2) # output is now 256x32x32
        )

        self.fc1 = nn.Linear(256*32*32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
