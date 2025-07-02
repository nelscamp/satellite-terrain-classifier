import torch
import torch.nn as nn
import torchvision.models as models

class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(SatelliteCNN, self).__init__()
        
        # Using pre-trained ResNet as base model
        self.base_model = models.resnet50(pretrained=True)

        # classification layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.basemodel.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
