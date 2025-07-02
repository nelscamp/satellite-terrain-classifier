import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class SatelliteDataLoader:
    def __init__(self, data_path, batch_size=32, val_split=0.2, num_workers=4):
        self.data_path = data_path
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data = datasets.ImageFolder(root=self.data_path, transform=self.train_transform)

        self.dataset_size = len(data)
        self.train_size = int((1 - val_split) * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size

        self.train_dataset, self.val_dataset = random_split(data, [self.train_size, self.val_size])
        self.val_dataset.dataset.transform = self.val_transform

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        self.class_names = data.classes

    def get_loaders(self):
        return self.train_loader, self.val_loader
    
    def get_class_names(self):
        return self.class_names