import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class SatelliteDataLoader:
    def __init__(self, data_path, batch_size=32, val_split=0.2, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_split = val_split

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_full_dataset = datasets.ImageFolder(root=self.data_path, transform=self.train_transform)
        val_full_dataset = datasets.ImageFolder(root=self.data_path, transform=self.val_transform)

        self.class_names = train_full_dataset.classes

        labels = [train_full_dataset[i][1] for i in range(len(train_full_dataset))]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        train_indices, val_indices = next(sss.split(range(len(train_full_dataset)), labels))

        self.train_dataset = Subset(train_full_dataset, train_indices.tolist())
        self.val_dataset = Subset(val_full_dataset, val_indices.tolist())

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def get_loaders(self):
        return self.train_loader, self.val_loader
    
    def get_class_names(self):
        return self.class_names