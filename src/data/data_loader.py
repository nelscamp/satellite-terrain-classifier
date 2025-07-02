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

        self.full_dataset = datasets.ImageFolder(root=self.data_path)
        self.class_names = self.full_dataset.classes

        train_indices, val_indices = self._stratified_split()

        self.train_dataset = self._create_subset(train_indices, self.train_transform)
        self.val_dataset = self._create_subset(val_indices, self.val_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def get_loaders(self):
        return self.train_loader, self.val_loader
    
    def get_class_names(self):
        return self.class_names
    
    def _stratified_split(self):
        labels = [self.full_dataset[i][1] for i in range(len(self.full_dataset))]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=42)
        train_indices, val_indices = next(sss.split(range(len(self.full_dataset)), labels))

        return train_indices.tolist(), val_indices.tolist()
    
    def _create_subset(self, indices, transform):
        subset = Subset(self.full_dataset, indices)
        subset.dataset.transform = transform
        return subset