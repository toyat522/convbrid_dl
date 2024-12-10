import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

class CIFARDataLoader:
    default_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    default_val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_set=torchvision.datasets.CIFAR100, root = './data',
                 train_transform = None, val_transform = None,
                 batch_size=1024, num_workers=8, seed=42):
        self.root = root

        self.train_transform = train_transform if train_transform is not None else self.default_train_transform
        self.val_transform = val_transform if val_transform is not None else self.default_val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.seed = seed
        torch.random.manual_seed(seed)

        self.train_dataset, self.val_dataset = self._get_datasets(data_set)
        self.train_loader, self.val_loader = self._get_dataloaders()

    def _get_datasets(self, data_set):
        train_dataset = data_set(root=self.root, train=True, download=True, transform=self.train_transform)
        val_dataset = data_set(root=self.root, train=False, download=True, transform=self.val_transform)

        return train_dataset, val_dataset
    
    def _get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader