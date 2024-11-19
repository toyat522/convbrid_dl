import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor, v2
import matplotlib.pyplot as plt
import random
import os

class RequiredDatasets(Dataset):
    def __init__(self, root_dir, train_transform=None, test_transform=None):
        """
        root_dir: root directory where dataset is downloaded
        train_transform: transform to train dataset
        test_transform: transform to test dataset
        """
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform

    def get_CIFAR100_dataset(self, split=0.8):
        """
        Downloads dataset locally and generates training, validation and test datasets.

        split is the proportion of the train/val data that will be used for training.
        """
        train_and_val_dataset = datasets.CIFAR100(root=self.root_dir,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.ToTensor(),
                                                  )
        
        train_size = int(len(train_and_val_dataset) * split)
        val_size = len(train_and_val_dataset) - train_size

        train_dataset, val_dataset = random_split(train_and_val_dataset, [train_size, val_size])
        
        test_dataset = datasets.CIFAR100(root=self.root_dir,
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.ToTensor(),
                                                 )
        return train_dataset, val_dataset, test_dataset

    def augment(self, transform, dataset, num_augments, download=False, save_dir=None):
        """
        Performs augmentation using a specified transform on a dataset, 
        such that the new dataset augments each image num_augments times.
        If download = True, downloads the augmented images locally at location save_dir

        Returns a new dataset of size target_size.
        """
        new_dataset = []
        for image in dataset:
            for _ in range(num_augments):
                image = transform(image)
                new_dataset.append(image)
        
        if download:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for idx, image in enumerate(new_dataset):
                utils.save_image(image[0], '{}/image_{}.png'.format(save_dir, idx))
        return new_dataset
        
    # def augment_example(self, dataset, download=False, save_dir = None):
    #     """
    #     Example transform.
    #     """
    #     torch.manual_seed(10)
    #     random.seed(10)
    #     transforms = v2.Compose([
    #         v2.RandomResizedCrop(size=(32, 32), antialias=True),
    #         v2.RandomHorizontalFlip(p=0.5),
    #         v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     return self.augment(transforms, dataset, 2, download, save_dir)
    
    def get_dataloader(self, dataset, batch_size):
        """
        Split is proportion of training that will be used 
        """        
        return DataLoader(dataset, batch_size,
                        shuffle=True, num_workers=10)

if __name__ == "__main__":
    dataset = RequiredDatasets('./data')
    batch_size = 64

    train, val, test = dataset.get_CIFAR100_dataset()

    torch.manual_seed(10)
    random.seed(10)

    transform = v2.Compose([
            v2.RandomResizedCrop(size=(32, 32), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    augmented_train = dataset.augment(transform, train, 2, download=True, save_dir="./augment_test") #each image is augmented to make 2
    train_dataloader = dataset.get_dataloader(augmented_train, batch_size)
    val_dataloader = dataset.get_dataloader(val, batch_size)
    test_dataloader = dataset.get_dataloader(test, batch_size)