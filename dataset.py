import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor, v2
import matplotlib.pyplot as plt
import random
import os

class CIFAR100(Dataset):
    def __init__(self, root_dir, train_transform=None, test_transform=None):
        """
        root_dir: root directory where dataset is downloaded
        train_transform: transform to train dataset
        test_transform: transform to test dataset
        """
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform

    def get_dataset(self):
        """
        Downloads dataset locally and generates training and test datasets.
        """
        train_dataset = datasets.CIFAR100(root=self.root_dir,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.ToTensor(),
                                                  )
        test_dataset = datasets.CIFAR100(root=self.root_dir,
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.ToTensor(),
                                                 )
        return train_dataset, test_dataset

    def augment(self, transform, dataset, target_size, download=False, save_dir=None):
        """
        Performs augmentation using a specified transform on a dataset, 
        such that the new dataset has size target_size.

        Returns a new dataset of size target_size.
        """
        num_images = len(dataset)
        new_dataset = []
        for _ in range(target_size):
            idx = random.randint(0, num_images-1) 
            print(idx)
            image = dataset[idx]
            image = transform(image)
            new_dataset.append(image)
        
        if download:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for idx, image in enumerate(new_dataset):
                utils.save_image(image[0], '{}/image_{}.png'.format(save_dir, idx))
        return new_dataset
        
    def augment_example(self, dataset, download=False, save_dir = None):
        """
        Example transform.
        """
        transforms = v2.Compose([
            v2.RandomResizedCrop(size=(32, 32), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return self.augment(transforms, dataset, 1000, download, save_dir)
    


if __name__ == "__main__":
    test_dataset = CIFAR100('./data')
    train, test = test_dataset.get_dataset()
    # for idx in range(5):
    #     plt.subplot(1, 5, idx + 1)
    #     plt.imshow(train[idx][0].permute(1, 2, 0))
    
    augmented_train = test_dataset.augment_example(train, download=True, save_dir='./augmented')
    for idx in range(5):
        plt.subplot(1, 5, idx + 1)
        plt.imshow(augmented_train[idx][0].permute(1, 2, 0))
    plt.show()