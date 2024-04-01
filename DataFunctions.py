import _pickle as cPickle
import time
import os

import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
import numpy as np
import torch
from OutputHandler import print_and_log_message
from torch.utils.data import Dataset


def gi_simulation(mnist_loader, diffuser, img_dim):
    images = torch.stack([image for image, _ in mnist_loader], dim=0)
    measurements = torch.matmul(diffuser, images.view(-1, img_dim).float().t())
    return measurements.t()


def load_data_mnist(pic_width):
    transform = transforms.Compose([transforms.Resize((pic_width, pic_width)),
                                    transforms.ToTensor()])  # convert data to torch.FloatTensor
    data_set = datasets.MNIST(root='data', download=True, transform=transform)
    return data_set


def build_dataset(pic_width, n_masks):
    img_dim = pic_width**2
    data_set = load_data_mnist(pic_width)
    diffuser = torch.randn(n_masks, img_dim)
    gi_data_set = gi_simulation(data_set, diffuser, img_dim)
    return gi_data_set, diffuser

#
# class GI_Mnist(Dataset):
#     def __init__(self, pic_width, n_masks):
#         self.data, self.diffuser = build_dataset(n_masks, pic_width)
#
#     def __len__(self):
#         # returns the length of the dataset.
#         return len(self.data)
#
#     def __getitem__(self, index):
#         # return the element at a given index in the dataset.
#         return self.data[index], self.data[index]

class CustomDataset(torch.utils.data.Dataset):
    # Define a custom dataset class to combine measurements with images
    def __init__(self, dataset, diffuser, img_dim):
        self.diffuser = diffuser
        self.img_dim = img_dim
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        measurement = torch.matmul(self.diffuser, image.view(-1, self.img_dim).float().t())
        return measurement, image


def shorten_ds(n_samples, train_dataset, test_dataset):
    # Calculate number of samples for train and test datasets
    n_train_samples = int(n_samples * 0.8)
    n_test_samples = n_samples - n_train_samples

    # Create random indices for train and test subsets
    train_indices = torch.randperm(len(train_dataset))[:n_train_samples]
    test_indices = torch.randperm(len(test_dataset))[:n_test_samples]

    # Create train and test subsets
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    return train_subset, test_subset

def get_data(pic_width, n_masks, batch_size, n_samples):
     img_dim = pic_width**2
     transform = transforms.Compose([
         transforms.Resize(pic_width),  # Resize the image to pic_width x pic_width
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
     ])

     train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
     test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
     diffuser = torch.randn(n_masks, img_dim)  # Define diffuser tensor

     train_dataset, test_dataset = shorten_ds(n_samples, train_dataset, test_dataset)
     train_custom_dataset = CustomDataset(train_dataset, diffuser, img_dim)
     test_custom_dataset = CustomDataset(test_dataset, diffuser, img_dim)

     # Create data loaders for the custom datasets
     train_loader = torch.utils.data.DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True)
     test_loader = torch.utils.data.DataLoader(test_custom_dataset, batch_size=batch_size, shuffle=False)
     return train_loader, test_loader





def save_generated_data(train_set_file_name, train_loader, test_set_file_name, test_loader):
    with open(train_set_file_name, "wb") as output_file:
        cPickle.dump(train_loader, output_file)
    with open(test_set_file_name, "wb") as output_file:
        cPickle.dump(test_loader, output_file)


def load_generated_data(train_set_file_name, test_set_file_name):
    with open(train_set_file_name, "rb") as input_file:
        train_loader = cPickle.load(input_file)
    with open(test_set_file_name, "rb") as input_file:
        test_loader = cPickle.load(input_file)
    return train_loader, test_loader


def generate_data(train_data, test_data, pic_width, m_patterns, batch_size, patterns):
    """ Get 2 datasets (train & test) of images.
        Return 2 data loaders (train & test) of simulated detector data of the images (GI samples).
        patterns = 'new' will get random light patterns. Else it will use the patterns from the input."""

    if patterns == 'new':
        patterns = define_m_random_patterns(pic_width, m_patterns)
    train_detector_data, test_detector_data = create_gi_data(train_data, test_data, patterns, pic_width)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_detector_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_detector_data, batch_size=batch_size)

    return train_loader, test_loader, patterns


def define_m_random_patterns(pic_width, m):
    """ define the light patterns"""
    patterns = [np.random.rand(pic_width, pic_width) for i in range(m)]
    return patterns


def create_gi_data(train_data, test_data, patterns, pic_width):
    """ Get 2 datasets (train & test) of images and light patterns.
        Return 2 tensors (train & test) of couples of [GI_sample, low_gray_image]"""
    train_detector_data = []
    for sample in train_data:
        image = sample[0].view(pic_width, pic_width)  # tensor
        train_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])

    test_detector_data = []
    for sample in test_data:
        image = sample[0].view(pic_width, pic_width)
        test_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])
    return train_detector_data, test_detector_data


def sample_after_patterns(image, patterns):
    """ Simulate ghost imaging. Get one image and return GI output from the patterns."""
    detector_output = []
    for i, i_pattern in enumerate(patterns):
        image_after_mask = np.array(image) * i_pattern
        detector_output.append(np.float32(sum(sum(image_after_mask))))
    return detector_output

