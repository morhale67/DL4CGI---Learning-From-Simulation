import numpy as np
from Load_Data import load_data_mnist
import torch


def build_dataset(batch_size, num_workers, pic_width, n_samples, data_root_medical, data_name):

    if data_name.lower() == 'medical':
        data_set = ImageFolder(root=data_root_medical, transform=transform)
    if data_name.lower() == 'simple_cifar':
        data_set = ImageFolder(root='./data_DSI/GCP_data/simple_cifar', transform=transform)
    elif data_name.lower() == 'cifar' or data_name.lower() == 'cifar10':
        data_set = dset.CIFAR10(root='./data/cifar10', train=True, transform=transform, download=True)
    elif data_name.lower() == 'stl' or data_name.lower() == 'stl-10':
        data_set = datasets.STL10(root='./data/stl-10', split='train', transform=transform, download=True)
    elif data_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((pic_width, pic_width))
        ])
        data_set = dset.MNIST(root='./data', train=True, transform=transform, download=True)

    train_loader, test_loader = create_loader_from_data_set(data_set, n_samples, batch_size, num_workers)
    # save_random_image_from_loader(train_loader, pic_width)
    return train_loader, test_loader


def load_data_mnist(pic_width, batch_size=20, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((pic_width, pic_width)),
        transforms.Grayscale(num_output_channels=1)
    ])
    data_set = dset.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader, test_loader = create_loader_from_data_set(data_set, n_samples, batch_size, num_workers)
    return train_loader, test_loader


def generate_paper_data(batch_size=20, num_workers=0):
    train_data, test_data, pic_width = load_data_mnist()
    patterns = define_m_random_patterns(pic_width)
    train_detector_data, test_detector_data = create_detector_data(train_data, test_data, patterns, pic_width)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_detector_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_detector_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, batch_size


def define_m_random_patterns(pic_width):
    """ define the light patterns"""
    m = pic_width
    patterns = [np.random.rand(m, m) for i in range(m**2)]
    return patterns


def create_detector_data(train_data, test_data, patterns, pic_width):
    """ create couples of [CGI_sample, original_image] """
    train_detector_data = []
    for sample in train_data:
        image = sample[0].view(pic_width, pic_width)
        train_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])

    test_detector_data = []
    for sample in test_data:
        image = sample[0].view(pic_width, pic_width)
        test_detector_data.append([torch.tensor(sample_after_patterns(image, patterns)), torch.flatten(image)])

    return train_detector_data, test_detector_data


def sample_after_patterns(image, patterns):
    """ calculate CGI_sample from original_image"""
    detector_output = []
    for i, i_pattern in enumerate(patterns):
        image_after_mask = np.array(image)*i_pattern
        detector_output.append(np.float32(sum(sum(image_after_mask))))
    return detector_output






