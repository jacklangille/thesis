import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

IMG_DIR = "/home/jwl/fgvc-aircraft-2013b/data/images/"
IMG_ROOT = "/home/jwl/fgvc-aircraft-2013b/data/"
BBOX_FILE = "/home/jwl/fgvc-aircraft-2013b/data/images_box.txt"


def make_datasets():
    train_transformations, val_transformations = _transforms()

    images_dir = IMG_DIR
    file_root_dir = IMG_ROOT
    train_dataset = MakeDataset(
        images_dir,
        file_root_dir + "images_family_train.txt",
        transform=train_transformations,
    )
    val_dataset = MakeDataset(
        images_dir,
        file_root_dir + "images_family_val.txt",
        transform=val_transformations,
    )
    test_dataset = MakeDataset(
        images_dir,
        file_root_dir + "images_family_test.txt",
        transform=val_transformations,
    )
    return train_dataset, val_dataset, test_dataset


def _transforms():
    train_transformations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transformations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transformations, val_transformations


class MakeDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform):
        self.images_dir = images_dir
        self.transform = transform
        self.img_labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        idx_counter = 0
        with open(annotations_file, "r") as file:
            for line in file:
                img_name, label = line.strip().split(" ", 1)
                if label not in self.label_to_idx:
                    self.label_to_idx[label] = idx_counter
                    self.idx_to_label[idx_counter] = label
                    idx_counter += 1
                self.img_labels.append((img_name, self.label_to_idx[label]))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label_idx = self.img_labels[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        image = Image.open(img_path)
        image = image.crop((0, 0, image.width, image.height - 20))
        image = self.transform(image)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor


class MakeNoiseDataset(Dataset):
    def __init__(self, base_dataset, noise_info_file, N, snr_db, beta, transform=None):
        self.base_dataset = base_dataset
        self.snr_db = snr_db
        self.beta = beta
        self.transform = transform
        self.noise_regions = {}
        self.indices = np.random.choice(len(base_dataset), N, replace=False)
        with open(noise_info_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                self.noise_regions[parts[0]] = tuple(map(int, parts[1:]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name, label_idx = self.base_dataset.img_labels[self.indices[idx]]
        img_path = os.path.join(self.base_dataset.images_dir, f"{img_name}.jpg")
        image = Image.open(img_path)
        image = image.convert("RGB")

        if img_name in self.noise_regions:
            bbox = self.noise_regions[img_name]
            image = self.add_noise_to_region(image, bbox)

        image = image.crop((0, 0, image.width, image.height - 20))
        image = self.transform(image)

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

    def add_noise_to_region(self, image, bbox):
        data = np.array(image, dtype=float)
        x_min, y_min, x_max, y_max = bbox
        region = data[y_min:y_max, x_min:x_max]

        signal_power = np.mean(region**2)
        snr_linear = 10 ** (self.snr_db / 10)

        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        if self.beta == -2:
            noise = self.spatial_pattern((y_max - y_min, x_max - x_min), self.beta)
            sf = 255
        elif self.beta == -1:
            noise = np.random.randn(y_max - y_min, x_max - x_min)
            sf = 255 / 10

        noise *= noise_std * sf
        for channel in range(3):
            region[:, :, channel] = np.clip(region[:, :, channel] + noise, 0, 255)

        data[y_min:y_max, x_min:x_max] = region.astype(np.uint8)

        return Image.fromarray(data.astype(np.uint8))

    def spatial_pattern(self, dim, beta):
        u = np.fft.fftfreq(dim[1])
        v = np.fft.fftfreq(dim[0])
        u, v = np.meshgrid(u, v)
        f = np.sqrt(u**2 + v**2)
        S_f = np.power(f, beta / 2.0)
        S_f[f == 0] = 1e-10
        phi = np.random.rand(*dim)
        noise_in_freq = S_f * np.exp(2j * np.pi * phi)
        noise = np.fft.ifft2(noise_in_freq).real
        return noise


def make_noise_loader(N, snr_db, batch_size, beta):
    _, val_transformations = _transforms()
    train_dataset, val_dataset, test_dataset = make_datasets()
    bbox_info_file = BBOX_FILE
    noise_dataset = MakeNoiseDataset(
        test_dataset,
        bbox_info_file,
        N,
        snr_db,
        beta,
        transform=val_transformations,
    )
    return DataLoader(noise_dataset, batch_size=batch_size, shuffle=True)


def make_regular_loaders():
    train_dataset, val_dataset, test_dataset = make_datasets()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_loader, val_loader, test_loader
