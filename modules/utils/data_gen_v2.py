import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

IMG_DIR = "/home/jwl/fgvc-aircraft-2013b/data/images/"
IMG_ROOT = "/home/jwl/fgvc-aircraft-2013b/data/"
BBOX_FILE = "/home/jwl/fgvc-aircraft-2013b/data/images_box.txt"

base_transformations = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

train_transformations = transforms.Compose(
    base_transformations
    + [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ]
)

val_transformations = transforms.Compose(base_transformations)


def make_datasets():
    train_dataset = MakeDataset(
        IMG_DIR, IMG_ROOT + "images_family_train.txt", transform=train_transformations
    )
    val_dataset = MakeDataset(
        IMG_DIR, IMG_ROOT + "images_family_val.txt", transform=val_transformations
    )
    test_dataset = MakeDataset(
        IMG_DIR, IMG_ROOT + "images_family_test.txt", transform=val_transformations
    )
    return train_dataset, val_dataset, test_dataset


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
    def __init__(
        self,
        base_dataset,
        noise_info_file,
        N,
        noise_type,
        snr_db=None,
        beta=None,
        noise_density=None,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.beta = beta
        self.noise_density = noise_density
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
            if self.noise_type == "colored":
                image = self.add_colored_noise_to_region(image, bbox)
            elif self.noise_type == "impulse":
                image = self.add_impulse_noise(image, bbox, self.noise_density)

        image = image.crop((0, 0, image.width, image.height - 20))
        image = self.transform(image)

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

    def add_impulse_noise(self, image, bbox, density):
        data = np.array(image)
        x_min, y_min, x_max, y_max = bbox
        region = data[y_min:y_max, x_min:x_max]
        total_pixels = (x_max - x_min) * (y_max - y_min)
        num_pixels = int(total_pixels * density)  # Total no. of pixels to noise

        # Create masks for salt and pepper
        pepper_mask = np.zeros(region.shape[:2], dtype=bool)
        salt_mask = np.zeros(region.shape[:2], dtype=bool)

        indices = np.random.choice(total_pixels, num_pixels, replace=False)
        pepper_indices = indices[: num_pixels // 2]
        salt_indices = indices[num_pixels // 2 :]

        # Set the masks
        pepper_mask.flat[pepper_indices] = True
        salt_mask.flat[salt_indices] = True

        # Apply the noise
        region[pepper_mask] = 0  # Set pepper pixels to black
        region[salt_mask] = 255  # Set salt pixels to white

        # Update the image data
        data[y_min:y_max, x_min:x_max] = region
        return Image.fromarray(data)

    def add_colored_noise_to_region(self, image, bbox):
        data = np.array(image, dtype=float)
        x_min, y_min, x_max, y_max = bbox
        region = data[y_min:y_max, x_min:x_max]

        signal_power = np.mean(region**2)
        snr_linear = 10 ** (self.snr_db / 10)

        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        if self.beta == -2:
            # Generate single-channel noise
            noise = (
                self.spatial_pattern((y_max - y_min, x_max - x_min), self.beta)
                * noise_std
            )
            # Replicate noise across all three channels
            noise = np.stack((noise,) * 3, axis=-1)
            region += noise
        elif self.beta == 0:
            # Generate single-channel Gaussian noise and replicate it across three channels
            noise = np.random.normal(0, noise_std, size=(y_max - y_min, x_max - x_min))
            noise = np.stack((noise,) * 3, axis=-1)
            region += noise

        # Ensure values remain within the correct range
        region = np.clip(region, 0, 255)
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


def make_regular_loaders(batch_size=32):
    train_dataset, val_dataset, test_dataset = make_datasets()
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=True),
    )


def make_noise_loader(
    N, noise_type, snr_db=None, beta=None, noise_density=None, batch_size=32
):
    train_dataset, val_dataset, test_dataset = make_datasets()
    noise_dataset = MakeNoiseDataset(
        test_dataset,
        BBOX_FILE,
        N,
        noise_type,
        snr_db=snr_db,
        beta=beta,
        noise_density=noise_density,
        transform=val_transformations,
    )
    return DataLoader(noise_dataset, batch_size=batch_size, shuffle=True)
