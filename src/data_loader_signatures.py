import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SignatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Transform pipeline (IMPORTANT)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
            transforms.Resize((64, 64)),                  # Match GAN input
            transforms.ToTensor(),                        # [0,1]
            transforms.Normalize([0.5], [0.5])            # → [-1,1] (GAN compatible)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("L")  # Force grayscale
        image = self.transform(image)

        return image


def get_dataloader(data_dir, batch_size=32, num_workers=0):
    dataset = SignatureDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Keep 0 for Windows/Python 3.12 stability
        pin_memory=False
    )

    return dataloader