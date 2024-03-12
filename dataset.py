import torch
import numpy as np
from pathlib import Path
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


class FFHQ(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
    
    def __getitem__(self, idx):
        image = default_loader(self.root / f"{idx:05d}.png").convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"images": image}
    
    def __len__(self):
        return 70000


class LatentFFHQ(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
    
    def __getitem__(self, idx):
        latent = torch.from_numpy(np.load(self.root / f"{idx:05d}.npy"))

        return latent
    
    def __len__(self):
        return 70000
