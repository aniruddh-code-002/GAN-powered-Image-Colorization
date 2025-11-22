# dataset_loader.py
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

class ImageColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=(128,128), limit=None):
        self.paths = []
        for root, _, files in os.walk(root_dir):
            for fn in files:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.paths.append(os.path.join(root, fn))
        if limit:
            self.paths = self.paths[:limit]
        self.img_size = img_size

        # PIL-based transform (fast if pillow-simd installed)
        self.transform_color = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),  # gives [0,1], CxHxW
        ])
        # grayscale transform will reuse Resize then convert to L
        self.transform_gray = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.LANCZOS),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        color = self.transform_color(img)        # (3,H,W)
        gray = self.transform_gray(img)          # (1,H,W)
        return gray, color

def get_dataloaders(train_dir='data/train', val_dir='data/val',
                    img_size=(128,128), batch_size=64, num_workers=4, limit=None):
    train_ds = ImageColorizationDataset(train_dir, img_size=img_size, limit=limit)
    val_ds = ImageColorizationDataset(val_dir, img_size=img_size, limit=limit)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=max(1, num_workers//2), pin_memory=True)
    return train_loader, val_loader
