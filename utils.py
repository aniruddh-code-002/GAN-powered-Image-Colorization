# utils.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torchvision.utils import save_image
import os

def compute_metrics_batch(preds, targets):
    # preds, targets: torch tensors (B,3,H,W) in [0,1]
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    psnrs, ssims = [], []
    for p, t in zip(preds, targets):
        # p,t shape: (3,H,W) -> (H,W,3)
        p = np.transpose(p, (1,2,0))
        t = np.transpose(t, (1,2,0))
        psnrs.append(psnr_fn(t, p, data_range=1.0))
        ssims.append(ssim_fn(t, p, channel_axis=-1, data_range=1.0, win_size=3))

    return float(np.mean(psnrs)), float(np.mean(ssims))

def save_sample(generator, gray_batch, color_batch, epoch, out_dir='outputs', device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        if device == 'cpu':
            preds = generator(gray_batch).cpu()
        else:
            preds = generator(gray_batch.to(device)).cpu()

    B = min(4, preds.shape[0])
    for i in range(B):
        # Move all to CPU before concatenation
        g = gray_batch[i].detach().cpu()
        t = color_batch[i].detach().cpu()
        p = preds[i].detach().cpu()

        # If grayscale input, repeat channels to make it 3
        if g.shape[0] == 1:
            g = g.repeat(3, 1, 1)

        # Concatenate side-by-side: grayscale | prediction | ground truth
        out = torch.cat([g, p, t], dim=2)
        save_image(out, os.path.join(out_dir, f'epoch{epoch}_sample{i}.png'), normalize=True)

    generator.train()

