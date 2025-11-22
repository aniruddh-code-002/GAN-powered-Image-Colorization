# train.py
import os
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generator import build_generator
from discriminator import build_discriminator
from dataset_loader import get_dataloaders
from utils import compute_metrics_batch, save_sample
import argparse
from torch import amp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='data/train')
    parser.add_argument('--val-dir', default='data/val')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)  # 0 for Windows safety
    parser.add_argument('--limit', type=int, default=None)     # limit images for quick tests
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build models
    generator = build_generator(
        input_shape=(1, args.img_size, args.img_size),
        base_filters=args.base_filters
    ).to(device)
    discriminator = build_discriminator(
        input_shape=(1, args.img_size, args.img_size),
        base_filters=args.base_filters
    ).to(device)

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Optimizers
    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # AMP scaler
    scaler = amp.GradScaler('cuda',enabled=torch.cuda.is_available())

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,  # set to 0
        limit=args.limit
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    best_val_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for gray, color in epoch_iter:
            gray, color = gray.to(device), color.to(device)
            B = gray.size(0)
            # Dynamically match the discriminator output size
            with torch.no_grad():
                out_shape = discriminator(gray, color).shape  # (B, 1, H', W')
                real_labels = torch.ones(out_shape, device=device)
                fake_labels = torch.zeros_like(real_labels)


            # Training Discriminator 
            with amp.autocast('cuda',enabled=torch.cuda.is_available()):
                fake_color = generator(gray)
                real_out = discriminator(gray, color)
                fake_out = discriminator(gray, fake_color.detach())
                d_loss_real = bce_loss(real_out, real_labels)
                d_loss_fake = bce_loss(fake_out, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_opt.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)

            # Training Generator
            with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                fake_color = generator(gray)
                fake_out = discriminator(gray, fake_color)
                adv_loss = bce_loss(fake_out, real_labels)
                recon_loss = l1_loss(fake_color, color)
                g_loss = adv_loss + 100.0 * recon_loss

            g_opt.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            epoch_iter.set_postfix({
                "g_loss": sum(g_losses)/len(g_losses),
                "d_loss": sum(d_losses)/len(d_losses)
            })

        # End epoch
        avg_g = sum(g_losses)/len(g_losses)
        avg_d = sum(d_losses)/len(d_losses)
        print(f"\nEpoch {epoch} complete | Gen Loss={avg_g:.4f} | Disc Loss={avg_d:.4f}")

        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == 1:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pt")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pt")
            try:
                sample_gray, sample_color = next(iter(val_loader))
                save_sample(generator, sample_gray.to(device), sample_color.to(device), epoch, out_dir='outputs', device=device.type)
            except StopIteration:
                pass

        # --- Validation ---
        generator.eval(); discriminator.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for gray_v, color_v in tqdm(val_loader, desc="Validation", ncols=100):
                gray_v, color_v = gray_v.to(device), color_v.to(device)
                pred_v = generator(gray_v)
                psnr_val, ssim_val = compute_metrics_batch(pred_v, color_v)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

        mean_psnr = sum(psnr_vals)/len(psnr_vals)
        mean_ssim = sum(ssim_vals)/len(ssim_vals)
        print(f"Validation PSNR: {mean_psnr:.4f} | SSIM: {mean_ssim:.4f}")

        if mean_psnr > best_val_psnr:
            best_val_psnr = mean_psnr
            torch.save(generator.state_dict(), "models/generator_best.pt")
            print(f"Saved new best generator at epoch {epoch} with PSNR {mean_psnr:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows
    main()
