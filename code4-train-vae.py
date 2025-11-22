#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean VAE - No k-NN Complexity
================================
Updated for Corel Dataset & Accelerate
IMPROVED ARCHITECTURE: GroupNorm, SiLU, Attention, Low KL

Focuses on what actually works:
- Proper KL control (Low KL for reconstruction focus)
- Perceptual loss for sharpness
- Modern Architecture (GN + SiLU + Attention)
- Simple and fast
- Multi-device support via Accelerate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed

# --- CONFIG ---
class Config:
    data_dir = "./corel"
    output_dir = "./vae_clean"
    image_size = 128
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]
    num_epochs = 300
    batch_size = 16
    learning_rate = 1e-4
    
    # KL control (DRASTICALLY REDUCED for better reconstruction)
    kl_weight_final = 0.00001  # 1e-5
    kl_warmup_epochs = 50
    kl_target = None # Not used with fixed small weight
    
    # Perceptual loss
    use_perceptual = True
    perceptual_weight = 0.1  # Increased slightly
    
    weight_decay = 1e-5
    grad_clip = 1.0
    num_workers = 8
    save_every = 20
    sample_every = 10
    seed = 42
    mixed_precision = "fp16"

config = Config()

# --- DATASET ---
class SimpleDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir} with standard extensions. Checking recursively...")
            
        print(f"✓ Found {len(self.image_paths)} images in {data_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, self.image_size, self.image_size))

# --- MODELS ---
class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = vgg.features[:16].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        return F.mse_loss(x_features, y_features)

class AttnBlock(nn.Module):
    """Self-Attention Block"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)      # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        
        h_ = self.proj_out(h_)
        
        return x + h_

class ResidualBlock(nn.Module):
    """Modern Residual Block with GroupNorm and SiLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = config.image_channels
        
        # Initial conv
        layers.append(nn.Conv2d(in_channels, config.hidden_dims[0], 3, padding=1))
        in_channels = config.hidden_dims[0]
        
        # Downsampling blocks
        for i, h_dim in enumerate(config.hidden_dims):
            # Residual blocks
            layers.append(ResidualBlock(in_channels, h_dim))
            layers.append(ResidualBlock(h_dim, h_dim))
            
            # Downsample (except last if we want, but here we downsample every time)
            # Stride 2 conv for downsampling
            if i < len(config.hidden_dims):
                layers.append(nn.Conv2d(h_dim, h_dim, 3, stride=2, padding=1))
            
            # Attention at lowest resolution (bottleneck)
            if i == len(config.hidden_dims) - 1:
                layers.append(AttnBlock(h_dim))
                
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Calculate final size
        # We did len(hidden_dims) downsamples
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        flatten_dim = self.final_channels * self.final_size * self.final_size
        
        self.fc_mu = nn.Linear(flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, config.latent_dim)
        
        # Init
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -3.0)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        
        self.decoder_input = nn.Linear(config.latent_dim, 
                                       self.final_channels * self.final_size * self.final_size)
        
        layers = []
        reversed_dims = list(reversed(config.hidden_dims))
        
        # Start with attention
        layers.append(AttnBlock(reversed_dims[0]))
        
        for i in range(len(reversed_dims)):
            h_dim = reversed_dims[i]
            next_dim = reversed_dims[i+1] if i < len(reversed_dims) - 1 else reversed_dims[-1]
            
            layers.append(ResidualBlock(h_dim, h_dim))
            layers.append(ResidualBlock(h_dim, h_dim))
            
            # Upsample
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(h_dim, next_dim if i < len(reversed_dims)-1 else h_dim, 3, padding=1))
            
        # Final output
        layers.append(nn.GroupNorm(32, reversed_dims[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(reversed_dims[-1], config.image_channels, 3, padding=1))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.final_channels, self.final_size, self.final_size)
        return self.decoder(h)

class CleanVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        if config.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

# --- LOSS & TRAINING ---
def vae_loss(recon_x, x, mu, logvar, model, beta=1.0):
    # L1 Loss for sharper images
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    
    if model.config.use_perceptual:
        perceptual = model.perceptual_loss(recon_x, x)
        recon_loss = recon_loss + model.config.perceptual_weight * perceptual
    else:
        perceptual = torch.tensor(0.0, device=x.device)
    
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = torch.mean(kl_div)
    
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div, perceptual

def get_kl_weight(epoch, current_kl, config):
    # Fixed small weight is better for reconstruction quality
    return config.kl_weight_final

def train_epoch(model, dataloader, optimizer, config, epoch, kl_history, accelerator):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_perceptual = 0
    num_batches = 0
    
    recent_kl = np.mean(kl_history[-10:]) if len(kl_history) > 0 else 50.0
    current_beta = get_kl_weight(epoch, recent_kl, config)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", disable=not accelerator.is_local_main_process)
    
    for data in pbar:
        optimizer.zero_grad()
        
        recon, mu, logvar = model(data)
        loss, recon_loss, kl_div, perceptual = vae_loss(
            recon, data, mu, logvar, model, beta=current_beta
        )
        
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        total_perceptual += perceptual.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_div.item():.1f}',
            'perc': f'{perceptual.item():.4f}'
        })
    
    avg_kl = total_kl / num_batches
    return (total_loss / num_batches, total_recon / num_batches, 
            avg_kl, current_beta, total_perceptual / num_batches)

@torch.no_grad()
def generate_samples(model, epoch, output_dir, accelerator, num_samples=16):
    model.eval()
    z = torch.randn(num_samples, model.latent_dim, device=accelerator.device)
    samples = model.decode(z)
    
    if accelerator.is_main_process:
        samples_dir = Path(output_dir) / 'samples'
        samples_dir.mkdir(exist_ok=True, parents=True)
        save_image(samples, samples_dir / f'samples_epoch_{epoch}.png', 
                   nrow=4, normalize=True, value_range=(-1, 1))

@torch.no_grad()
def visualize_reconstruction(model, dataloader, epoch, output_dir, accelerator, num_images=8):
    model.eval()
    data = next(iter(dataloader))[:num_images]
    recon, mu, logvar = model(data)
    
    if accelerator.is_main_process:
        comparison = torch.cat([data, recon])
        recon_dir = Path(output_dir) / 'reconstructions'
        recon_dir.mkdir(exist_ok=True, parents=True)
        save_image(comparison, recon_dir / f'reconstruction_epoch_{epoch}.png',
                   nrow=num_images, normalize=True, value_range=(-1, 1))

def plot_losses(losses, output_dir, config):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(losses['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses['recon'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(losses['kl'])
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(losses['perceptual'])
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_losses.png', dpi=150)
    plt.close()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Clean VAE - Corel Dataset')
    parser.add_argument('--data_dir', type=str, default='./corel')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--kl-weight', type=float, default=0.00001)
    parser.add_argument('--kl-warmup', type=int, default=50)
    parser.add_argument('--kl-target', type=float, default=25.0)
    parser.add_argument('--no-perceptual', action='store_true')
    parser.add_argument('--perceptual-weight', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./vae_clean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--sample-every', type=int, default=10)
    
    args = parser.parse_args()
    
    config.data_dir = args.data_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.latent_dim = args.latent_dim
    config.learning_rate = args.learning_rate
    config.kl_weight_final = args.kl_weight
    config.kl_warmup_epochs = args.kl_warmup
    config.kl_target = args.kl_target
    config.use_perceptual = not args.no_perceptual
    config.perceptual_weight = args.perceptual_weight
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.save_every = args.save_every
    config.sample_every = args.sample_every
    
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(config.seed)
    
    if accelerator.is_main_process:
        print("="*80)
        print("CLEAN VAE - IMPROVED ARCHITECTURE")
        print("="*80)
        print(f"Device:          {accelerator.device}")
        print(f"Data:            {config.data_dir}")
        print(f"KL weight:       {config.kl_weight_final}")
        print(f"Perceptual:      {'YES' if config.use_perceptual else 'NO'}")
        print("="*80 + "\n")
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    dataset = SimpleDataset(config.data_dir, config.image_size)
    if len(dataset) == 0:
        if accelerator.is_main_process:
            print("ERROR: No images found.")
        return

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    model = CleanVAE(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    losses = {'total': [], 'recon': [], 'kl': [], 'beta': [], 'perceptual': []}
    kl_history = []
    best_loss = float('inf')
    start_epoch = 0
    
    if args.resume:
        if accelerator.is_main_process:
            print(f"Loading: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        best_loss = checkpoint['loss']
        if accelerator.is_main_process:
            print(f"✓ Resumed from epoch {start_epoch}\n")
    
    if accelerator.is_main_process:
        print("Training...")
    
    for epoch in range(start_epoch, config.num_epochs):
        avg_loss, avg_recon, avg_kl, beta, avg_perc = train_epoch(
            model, dataloader, optimizer, config, epoch, kl_history, accelerator
        )
        
        if accelerator.is_main_process:
            kl_history.append(avg_kl)
            losses['total'].append(avg_loss)
            losses['recon'].append(avg_recon)
            losses['kl'].append(avg_kl)
            losses['beta'].append(beta)
            losses['perceptual'].append(avg_perc)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Loss:  {avg_loss:.4f}")
            print(f"  Recon: {avg_recon:.4f}")
            print(f"  Perc:  {avg_perc:.4f}")
            print(f"  KL:    {avg_kl:.1f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'losses': losses,
                }, Path(config.output_dir) / 'best_model.pt')
                print(f"  ✓ Saved Best")
            
            if (epoch + 1) % config.sample_every == 0:
                generate_samples(accelerator.unwrap_model(model), epoch + 1, config.output_dir, accelerator, 16)
                visualize_reconstruction(accelerator.unwrap_model(model), dataloader, epoch + 1, config.output_dir, accelerator)
                plot_losses(losses, config.output_dir, config)
            
            if (epoch + 1) % config.save_every == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'losses': losses,
                }, Path(config.output_dir) / f'checkpoint_{epoch+1:04d}.pt')
    
    if accelerator.is_main_process:
        print("\nDONE!")

if __name__ == "__main__":
    main()
