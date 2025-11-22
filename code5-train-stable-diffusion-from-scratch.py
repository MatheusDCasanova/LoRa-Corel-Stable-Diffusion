"""
Stable Diffusion - Latent Space Diffusion Model
================================================
Updated for Corel Dataset & Accelerate
Synced with Improved VAE Architecture

This code trains a diffusion model in the latent space of a pre-trained VAE.
It loads the encoder and decoder from code4-train-vae.py and applies diffusion
to the latent vectors, then uses the decoder to generate final images.

Features:
- Works in VAE latent space (much more efficient)
- Cosine Annealing with Warm Restarts
- Early Stopping with oscillation detection
- Accelerate integration for mixed precision and multi-device support
- EMA for stable generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import deque
import warnings
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse

# ============================================================================
# VAE COMPONENTS (Matching code4-train-vae.py)
# ============================================================================

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

class VAEConfig:
    """Configuration matching code4-train-vae.py"""
    image_size = 128
    image_channels = 3
    latent_dim = 128
    hidden_dims = [64, 128, 256, 512]

def load_vae_components(vae_checkpoint_path, device):
    """Load pre-trained VAE encoder and decoder from code4 checkpoint"""
    print(f"Loading VAE from: {vae_checkpoint_path}")
    
    if not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}. Please run code4 first.")

    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_config = VAEConfig()
    
    encoder = Encoder(vae_config).to(device)
    decoder = Decoder(vae_config).to(device)
    
    state_dict = checkpoint['model_state_dict']
    
    encoder_state = {}
    decoder_state = {}
    
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            encoder_state[new_key] = v
        elif k.startswith('decoder.'):
            new_key = k[8:]
            decoder_state[new_key] = v
    
    encoder.load_state_dict(encoder_state, strict=False)
    decoder.load_state_dict(decoder_state, strict=False)
    
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    
    print(f"✓ VAE loaded successfully!")
    return encoder, decoder, vae_config

# ============================================================================
# DIFFUSION SCHEDULE AND UTILITIES
# ============================================================================

class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.alphas_cumprod = self._cosine_beta_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
            self.alphas = 1.0 - self.betas
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5, window_size=10, oscillation_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        
        self.best_loss = float('inf')
        self.counter = 0
        self.loss_history = deque(maxlen=window_size)
        self.should_stop = False
        
    def __call__(self, loss):
        self.loss_history.append(loss)
        is_best = False
        reason = None
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            reason = f"Loss plateau detected ({self.patience} epochs without improvement)"
            return True, reason, is_best
        
        if len(self.loss_history) == self.window_size:
            mean_loss = np.mean(self.loss_history)
            std_loss = np.std(self.loss_history)
            
            if std_loss < self.oscillation_threshold and abs(loss - mean_loss) < self.oscillation_threshold:
                self.should_stop = True
                reason = f"Loss oscillating around {mean_loss:.4f} (std={std_loss:.6f})"
                return True, reason, is_best
        
        return False, None, is_best

# ============================================================================
# LATENT DIFFUSION MODEL (U-Net for Latent Space)
# ============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class LatentResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_proj = self.time_mlp(time_emb)
        h = h + time_proj
        h = self.block2(h)
        return h + self.residual_proj(x)

class LatentUNet(nn.Module):
    def __init__(self, latent_dim=128, time_dim=256, hidden_dims=[256, 512, 512], dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.dims = [latent_dim] + [hidden_dims[0]] + hidden_dims
        
        self.input_proj = nn.Linear(self.dims[0], self.dims[1])
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.encoder_blocks.append(
                LatentResidualBlock(self.dims[i+1], self.dims[i+2], time_dim, dropout)
            )
        
        mid_dim = self.dims[-1]
        self.middle_block1 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        self.middle_block2 = LatentResidualBlock(mid_dim, mid_dim, time_dim, dropout)
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            curr_dim = self.dims[-(i+1)]
            skip_dim = self.dims[-(i+2)]
            in_dim = curr_dim + curr_dim
            out_dim = skip_dim
            
            self.decoder_blocks.append(
                LatentResidualBlock(in_dim, out_dim, time_dim, dropout)
            )
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.dims[1] * 2, self.dims[1]),
            nn.SiLU(),
            nn.Linear(self.dims[1], self.dims[0])
        )
    
    def forward(self, x, t):
        time_emb = self.time_embed(t)
        h = self.input_proj(x)
        skips = [h]
        
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skips.append(h)
        
        h = self.middle_block1(h, time_emb)
        h = self.middle_block2(h, time_emb)
        
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = block(h, time_emb)
        
        h = torch.cat([h, skips[0]], dim=-1)
        h = self.output_proj(h)
        
        return h

# ============================================================================
# DATASET
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.rglob(ext)))
            self.image_paths.extend(list(self.image_dir.rglob(ext.upper())))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir} with standard extensions.")
        
        print(f"✓ Found {len(self.image_paths)} images in {image_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
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
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, self.image_size, self.image_size))

# ============================================================================
# TRAINING
# ============================================================================

def train_latent_diffusion(
    vae_checkpoint_path,
    image_dir,
    output_dir='latent_diffusion',
    num_epochs=500,
    batch_size=16,
    learning_rate=1e-4,
    image_size=128,
    use_ema=True,
    ema_decay=0.995,
    schedule_type='linear',
    resume_checkpoint=None,
    lr_scheduler='cosine_warm_restarts',
    early_stopping_patience=30,
    seed=42
):
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(seed)
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Load VAE components
    # Note: VAE is kept frozen, so we don't need to wrap it with accelerator.prepare
    # We just move it to the correct device.
    vae_encoder, vae_decoder, vae_config = load_vae_components(vae_checkpoint_path, device)
    latent_dim = vae_config.latent_dim
    
    # Dataset and dataloader
    dataset = ImageDataset(image_dir, image_size)
    if len(dataset) == 0:
        if accelerator.is_main_process:
            print("ERROR: No images found.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize diffusion schedule
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    # Initialize latent diffusion model
    model = LatentUNet(
        latent_dim=latent_dim,
        time_dim=256,
        hidden_dims=[256, 512, 512],
        dropout=0.1
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-5
        )
    
    # Prepare with Accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None
    
    # Early stopping
    early_stopper = EarlyStopping(patience=early_stopping_patience)
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    loss_history = []
    
    if resume_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        if accelerator.is_main_process:
            print(f"✓ Resumed from epoch {start_epoch}")
    
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("TRAINING LATENT DIFFUSION MODEL")
        print("="*80)
        print(f"Dataset size:     {len(dataset)}")
        print(f"Latent dimension: {latent_dim}")
        print(f"Batch size:       {batch_size}")
        print(f"Epochs:           {num_epochs}")
        print("="*80 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, images in enumerate(pbar):
            # Encode images to latent space
            with torch.no_grad():
                # images are already on device
                mu, logvar = vae_encoder(images)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                latents = mu + eps * std
            
            # Sample random timesteps
            t = torch.randint(0, schedule.timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise
            noise = torch.randn_like(latents)
            sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, latents.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, latents.shape)
            noisy_latents = sqrt_alphas_cumprod_t * latents + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Predict noise
            optimizer.zero_grad()
            predicted_noise = model(noisy_latents, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if ema:
                ema.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if scheduler:
            scheduler.step()
        
        # Logging and saving
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            
            should_stop, stop_reason, is_best = early_stopper(avg_loss)
            
            if is_best:
                print(f"✨ New best loss: {avg_loss:.4f}")
                best_loss = avg_loss
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_loss,
                }
                if ema:
                    checkpoint['ema_shadow'] = ema.shadow
                torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
            
            if should_stop:
                print(f"\n⚠️  Early stopping triggered: {stop_reason}")
                break
            
            if (epoch + 1) % 10 == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_loss,
                }
                torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
                
                print(f"Generating samples...")
                if ema:
                    ema.apply_shadow()
                
                generate_samples(
                    unwrapped_model, vae_decoder, schedule, device,
                    os.path.join(output_dir, 'samples', f'epoch_{epoch+1}.png'),
                    latent_dim=latent_dim,
                    num_images=4
                )
                
                if ema:
                    ema.restore()

    if accelerator.is_main_process:
        # Save final model
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title('Training Loss History')
        plt.savefig(os.path.join(output_dir, 'loss_history.png'))
        plt.close()
        
        print(f"\n✅ Training complete! Models saved to {output_dir}")

# ============================================================================
# SAMPLING
# ============================================================================

@torch.no_grad()
def generate_samples(model, decoder, schedule, device, output_path, latent_dim=128, num_images=4):
    model.eval()
    decoder.eval()
    
    # Start from random noise
    latents = torch.randn(num_images, latent_dim).to(device)
    
    # Denoise
    for t in tqdm(reversed(range(schedule.timesteps)), desc="Sampling", total=schedule.timesteps, leave=False):
        t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(latents, t_tensor)
        
        # Update latents (p_sample)
        beta_t = schedule.betas[t]
        sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / schedule.alphas[t])
        posterior_variance_t = schedule.posterior_variance[t]
        
        # Mean
        model_mean = sqrt_recip_alpha_t * (
            latents - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
        )
        
        if t > 0:
            noise = torch.randn_like(latents)
            latents = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            latents = model_mean
            
    # Decode latents to images
    images = decoder(latents)
    
    save_image(images, output_path, nrow=2, normalize=True, value_range=(-1, 1))

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion')
    parser.add_argument('--vae-path', type=str, default='./vae_clean/best_model.pt')
    parser.add_argument('--data-dir', type=str, default='./corel')
    parser.add_argument('--output-dir', type=str, default='./latent_diffusion')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    train_latent_diffusion(
        vae_checkpoint_path=args.vae_path,
        image_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_checkpoint=args.resume
    )

if __name__ == "__main__":
    main()
