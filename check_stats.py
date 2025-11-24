
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os
from accelerate import Accelerator

# --- COPY OF CLASSES FROM code4-train-vae.py ---

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
    kl_weight_final = 0.00001
    kl_warmup_epochs = 50
    kl_target = None
    use_perceptual = True
    perceptual_weight = 0.1
    weight_decay = 1e-5
    grad_clip = 1.0
    num_workers = 8
    save_every = 20
    sample_every = 10
    seed = 42
    mixed_precision = "fp16"

class SimpleDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(image)
        except: return torch.zeros((3, self.image_size, self.image_size))

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k) * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        return x + self.proj_out(h_)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        in_channels = config.image_channels
        layers.append(nn.Conv2d(in_channels, config.hidden_dims[0], 3, padding=1))
        in_channels = config.hidden_dims[0]
        for i, h_dim in enumerate(config.hidden_dims):
            layers.append(ResidualBlock(in_channels, h_dim))
            layers.append(ResidualBlock(h_dim, h_dim))
            if i < len(config.hidden_dims):
                layers.append(nn.Conv2d(h_dim, h_dim, 3, stride=2, padding=1))
            if i == len(config.hidden_dims) - 1:
                layers.append(AttnBlock(h_dim))
            in_channels = h_dim
        self.encoder = nn.Sequential(*layers)
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        flatten_dim = self.final_channels * self.final_size * self.final_size
        self.fc_mu = nn.Linear(flatten_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, config.latent_dim)
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10, max=2)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_size = config.image_size // (2 ** len(config.hidden_dims))
        self.final_channels = config.hidden_dims[-1]
        self.decoder_input = nn.Linear(config.latent_dim, self.final_channels * self.final_size * self.final_size)
        layers = []
        reversed_dims = list(reversed(config.hidden_dims))
        layers.append(AttnBlock(reversed_dims[0]))
        for i in range(len(reversed_dims)):
            h_dim = reversed_dims[i]
            next_dim = reversed_dims[i+1] if i < len(reversed_dims) - 1 else reversed_dims[-1]
            layers.append(ResidualBlock(h_dim, h_dim))
            layers.append(ResidualBlock(h_dim, h_dim))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(h_dim, next_dim if i < len(reversed_dims)-1 else h_dim, 3, padding=1))
        layers.append(nn.GroupNorm(32, reversed_dims[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(reversed_dims[-1], config.image_channels, 3, padding=1))
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)
    def forward(self, z):
        h = self.decoder_input(z).view(-1, self.final_channels, self.final_size, self.final_size)
        return self.decoder(h)

class CleanVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar

# --- MAIN SCRIPT ---

def main():
    accelerator = Accelerator()
    device = accelerator.device
    config = Config()
    
    print(f"Loading dataset from {config.data_dir}")
    dataset = SimpleDataset(config.data_dir, config.image_size)
    if len(dataset) == 0:
        print("No images found!")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = CleanVAE(config)
    checkpoint_path = os.path.join(config.output_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    all_means = []
    all_stds = []
    
    print("Computing latent statistics...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            
            all_means.append(z.mean().item())
            all_stds.append(z.std().item())
            
    mean_val = np.mean(all_means)
    std_val = np.mean(all_stds)
    
    print(f"\nRESULTS:")
    print(f"Average Mean of Latents: {mean_val:.6f}")
    print(f"Average Std of Latents:  {std_val:.6f}")
    print(f"Suggested Scale Factor:  {1.0/std_val:.6f}")

if __name__ == "__main__":
    main()
