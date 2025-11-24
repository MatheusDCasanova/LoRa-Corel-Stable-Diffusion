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
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import copy

# --- CONFIG ---
class Config:
    data_dir = "./corel"
    output_dir = "./dgae_output"
    lora_path = "./corel_model_lora/lora_corel_v1-5_rank16_20251120-234515.safetensors"
    
    image_size = 512  # SD v1.5 standard
    latent_dim = 128  # Encoder latent dim
    sd_embed_dim = 768 # SD v1.5 context dim
    
    hidden_dims = [64, 128, 256, 512] # Encoder dims
    
    num_epochs = 100
    batch_size = 1  # Reduced for debugging
    learning_rate = 1e-4
    
    # Loss Weights
    alpha_dsm = 1.0
    beta_kl = 0.00001
    eta_lpips = 0.5
    
    grad_clip = 1.0
    num_workers = 4
    save_every = 1
    sample_every = 1
    seed = 42
    mixed_precision = "fp16"

config = Config()

# --- DATASET ---
class CorelDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
            self.image_paths.extend(list(self.data_dir.rglob(ext.upper())))
            
        print(f"✓ Found {len(self.image_paths)} images in {data_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, self.image_size, self.image_size))

# --- ENCODER COMPONENTS (From code4) ---
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
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
        in_channels = 3
        
        # Initial conv
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
        
        # Calculate flatten dim: 512 -> 256 -> 128 -> 64 -> 32. 
        # With 4 layers of downsampling: 512 / 2^4 = 32.
        # Channels = 512.
        self.final_size = config.image_size // (2 ** len(config.hidden_dims)) # 32
        self.final_channels = config.hidden_dims[-1] # 512
        flatten_dim = self.final_channels * self.final_size * self.final_size
        
        # We use a Global Average Pooling to reduce parameters and make it robust
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(self.final_channels, config.latent_dim)
        self.fc_logvar = nn.Linear(self.final_channels, config.latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        h = self.gap(h).view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.SiLU(),
            nn.Linear(input_dim * 2, output_dim),
        )
    def forward(self, x):
        # x: (B, latent_dim) -> (B, 1, sd_embed_dim)
        return self.net(x).unsqueeze(1)

# --- PERCEPTUAL LOSS ---
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = vgg.features[:16].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        # x is [-1, 1], convert to [0, 1] then normalize
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        return F.mse_loss(self.feature_extractor(self.normalize(x)), 
                          self.feature_extractor(self.normalize(y)))

# --- DGAE MODEL ---
class DGAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Encoder (Trainable)
        self.encoder = Encoder(config)
        
        # 2. Projector (Trainable)
        self.projector = Projector(config.latent_dim, config.sd_embed_dim)
        
        # 3. Diffusion Components (Loaded via Pipeline as in code7)
        print("Loading Stable Diffusion Pipeline...")
        from diffusers import StableDiffusionPipeline
        
        # Load Pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None
        )
        
        # Load LoRA
        lora_dir = os.path.dirname(config.lora_path)
        lora_name = os.path.basename(config.lora_path)
        
        print(f"Loading LoRA: {lora_name} from {lora_dir}")
        try:
            pipe.load_lora_weights(lora_dir, weight_name=lora_name)
            print("✓ LoRA loaded successfully via Pipeline")
        except Exception as e:
            print(f"Warning: Failed to load LoRA via pipeline: {e}")
            
        # Extract Components
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        
        # Cleanup pipeline to save memory
        del pipe
        
        # Freeze VAE (Always frozen)
        self.vae.requires_grad_(False)
        
        # UNet: Freeze base AND LoRA
        self.unet.requires_grad_(False)
        
        print("✓ Model components extracted. Decoder (UNet+LoRA) frozen.")
        
        # 4. Perceptual Loss
        self.perceptual_loss = PerceptualLoss()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def forward(self, x):
        # This forward is for training step logic
        return self.encode(x)

# --- TRAINING ---
def train_step(batch, model, accelerator, config):
    # 1. Prepare Inputs
    x = batch
    
    # 2. Encode -> Latent z
    z, mu, logvar = model.encode(x)
    
    # 3. Project z -> Condition c
    c = model.projector(z) # (B, 1, 768)
    
    # 4. Prepare Diffusion Targets (VAE Encode x)
    with torch.no_grad():
        # SD VAE expects [-1, 1]
        latents = model.vae.encode(x).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
    
    # 5. Add Noise
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
    
    noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
    
    # 6. Predict (Decoder)
    # UNet expects (B, 4, H, W), t, encoder_hidden_states=(B, Seq, 768)
    # Our c is (B, 1, 768). SD usually takes (B, 77, 768). 
    # We can repeat or just pass 1 token if UNet allows (usually fine with cross-attn).
    # However, SD v1.5 might expect 77. Let's try passing just 1 first.
    
    model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states=c).sample
    
    # 7. Calculate Losses
    
    # A) DSM Loss (MSE)
    if model.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif model.scheduler.config.prediction_type == "v_prediction":
        target = model.scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = latents # sample prediction?
        
    loss_dsm = F.mse_loss(model_pred, target, reduction="mean")
    
    # B) KL Loss
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bsz
    
    # C) LPIPS Loss (Perceptual on x0 prediction)
    # We need to estimate x0 (latents_0) from noisy_latents and model_pred
    # Scheduler step() usually returns prev_sample, but we want pred_original_sample.
    # We can manually compute it based on scheduler type.
    
    # For DDPM/DDIM with epsilon prediction:
    # x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
    alpha_prod_t = model.scheduler.alphas_cumprod[timesteps]
    beta_prod_t = 1 - alpha_prod_t
    
    # Reshape for broadcasting
    alpha_prod_t = alpha_prod_t.flatten().view(bsz, 1, 1, 1)
    beta_prod_t = beta_prod_t.flatten().view(bsz, 1, 1, 1)
    
    if model.scheduler.config.prediction_type == "epsilon":
        pred_latents_0 = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
    elif model.scheduler.config.prediction_type == "v_prediction":
        pred_latents_0 = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
    else:
        pred_latents_0 = model_pred # sample prediction
        
    # Decode predicted latents to image
    # Scale back
    pred_latents_0 = pred_latents_0 / model.vae.config.scaling_factor
    with torch.no_grad(): # VAE decode is heavy, maybe no_grad for memory? 
        # But we need grads for LPIPS to flow back to Encoder?
        # WAIT. The Encoder influenced 'c'. 'c' influenced 'model_pred'. 
        # 'model_pred' influenced 'pred_latents_0'.
        # So yes, we need gradients through VAE decoder? 
        # SD VAE is usually frozen. If VAE is frozen, we can't backprop LPIPS to Encoder 
        # UNLESS we differentiate through VAE.
        # But SD VAE is frozen. 
        # Actually, if we want LPIPS to guide the Encoder, we need the path:
        # Encoder -> z -> c -> UNet -> pred_noise -> pred_latents_0 -> VAE Decoder -> pred_image
        # If UNet and VAE are frozen, gradients CAN flow through them to 'c' and then to Encoder.
        # So we must NOT use no_grad here if we want LPIPS to affect Encoder.
        pass
        
    pred_imgs = model.vae.decode(pred_latents_0).sample
    
    # LPIPS expects [-1, 1]
    # SD VAE output is usually roughly [-1, 1] but not strictly clamped.
    pred_imgs = torch.clamp(pred_imgs, -1, 1)
    
    loss_lpips = model.perceptual_loss(pred_imgs, x)
    
    # Total Loss
    total_loss = (config.alpha_dsm * loss_dsm + 
                  config.beta_kl * loss_kl + 
                  config.eta_lpips * loss_lpips)
                  
    return total_loss, loss_dsm, loss_kl, loss_lpips

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    config.num_epochs = args.epochs
    
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    set_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Data
    dataset = CorelDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                            num_workers=config.num_workers, pin_memory=True)
    
    # Model
    model = DGAE(config)
    
    # Optimizer (Only Encoder and Projector)
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training Loop
    print("Starting Training...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss_avg = 0
        
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                loss, dsm, kl, lpips = train_step(batch, model, accelerator, config)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, config.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss_avg += loss.item()
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'DSM': f'{dsm.item():.4f}',
                    'KL': f'{kl.item():.4f}',
                    'LPIPS': f'{lpips.item():.4f}'
                })
        
        # Save & Sample
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_every == 0:
                torch.save(accelerator.unwrap_model(model).encoder.state_dict(), 
                           f"{config.output_dir}/encoder_epoch_{epoch+1}.pt")
                torch.save(accelerator.unwrap_model(model).projector.state_dict(), 
                           f"{config.output_dir}/projector_epoch_{epoch+1}.pt")
                           
            if (epoch + 1) % config.sample_every == 0:
                # Generate a sample reconstruction
                model.eval()
                with torch.no_grad():
                    # Take first batch
                    sample_img = batch[:4]
                    # Encode
                    z, _, _ = model.encode(sample_img)
                    c = model.projector(z)
                    # Denoise (Full Loop)
                    latent_size = config.image_size // 8
                    latents = torch.randn(4, 4, latent_size, latent_size, device=accelerator.device)
                    # Expand c to match batch size
                    c = c.repeat(4, 1, 1)
                    # Use scheduler to denoise
                    model.scheduler.set_timesteps(50)
                    print("Sampling...")
                    for i, t in enumerate(model.scheduler.timesteps):
                        if i % 10 == 0: print(f"Step {i}/50")
                        # Expand latents for classifier free guidance if needed (not here)
                        model_output = model.unet(latents, t, encoder_hidden_states=c).sample
                        latents = model.scheduler.step(model_output, t, latents).prev_sample
                    print("Sampling done. Decoding...")
                    recon = model.vae.decode(latents / model.vae.config.scaling_factor).sample
                    print("Decoding done. Saving...")
                    
                    # Save
                    comparison = torch.cat([sample_img, recon], dim=0)
                    save_image(comparison, f"{config.output_dir}/sample_epoch_{epoch+1}.png", normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()
