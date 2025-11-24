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
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# --- CONFIG ---
class Config:
    data_dir = "./corel"
    output_dir = "./dgae_output"
    lora_path = "./corel_model_lora/lora_corel_v1-5_rank16_20251120-234515.safetensors"
    
    image_size = 512
    latent_dim = 128
    sd_embed_dim = 768
    
    hidden_dims = [64, 128, 256, 512]
    
    num_epochs = 50
    batch_size = 1 # Increased from 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    
    # Loss Weights
    alpha_dsm = 1.0
    beta_kl = 0.00001
    eta_lpips = 0.5
    
    grad_clip = 1.0
    num_workers = 8
    save_every = 10
    eval_every = 5
    seed = 42
    mixed_precision = "fp16"

config = Config()

# --- DATASET ---
class CorelDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        self.image_paths = []
        self.labels = []
        
        # Corel dataset usually has format like 0001_xxxx.png where 0001 is class
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        found_files = []
        for ext in extensions:
            found_files.extend(list(self.data_dir.rglob(ext)))
            found_files.extend(list(self.data_dir.rglob(ext.upper())))
            
        for path in found_files:
            self.image_paths.append(path)
            # Extract class from filename (assuming format class_id_image_id.ext)
            try:
                filename = path.stem
                class_id = int(filename.split('_')[0])
                self.labels.append(class_id)
            except:
                self.labels.append(-1) # Unknown class
                
        # Remap labels to 0..N-1
        unique_labels = sorted(list(set(self.labels)))
        self.label_map = {l: i for i, l in enumerate(unique_labels)}
        self.labels = [self.label_map[l] for l in self.labels]
        
        print(f"✓ Found {len(self.image_paths)} images in {data_dir}")
        print(f"✓ Found {len(unique_labels)} classes: {unique_labels}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, self.image_size, self.image_size)), label

# --- ENCODER COMPONENTS ---
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
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*layers)
        
        self.final_channels = config.hidden_dims[-1]
        
        # Global Average Pooling
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
        
        self.encoder = Encoder(config)
        self.projector = Projector(config.latent_dim, config.sd_embed_dim)
        
        print("Loading Stable Diffusion Pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None
        )
        
        lora_dir = os.path.dirname(config.lora_path)
        lora_name = os.path.basename(config.lora_path)
        
        print(f"Loading LoRA: {lora_name} from {lora_dir}")
        try:
            pipe.load_lora_weights(lora_dir, weight_name=lora_name)
            print("✓ LoRA loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load LoRA: {e}")
            
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        
        del pipe
        
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Memory optimizations
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()
        if hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
            
        # Enable xformers if available
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"Note: xformers not enabled: {e}")
        
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
        return self.encode(x)

# --- TRAINING ---
def train_step(batch, model, accelerator, config):
    x, _ = batch # Ignore labels for training
    
    z, mu, logvar = model.encode(x)
    c = model.projector(z)
    
    with torch.no_grad():
        latents = model.vae.encode(x).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
    
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
    
    noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
    
    model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states=c).sample
    
    # DSM Loss
    if model.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif model.scheduler.config.prediction_type == "v_prediction":
        target = model.scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = latents
        
    loss_dsm = F.mse_loss(model_pred, target, reduction="mean")
    
    # KL Loss
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bsz
    
    # LPIPS Loss
    alpha_prod_t = model.scheduler.alphas_cumprod[timesteps]
    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t = alpha_prod_t.flatten().view(bsz, 1, 1, 1)
    beta_prod_t = beta_prod_t.flatten().view(bsz, 1, 1, 1)
    
    if model.scheduler.config.prediction_type == "epsilon":
        pred_latents_0 = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
    elif model.scheduler.config.prediction_type == "v_prediction":
        pred_latents_0 = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
    else:
        print("Did not find prediction type, using model_pred")
        pred_latents_0 = model_pred
        
    pred_latents_0 = pred_latents_0 / model.vae.config.scaling_factor
    pred_imgs = model.vae.decode(pred_latents_0).sample
    pred_imgs = torch.clamp(pred_imgs, -1, 1)
    
    loss_lpips = model.perceptual_loss(pred_imgs, x)
    
    total_loss = (config.alpha_dsm * loss_dsm + 
                  config.beta_kl * loss_kl + 
                  config.eta_lpips * loss_lpips)
                  
    return total_loss, loss_dsm, loss_kl, loss_lpips

def evaluate_clustering(model, dataloader, accelerator):
    model.eval()
    all_latents = []
    all_labels = []
    
    print("Evaluating clustering...")
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            imgs, labels = batch
            # Encode to get mu (latent representation)
            mu, _ = model.encoder(imgs)
            
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # KMeans
    n_classes = len(np.unique(all_labels))
    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(all_latents)
    
    nmi = normalized_mutual_info_score(all_labels, pred_labels)
    ari = adjusted_rand_score(all_labels, pred_labels)
    
    return nmi, ari

def plot_metrics(history, output_dir):
    epochs = [h['epoch'] for h in history]
    nmis = [h['nmi'] for h in history]
    aris = [h['ari'] for h in history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, nmis, label='NMI')
    plt.plot(epochs, aris, label='ARI')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Clustering Metrics over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/metrics.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    config.num_epochs = args.epochs
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    set_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Data
    dataset = CorelDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                            num_workers=config.num_workers, pin_memory=True)
    
    # Model
    model = DGAE(config)
    
    # Optimizer
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    metrics_history = []
    
    print("Starting Training...")
    for epoch in range(config.num_epochs):
        model.train()
        
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                loss, dsm, kl, lpips = train_step(batch, model, accelerator, config)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, config.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'DSM': f'{dsm.item():.4f}',
                    'KL': f'{kl.item():.4f}',
                    'LPIPS': f'{lpips.item():.4f}'
                })
        
        # Evaluation
        if (epoch + 1) % config.eval_every == 0:
            nmi, ari = evaluate_clustering(model, dataloader, accelerator)
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}: NMI={nmi:.4f}, ARI={ari:.4f}")
                metrics_history.append({'epoch': epoch+1, 'nmi': nmi, 'ari': ari})
                plot_metrics(metrics_history, config.output_dir)
        
        # Save
        if accelerator.is_main_process and (epoch + 1) % config.save_every == 0:
            torch.save(accelerator.unwrap_model(model).encoder.state_dict(), 
                       f"{config.output_dir}/encoder_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(model).projector.state_dict(), 
                       f"{config.output_dir}/projector_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
