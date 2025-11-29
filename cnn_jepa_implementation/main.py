import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
from cnn_jepa import CNNJEPA
from masking import MaskCollator

# Configuration
BATCH_SIZE = 32
EPOCHS = 600
LR = 0.01
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1  # Shortened for demo
MOMENTUM_START = 0.996
MOMENTUM_END = 1.0
IMAGE_SIZE = 224
PATCH_SIZE = 32

import os
import glob
from PIL import Image
import torchvision.transforms as transforms

class CorelDataset(Dataset):
    def __init__(self, root_dirs, use_augmented=False, transform=None):
        # root_dir lista de pastas com as imagens
        self.root_dirs = root_dirs
        self.image_paths = glob.glob(os.path.join(root_dirs[0], "*.png"))
        if use_augmented:
            self.image_paths.extend(glob.glob(os.path.join(root_dirs[1], "*.png")))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random tensor or skip (simplified for stability)
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

def get_lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def get_momentum_schedule(step, total_steps, start_m, end_m):
    return start_m + (end_m - start_m) * (step / total_steps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_augmented', action='store_true', help='Use augmented dataset')
    args = parser.parse_args()
    use_augmented = args.use_augmented

    print("use_augmented", use_augmented)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to 224x224
        transforms.ToTensor(),
    ])
    

    dataset_path = "../corel" 
    augmented_path = "../generated_images_corel"
    if not os.path.exists(dataset_path):
        print("Corel dataset not found in parent directory. Using absolute path.")
        dataset_path = "/home/matheuscasanova/workspace/LoRa-Corel-Stable-Diffusion/corel"
        augmented_path = "/home/matheuscasanova/workspace/LoRa-Corel-Stable-Diffusion/generated_images_corel"
    
    dataset = CorelDataset(root_dirs=[dataset_path, augmented_path], use_augmented=use_augmented, transform=transform)

    print("Dataset size:", len(dataset))
    print("Using augmented dataset:", use_augmented)

    
    collator = MaskCollator(input_size=(IMAGE_SIZE, IMAGE_SIZE), patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=8, shuffle=True, drop_last=True)
    
    # Model
    model = CNNJEPA().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Training Loop
    total_steps = len(dataloader) * EPOCHS
    global_step = 0

    save_path = "best_cnn_jepa" if not use_augmented else "best_cnn_jepa_augmented"
    
    best_loss = float('inf')
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (images, masks, _) in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Update LR
            lr = get_lr_schedule(global_step, total_steps, WARMUP_EPOCHS * len(dataloader), LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            # Update Momentum
            momentum = get_momentum_schedule(global_step, total_steps, MOMENTUM_START, MOMENTUM_END)
            
            # Forward
            optimizer.zero_grad()
            loss = model(images, masks)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # EMA Update
            model.update_target_encoder(momentum)
            
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(i+1):.4f}',
                'lr': f'{lr:.6f}',
                'momentum': f'{momentum:.6f}'
            })
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"cnn_jepa_implementation/{save_path}.pth")
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
