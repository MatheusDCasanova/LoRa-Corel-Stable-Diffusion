import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
import os
import gc
from datetime import datetime

# --- SETUP ---
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

def main():
    # 1. Basic Config
    utils.write_basic_config()
    
    torch.backends.cudnn.benchmark = True
    
    output_dir = "corel_model_lora"
    train_data_dir = "./corel"
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    
    # Hyperparameters
    lora_rank = 16
    lora_alpha = 16
    text_encoder_lora_rank = 8
    text_encoder_lora_alpha = 8
    train_text_encoder = True
    
    learning_rate = 1e-4
    text_encoder_lr = 5e-5
    
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08

    resolution = 512
    center_crop = True
    random_flip = True
    
    train_batch_size = 8  

    gradient_accumulation_steps = 1 
    
    num_train_epochs = 50
    lr_scheduler_name = "constant"
    max_grad_norm = 1.0
    use_8bit_adam = True
    enable_xformers = True 

    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, 
        mixed_precision="fp16"
    )
    device = accelerator.device
    print(f"Using device: {device}")

    # 2. Load Models
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    weight_dtype = torch.float16
    
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    ).to(device)
    
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    
    if enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled for UNet")
        except:
            print("⚠ xFormers not installed. Using PyTorch 2.0 SDPA if available.")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    unet.enable_gradient_checkpointing()
    if train_text_encoder:
        text_encoder.gradient_checkpointing_enable()

    # 3. LoRA Config
    unet_lora_config = LoraConfig(
        r=lora_rank, 
        lora_alpha=lora_alpha, 
        init_lora_weights="gaussian", 
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(unet_lora_config)

    if train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r=text_encoder_lora_rank, 
            lora_alpha=text_encoder_lora_alpha, 
            init_lora_weights="gaussian", 
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        text_encoder.add_adapter(text_encoder_lora_config)

    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    if train_text_encoder:
        for param in text_encoder.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    # 4. Dataset & Latent Caching
    print(f"Carregando dataset de: {train_data_dir}")

    dataset = load_dataset("imagefolder", data_dir=train_data_dir)
    dataset_columns = list(dataset["train"].features.keys())
    image_column = "image" if "image" in dataset_columns else dataset_columns[0]
    caption_column = "text" if "text" in dataset_columns else dataset_columns[1]

    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids

    # Cache Latents
    print("⚡ Pre-computing Latents (Caching)...")
    processed_data = []
    
    vae.to(device, dtype=weight_dtype)
    vae.eval()
    
    train_data_raw = dataset["train"]
    
    for i in tqdm(range(len(train_data_raw)), desc="Caching"):
        example = train_data_raw[i]
        image = example[image_column].convert("RGB")
        pixel_values = train_transforms(image).unsqueeze(0).to(device, dtype=weight_dtype)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        caption = example[caption_column] if isinstance(example[caption_column], str) else "a photo of a corel image"
        input_ids = tokenize_captions([caption])[0]
        
        processed_data.append({
            "latents": latents.squeeze(0).cpu(), 
            "input_ids": input_ids
        })

    del vae
    del pipe
    torch.cuda.empty_cache()
    print("✓ Latents cached. VAE removed.")

    class CachedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    def collate_fn(examples):
        latents = torch.stack([example["latents"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"latents": latents, "input_ids": input_ids}

    train_dataset = CachedDataset(processed_data)
    
    # OPTIMIZED DATALOADER
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn, 
        batch_size=train_batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # 5. Optimizer
    unet_lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if train_text_encoder:
        text_encoder_lora_layers = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
        params_to_optimize = [
            {"params": unet_lora_layers, "lr": learning_rate},
            {"params": text_encoder_lora_layers, "lr": text_encoder_lr}
        ]
    else:
        params_to_optimize = unet_lora_layers

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params_to_optimize, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)
        except ImportError:
            optimizer = torch.optim.AdamW(params_to_optimize, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer=optimizer)

    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # 6. Training
    max_train_steps = num_train_epochs * len(train_dataloader)
    print(f"Iniciando treino: {num_train_epochs} épocas, {max_train_steps} passos totais.")
    print(f"Effective Batch Size: {train_batch_size * gradient_accumulation_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Training")

    for epoch in range(num_train_epochs):
        unet.train()
        if train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = batch["latents"].to(device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device)

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

                if train_text_encoder:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(input_ids)[0]

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + (list(text_encoder.parameters()) if train_text_encoder else [])
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

    # 7. Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        text_encoder_lora_state_dict = None
        if train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))

        weight_name = f"lora_corel_v1-5_rank{lora_rank}_{formatted_date}.safetensors"
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir, 
            unet_lora_layers=unet_lora_state_dict, 
            text_encoder_lora_layers=text_encoder_lora_state_dict, 
            safe_serialization=True, 
            weight_name=weight_name
        )
        print(f"\n✓ LoRA saved: {os.path.join(output_dir, weight_name)}")

    accelerator.end_training()

if __name__ == "__main__":
    main()