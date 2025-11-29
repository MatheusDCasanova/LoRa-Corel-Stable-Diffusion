# import packages
import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler,StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
import gc

from datetime import datetime
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')   

# train code 
def main():
    utils.write_basic_config()

    # hyperparameters - OPTIMIZED FOR 2080 (8GB) WITH TEXT ENCODER LORA
    output_dir                      = "."
    pretrained_model_name_or_path   = "runwayml/stable-diffusion-v1-5"
    lora_rank                       = 4
    lora_alpha                      = 4
    text_encoder_lora_rank          = 4     # NEW: Text encoder LoRA rank
    text_encoder_lora_alpha         = 4     # NEW: Text encoder LoRA alpha
    train_text_encoder              = True  # NEW: Enable text encoder training
    learning_rate                   = 1e-4
    text_encoder_lr                 = 5e-5  # NEW: Slightly lower LR for text encoder
    adam_beta1, adam_beta2          = 0.9, 0.999
    adam_weight_decay               = 1e-2
    adam_epsilon                    = 1e-08
    dataset_name                    = None
    train_data_dir                  = "./training_data/toys"
    top_rows                        = 4
    output_dir                      = "toys_model"
    resolution                      = 512
    center_crop                     = True
    random_flip                     = True
    train_batch_size                = 1
    gradient_accumulation_steps     = 4
    num_train_epochs                = 200
    lr_scheduler_name               = "constant"
    max_grad_norm                   = 1.0
    diffusion_scheduler             = DDPMScheduler
    enable_xformers                 = True
    use_8bit_adam                   = True

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps
        , mixed_precision           = "fp16" 
    )
    device      = accelerator.device

    # Load scheduler, tokenizer and unet models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = weight_dtype
    ).to(device)
    tokenizer,text_encoder, vae, unet   = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()
    if train_text_encoder:
        text_encoder.gradient_checkpointing_enable()
    
    # Enable memory-efficient attention
    if enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled for UNet")
        except:
            print("⚠ xFormers not available, using default attention")

    # freeze parameters initially
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Move VAE to CPU (will move to GPU only when encoding)
    vae.to('cpu')
    torch.cuda.empty_cache()

    # Configure UNet LoRA
    unet_lora_config = LoraConfig(
        r                     = lora_rank
        , lora_alpha          = lora_alpha
        , init_lora_weights   = "gaussian"
        , target_modules      = ["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Add UNet adapter
    unet.add_adapter(unet_lora_config)
    
    # NEW: Configure and add text encoder LoRA
    if train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r                     = text_encoder_lora_rank
            , lora_alpha          = text_encoder_lora_alpha
            , init_lora_weights   = "gaussian"
            , target_modules      = ["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        text_encoder.add_adapter(text_encoder_lora_config)
        print("✓ Text encoder LoRA enabled")
    
    # Upcast trainable parameters to fp32
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    if train_text_encoder:
        for param in text_encoder.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
    
    # Load dataset
    if dataset_name:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(
            "imagefolder"
            , data_dir = train_data_dir
        )
    
    train_data = dataset["train"]
    dataset["train"] = train_data.select(range(top_rows))
    print(dataset["train"])

    # Preprocessing
    dataset_columns = list(dataset["train"].features.keys())
    image_column, caption_column = dataset_columns[0],dataset_columns[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution
                , interpolation=transforms.InterpolationMode.BILINEAR
            )
            , transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
            , transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)
            , transforms.ToTensor()
            , transforms.Normalize([0.5], [0.5])
        ]
    )

    def preprocess_train(examples):
        images                      = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"]    = [train_transforms(image) for image in images]
        examples["input_ids"]       = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values    = torch.stack([example["pixel_values"] for example in examples])
        pixel_values    = pixel_values.to(memory_format = torch.contiguous_format).float()
        input_ids       = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset
        , shuffle       = True
        , collate_fn    = collate_fn
        , batch_size    = train_batch_size
        , num_workers   = 0
    )

    print("Data Size:",len(train_dataloader))

    num_update_steps_per_epoch  = math.ceil(len(train_dataloader) / train_batch_size)
    max_train_steps             = num_train_epochs * num_update_steps_per_epoch

    # NEW: Collect trainable parameters from both UNet and text encoder
    unet_lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    if train_text_encoder:
        text_encoder_lora_layers = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
        # Use different learning rates for UNet and text encoder
        params_to_optimize = [
            {"params": unet_lora_layers, "lr": learning_rate},
            {"params": text_encoder_lora_layers, "lr": text_encoder_lr}
        ]
    else:
        params_to_optimize = unet_lora_layers
    
    # Initialize optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            # When using parameter groups, don't pass lr to constructor
            if train_text_encoder:
                optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize  # Has lr specified per group
                    , betas         = (adam_beta1, adam_beta2)
                    , weight_decay  = adam_weight_decay
                    , eps           = adam_epsilon
                )
            else:
                optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize
                    , lr            = learning_rate
                    , betas         = (adam_beta1, adam_beta2)
                    , weight_decay  = adam_weight_decay
                    , eps           = adam_epsilon
                )
            print("✓ Using 8-bit AdamW")
        except ImportError:
            print("⚠ bitsandbytes not available, using regular AdamW")
            print("  Install with: pip install bitsandbytes")
            if train_text_encoder:
                optimizer = torch.optim.AdamW(
                    params_to_optimize  # Has lr specified per group
                    , betas         = (adam_beta1, adam_beta2)
                    , weight_decay  = adam_weight_decay
                    , eps           = adam_epsilon
                )
            else:
                optimizer = torch.optim.AdamW(
                    params_to_optimize
                    , lr            = learning_rate
                    , betas         = (adam_beta1, adam_beta2)
                    , weight_decay  = adam_weight_decay
                    , eps           = adam_epsilon
                )
    else:
        if train_text_encoder:
            optimizer = torch.optim.AdamW(
                params_to_optimize  # Has lr specified per group
                , betas         = (adam_beta1, adam_beta2)
                , weight_decay  = adam_weight_decay
                , eps           = adam_epsilon
            )
        else:
            optimizer = torch.optim.AdamW(
                params_to_optimize
                , lr            = learning_rate
                , betas         = (adam_beta1, adam_beta2)
                , weight_decay  = adam_weight_decay
                , eps           = adam_epsilon
            )

    lr_scheduler = get_scheduler(
        lr_scheduler_name
        , optimizer             = optimizer
    )

    # Prepare everything with accelerator
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    max_train_steps = num_train_epochs*len(train_dataloader)
    progress_bar = tqdm(
        range(0, max_train_steps)
        , initial   = 0
        , desc      = "Steps"
        , disable   = not accelerator.is_local_main_process,
    )

    # Training loop
    for epoch in range(num_train_epochs):
        unet.train()
        if train_text_encoder:
            text_encoder.train()
        
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Move VAE to GPU for encoding
                vae.to(device)
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                # Move VAE back to CPU
                vae.to('cpu')
                torch.cuda.empty_cache()

                # Sample noise
                noise = torch.randn_like(latents)

                # Sample timesteps
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    low         = 0
                    , high      = noise_scheduler.config.num_train_timesteps
                    , size      = (batch_size,)
                    , device    = latents.device
                )
                timesteps = timesteps.long()

                # Get text embeddings (with gradient if training text encoder)
                if train_text_encoder:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if train_text_encoder:
                        params_to_clip = unet_lora_layers + text_encoder_lora_layers
                    else:
                        params_to_clip = unet_lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    train_loss = 0.0
                
                logs = {"epoch": epoch,"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
            
            # Periodic cleanup
            if step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        if train_text_encoder:
            text_encoder = text_encoder.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        # NEW: Also save text encoder LoRA if trained
        text_encoder_lora_state_dict = None
        if train_text_encoder:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_text_encoder)
            )

        weight_name = f"lora_{pretrained_model_name_or_path.split('/')[-1]}_rank{lora_rank}_s{max_train_steps}_r{resolution}_{diffusion_scheduler.__name__}_{formatted_date}.safetensors"   
        
        StableDiffusionPipeline.save_lora_weights(
            save_directory          = output_dir
            , unet_lora_layers      = unet_lora_state_dict
            , text_encoder_lora_layers = text_encoder_lora_state_dict  # NEW: Include text encoder
            , safe_serialization    = True
            , weight_name           = weight_name
        )
        
        print(f"\n✓ LoRA saved: {weight_name}")
        if train_text_encoder:
            print("  Includes: UNet LoRA + Text Encoder LoRA")
        else:
            print("  Includes: UNet LoRA only")

    accelerator.end_training()

if __name__ == "__main__":
    main()
