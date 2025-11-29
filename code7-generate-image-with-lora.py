#%%
from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler
from datetime import datetime
import os

lora_name = "lora_stable-diffusion-v1-5_rank4_s800_r512_DDPMScheduler_20251028-185241.safetensors"
lora_dir = "./toys_model"

# Configuration
NUM_IMAGES = 4
WIDTH = 256
HEIGHT = 256
SEED = 12

# Create output directory
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"
print("=" * 50)
print("Loading Stable Diffusion model...")
print("=" * 50)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None  # Remove safety checker to save VRAM
).to(device)

# Enable memory optimizations
pipe.enable_attention_slicing(slice_size=1)
pipe.enable_vae_tiling()
print("✓ Memory optimizations enabled")

print(f"\nLoading LoRA: {lora_name}")
pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict=lora_dir,
    weight_name=lora_name,
    adapter_name="az_lora"
)
pipe.set_adapters(["az_lora"], adapter_weights=[1.0])
print("✓ LoRA loaded successfully")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "a toy bike. macro photo. 3d game asset"
negative_prompt = "low quality, blur, watermark, words, name"

print("\n" + "=" * 50)
print("GENERATION SETTINGS")
print("=" * 50)
print(f"Prompt: {prompt}")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print(f"Number of images: {NUM_IMAGES}")
print(f"Guidance scale: 8.5")
print(f"Seed: {SEED}")
print("=" * 50)

# Generate images ONE AT A TIME for better memory management
images = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i in range(NUM_IMAGES):
    print(f"\n[{i+1}/{NUM_IMAGES}] Generating image...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,  # ONE at a time
        generator=torch.Generator(device).manual_seed(SEED + i),
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=8.5
    ).images[0]
    
    images.append(image)
    
    # Save individual image immediately
    img_path = os.path.join(output_dir, f"toy_bike_{timestamp}_{i+1}.png")
    image.save(img_path)
    print(f"  ✓ Saved: {img_path}")
    
    # Clear cache between generations
    torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("Creating image grid...")
grid = make_image_grid(images, cols=2, rows=2)
grid_path = os.path.join(output_dir, f"toy_bike_grid_{timestamp}.png")
grid.save(grid_path)
print(f"✓ Grid saved: {grid_path}")

# Display in notebook if possible
try:
    from IPython.display import display, Image
    print("\nDisplaying grid in notebook:")
    display(grid)
except ImportError:
    print("\n(Not in Jupyter notebook - check saved files)")
except Exception as e:
    print(f"\n(Could not display: {e})")

# Cleanup
pipe.to("cpu")
torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("✓ GENERATION COMPLETE!")
print("=" * 50)
print(f"Total images generated: {len(images)}")
print(f"Output directory: {output_dir}")
print(f"\nIndividual images:")
for i in range(len(images)):
    print(f"  - toy_bike_{timestamp}_{i+1}.png")
print(f"\nGrid image:")
print(f"  - toy_bike_grid_{timestamp}.png")
print("=" * 50)

# %%
