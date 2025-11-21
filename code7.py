# %%
from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler
from datetime import datetime
import os
import glob

# --- CONFIGURAÇÃO DO MODELO ---
lora_dir = "./corel_model_lora"

# Tenta achar o arquivo .safetensors mais recente automaticamente
list_of_files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
if list_of_files:
    latest_file = max(list_of_files, key=os.path.getctime)
    lora_name = os.path.basename(latest_file)
    print(f"LoRA encontrado automaticamente: {lora_name}")
else:
    raise FileNotFoundError(
        "Nenhum arquivo .safetensors encontrado na pasta do modelo.")

# --- CONFIGURAÇÃO DE GERAÇÃO ---
# Lista de classes da Corel que você quer testar.
# Devem ser iguais aos nomes das pastas usados no treino.
classes_to_test = ["royal guards", "train", "dessert", "vegetables", "snow", "sunset"]

NUM_IMAGES_PER_CLASS = 2
WIDTH = 512  # Corel geralmente fica melhor em 512 do que 256
HEIGHT = 512
SEED = 42

output_dir = "./generated_images_corel"
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"

print("=" * 50)
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing(slice_size=1)
pipe.enable_vae_tiling()

print(f"\nLoading LoRA: {lora_name}")
pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict=lora_dir,
    weight_name=lora_name,
    adapter_name="corel_lora"
)
pipe.set_adapters(["corel_lora"], adapter_weights=[1.0])
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Loop pelas classes para provar que o modelo único funciona
all_images = []

for class_name in classes_to_test:
    prompt = f"a photo of a {class_name}"
    negative_prompt = "low quality, blur, watermark, distortion, bad anatomy"

    print("\n" + "-" * 50)
    print(f"GENERATING CLASS: {class_name}")
    print(f"Prompt: {prompt}")

    for i in range(NUM_IMAGES_PER_CLASS):
        seed_val = SEED + i
        print(f"  > Image {i+1}/{NUM_IMAGES_PER_CLASS} (Seed: {seed_val})")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device).manual_seed(seed_val),
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=7.5
        ).images[0]

        timestamp = datetime.now().strftime("%H%M%S")
        img_filename = f"{class_name}_{i+1}_{timestamp}.png"
        img_path = os.path.join(output_dir, img_filename)

        image.save(img_path)
        all_images.append(image)
        print(f"    Saved: {img_path}")

        torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("Creating summary grid...")
# Cria um grid se houver imagens suficientes
if len(all_images) > 0:
    rows = len(classes_to_test)
    cols = NUM_IMAGES_PER_CLASS
    grid = make_image_grid(all_images, rows=rows, cols=cols)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f"corel_summary_grid_{timestamp}.png")
    grid.save(grid_path)
    print(f"✓ Grid saved: {grid_path}")

print("✓ GENERATION COMPLETE!")
