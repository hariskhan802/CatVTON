import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"
RESUME_PATH = "zhengchong/CatVTON"

PERSON_PATH = "resource/test/person.jpg"
CLOTH_PATH = "resource/test/cloth.jpg"
OUTPUT_DIR = "resource/test-output"

WIDTH = 384
HEIGHT = 512
MIXED_PRECISION = "fp16"
CLOTH_TYPE = "upper"
NUM_STEPS = 30
GUIDANCE_SCALE = 2.5
SEED = 42


def image_grid(images):
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    total_width = sum(widths)
    max_height = max(heights)

    grid = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x = 0
    for img in images:
        grid.paste(img, (x, 0))
        x += img.size[0]
    return grid


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU available nahi hai.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading CatVTON checkpoints...")
    repo_path = snapshot_download(repo_id=RESUME_PATH)

    print("Initializing pipeline...")
    pipeline = CatVTONPipeline(
        base_ckpt=BASE_MODEL,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(MIXED_PRECISION),
        device="cuda",
        skip_safety_check=True,
        use_tf32=False,
    )

    print("Initializing automasker...")
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )

    print("Loading images...")
    person_image = Image.open(PERSON_PATH).convert("RGB")
    cloth_image = Image.open(CLOTH_PATH).convert("RGB")

    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))
    cloth_image = resize_and_padding(cloth_image, (WIDTH, HEIGHT))

    print("Generating mask...")
    mask = automasker(person_image, CLOTH_TYPE)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    print("Running try-on inference...")
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )[0]

    print("Saving outputs...")
    mask_preview = vis_mask(person_image, mask)

    person_out = os.path.join(OUTPUT_DIR, "person_resized.png")
    cloth_out = os.path.join(OUTPUT_DIR, "cloth_resized.png")
    mask_out = os.path.join(OUTPUT_DIR, "mask_preview.png")
    result_out = os.path.join(OUTPUT_DIR, "result.png")
    comparison_out = os.path.join(OUTPUT_DIR, "comparison.png")

    person_image.save(person_out)
    cloth_image.save(cloth_out)
    mask_preview.save(mask_out)
    result_image.save(result_out)
    image_grid([person_image, mask_preview, cloth_image, result_image]).save(comparison_out)

    print("\nDONE ✅")
    print("Result:", result_out)
    print("Comparison:", comparison_out)


if __name__ == "__main__":
    main()