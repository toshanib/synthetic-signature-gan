import os
import torch
from torchvision.utils import save_image

from generator_vanilla_gan import Generator

# ================= CONFIG =================
LATENT_DIM = 100
NUM_IMAGES = 25  # You can change this
CHECKPOINT_PATH = "checkpoints/G_final.pth"
OUTPUT_DIR = "generated_signatures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_generator():
    print("Loading trained Generator...")
    G = Generator(latent_dim=LATENT_DIM).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    G.load_state_dict(checkpoint)
    G.eval()

    print("Generator loaded successfully!")
    return G


def generate_signatures(generator, num_images):
    print(f"Generating {num_images} synthetic signatures...")

    # Random latent vectors
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        fake_images = generator(z)

    # Convert from [-1,1] → [0,1] for saving
    fake_images = (fake_images + 1) / 2

    for i in range(num_images):
        save_path = os.path.join(OUTPUT_DIR, f"synthetic_signature_{i+1}.png")
        save_image(fake_images[i], save_path)

    print(f"Saved {num_images} synthetic signatures to '{OUTPUT_DIR}/'")


def main():
    print(f"Using device: {DEVICE}")

    generator = load_generator()
    generate_signatures(generator, NUM_IMAGES)


if __name__ == "__main__":
    main()