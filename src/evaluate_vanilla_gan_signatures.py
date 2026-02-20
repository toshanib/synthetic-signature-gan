import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import euclidean_distances

from generator_vanilla_gan import Generator

# ================= CONFIG =================
REAL_DATA_DIR = "data/signatures/processed"
CHECKPOINT_PATH = "checkpoints/G_final.pth"
LATENT_DIM = 100
NUM_SYNTHETIC = 100  # Generate same size as real dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform for loading real images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def load_real_images():
    images = []
    for file in os.listdir(REAL_DATA_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(REAL_DATA_DIR, file)
            img = Image.open(img_path).convert("L")
            img = transform(img)
            images.append(img.view(-1).numpy())  # Flatten

    return np.array(images)


def generate_fake_images(generator, num_images):
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        fake_images = generator(z).cpu()

    # Convert [-1,1] → [0,1]
    fake_images = (fake_images + 1) / 2

    fake_list = []
    for img in fake_images:
        fake_list.append(img.view(-1).numpy())  # Flatten

    return np.array(fake_list)


def compute_fid_proxy(real, fake):
    """
    Stable lightweight FID proxy for small grayscale datasets.
    Avoids matrix sqrt instability.
    """
    mu_real = np.mean(real)
    mu_fake = np.mean(fake)

    std_real = np.std(real)
    std_fake = np.std(fake)

    # Simplified FID-style distance
    fid_proxy = (mu_real - mu_fake) ** 2 + (std_real - std_fake) ** 2

    return float(fid_proxy)


def compute_diversity(fake):
    """
    Measures diversity using average pairwise distance
    """
    distances = euclidean_distances(fake, fake)
    diversity = np.mean(distances)
    return float(diversity)


def main():
    print("Loading trained Generator...")
    G = Generator(LATENT_DIM).to(DEVICE)
    G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    G.eval()

    print("Loading real signature dataset...")
    real_images = load_real_images()
    print(f"Loaded {len(real_images)} real images")

    print("Generating synthetic signatures for evaluation...")
    fake_images = generate_fake_images(G, NUM_SYNTHETIC)

    print("\n--- Evaluation Results ---")

    fid_score = compute_fid_proxy(real_images, fake_images)
    diversity_score = compute_diversity(fake_images)

    print(f"FID Proxy Score      : {fid_score:.4f}")
    print(f"Diversity Score      : {diversity_score:.4f}")
    print(f"Real Mean Pixel      : {np.mean(real_images):.4f}")
    print(f"Fake Mean Pixel      : {np.mean(fake_images):.4f}")


if __name__ == "__main__":
    main()