import io
import os
import sys
import time
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

# 🔥 IMPORTANT: allow backend to import src files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from generator_vanilla_gan import Generator  # adjust if your filename differs

app = FastAPI()

# ===== CONFIG =====
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "G_final.pth")

# ===== LOAD MODEL ONCE (GLOBAL) =====
print("Loading Generator model...")
G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
G.eval()
print("Generator loaded successfully!")

# ===== ROOT TEST =====
@app.get("/")
def root():
    return {"message": "Signature GAN API is running"}

# ===== GENERATE SIGNATURES ENDPOINT =====
@app.get("/generate")
def generate_signatures(n: int = 16):
    # Random seed so images change every click
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)

    z = torch.randn(n, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        fake_images = G(z).cpu()

    grid = make_grid(fake_images, nrow=4, normalize=True)
    img = to_pil_image(grid)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(buffer.getvalue(), media_type="image/png")