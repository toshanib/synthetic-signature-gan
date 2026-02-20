import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator
from data_loader_signatures import get_dataloader

# ================= CONFIG =================
DATA_DIR = "data/signatures/processed"
CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"

LATENT_DIM = 100
BATCH_SIZE = 16
EPOCHS = 200  # Spec: 100–200 epochs
LR = 1e-4
BETA1 = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


def save_sample_images(generator, epoch, fixed_noise):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()

    grid = vutils.make_grid(fake_images, normalize=True, nrow=4)
    vutils.save_image(grid, f"{SAMPLES_DIR}/epoch_{epoch}.png")
    generator.train()


def main():
    print(f"Using device: {DEVICE}")

    # DataLoader
    dataloader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)

    # Models
    G = Generator(LATENT_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    # Loss function (Vanilla GAN = BCE)
    criterion = nn.BCELoss()

    # Optimizers (spec required)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Fixed noise for consistent sample visualization
    fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

    print("Starting Training...")

    for epoch in range(1, EPOCHS + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]")

        for real_images in loop:
            real_images = real_images.to(DEVICE)

            # 🔥 Instance noise (stabilizes discriminator)
            noise_strength = 0.02
            real_images = real_images + noise_strength * torch.randn_like(real_images)
            batch_size = real_images.size(0)

            # Real and Fake labels
            real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.8 # Label smoothing
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # =========================
            # Train Discriminator
            # =========================
            optimizer_D.zero_grad()

            # Real loss
            outputs_real = D(real_images)
            d_loss_real = criterion(outputs_real, real_labels)

            # Fake loss
            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = G(noise)
            outputs_fake = D(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # =========================
            # Train Generator
            # =========================
            for _ in range(2):
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_images = G(noise)
                outputs = D(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

            loop.set_postfix({
                "G_Loss": f"{g_loss.item():.4f}",
                "D_Loss": f"{d_loss.item():.4f}"
            })

        # Average losses
        avg_g = g_loss_epoch / len(dataloader)
        avg_d = d_loss_epoch / len(dataloader)

        print(f"\nEpoch [{epoch}/{EPOCHS}] | Avg G Loss: {avg_g:.4f} | Avg D Loss: {avg_d:.4f}")

        # Save samples every 10 epochs (spec visualization requirement)
        if epoch % 10 == 0:
            save_sample_images(G, epoch, fixed_noise)

        # Save checkpoints (spec requirement)
        if epoch % 20 == 0:
            torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_epoch_{epoch}.pth")
            torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_epoch_{epoch}.pth")

    # Final model save (important for inference & mobile app)
    torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_final.pth")
    torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_final.pth")

    print("Training Complete!")
    print("Final models saved in checkpoints/")


if __name__ == "__main__":
    main()