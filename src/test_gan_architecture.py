import torch
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
batch_size = 16

# Initialize models
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

# Generate fake noise
z = torch.randn(batch_size, latent_dim).to(device)

# Generate fake signatures
fake_images = G(z)
print("Generator output shape:", fake_images.shape)

# Pass through discriminator
validity = D(fake_images)
print("Discriminator output shape:", validity.shape)