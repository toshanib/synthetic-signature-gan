import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
DATA_DIR = "data/verification"
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0002
IMAGE_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================= DATASET =================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)  # fixed seed
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total samples: {len(dataset)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ================= CNN VERIFIER MODEL =================
class SignatureVerifierCNN(nn.Module):
    def __init__(self):
        super(SignatureVerifierCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 64x64 → 32x32
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32 → 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1), # 16x16 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SignatureVerifierCNN().to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= TRAINING LOOP =================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ================= SAVE MODEL =================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/signature_verifier.pth")
print("Verifier model saved to checkpoints/signature_verifier.pth")