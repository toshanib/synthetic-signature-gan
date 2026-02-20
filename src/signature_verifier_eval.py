import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

# ================= CONFIG =================
DATA_DIR = "data/verification"
MODEL_PATH = "checkpoints/signature_verifier.pth"
BATCH_SIZE = 16
IMAGE_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================= LOAD DATA =================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

# ================= MODEL (Same Architecture) =================
class SignatureVerifierCNN(nn.Module):
    def __init__(self):
        super(SignatureVerifierCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = SignatureVerifierCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= EVALUATION =================
TP = 0  # True Genuine correctly classified
TN = 0  # True Forgery correctly rejected
FP = 0  # Forgery accepted as genuine (False Acceptance)
FN = 0  # Genuine rejected as forgery (False Rejection)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)

        outputs = model(images)
        preds = (outputs >= 0.5).float()

        for i in range(len(labels)):
            if labels[i] == 0 and preds[i] == 0:
                TP += 1
            elif labels[i] == 1 and preds[i] == 1:
                TN += 1
            elif labels[i] == 1 and preds[i] == 0:
                FP += 1
            elif labels[i] == 0 and preds[i] == 1:
                FN += 1

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
FAR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Acceptance Rate
FRR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Rejection Rate

print("\n===== Verification Evaluation Results =====")
print(f"Accuracy                : {accuracy:.4f}")
print(f"False Acceptance Rate   : {FAR:.4f}")
print(f"False Rejection Rate    : {FRR:.4f}")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")