import torch
from data_loader_signatures import get_dataloader

data_dir = "data/signatures/processed"

dataloader = get_dataloader(data_dir, batch_size=16)

# Get one batch
batch = next(iter(dataloader))

print("Batch shape:", batch.shape)
print("Min pixel value:", batch.min().item())
print("Max pixel value:", batch.max().item())