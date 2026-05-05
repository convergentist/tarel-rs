#!/usr/bin/env python3
"""Auto-generated script to trace the logan-rs generator into TorchScript."""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

# tch-rs VarStore::save produces a TorchScript-style archive.
# Load it with weights_only=False (safe here — we generated this file).
state = torch.load("generator.weights.pt", map_location="cpu", weights_only=False)
model = Generator()

# tch-rs saves with flat key names (e.g. "deconv1.weight"),
# while PyTorch expects "net.0.weight" etc. Remap them.
LAYER_MAP = {
    "deconv1": "net.0",  "bn1": "net.1",
    "deconv2": "net.3",  "bn2": "net.4",
    "deconv3": "net.6",  "bn3": "net.7",
    "deconv4": "net.9",  "bn4": "net.10",
    "deconv5": "net.12",
}

remapped = {}
for k, v in state.items():
    prefix = k.split(".")[0]
    suffix = ".".join(k.split(".")[1:])
    if prefix in LAYER_MAP:
        remapped[f"{LAYER_MAP[prefix]}.{suffix}"] = v
    else:
        remapped[k] = v

model.load_state_dict(remapped)
model.eval()

# Trace and save
z = torch.randn(1, 128, 1, 1)
traced = torch.jit.trace(model, z)
traced.save("generator.pt")
print(f"TorchScript model saved to {'generator.pt'}")
