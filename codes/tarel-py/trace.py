
#!/usr/bin/env python3
"""Trace the logan-rs WGAN-GP generator into a deployable TorchScript model."""
import torch
import torch.nn as nn

LATENT_DIM = 128

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # net.0
            nn.BatchNorm2d(512),                             # net.1
            nn.ReLU(True),                                   # net.2
            nn.ConvTranspose2d(512, 256, 4, 2, 1),          # net.3
            nn.BatchNorm2d(256),                             # net.4
            nn.ReLU(True),                                   # net.5
            nn.ConvTranspose2d(256, 128, 4, 2, 1),          # net.6
            nn.BatchNorm2d(128),                             # net.7
            nn.ReLU(True),                                   # net.8
            nn.ConvTranspose2d(128, 64, 4, 2, 1),           # net.9
            nn.BatchNorm2d(64),                              # net.10
            nn.ReLU(True),                                   # net.11
            nn.ConvTranspose2d(64, 1, 4, 2, 1),             # net.12
            nn.Tanh(),                                       # net.13
        )

    def forward(self, z):
        return self.net(z)

# tch-rs key format: "deconv1|weight" → PyTorch: "net.0.weight"
LAYER_MAP = {
    "deconv1": "net.0",  "bn1": "net.1",
    "deconv2": "net.3",  "bn2": "net.4",
    "deconv3": "net.6",  "bn3": "net.7",
    "deconv4": "net.9",  "bn4": "net.10",
    "deconv5": "net.12",
}

# Load the tch-rs VarStore archive
m = torch.jit.load("generator.weights.pt", map_location="cpu")
tch_state = m.state_dict()

# Remap keys: "deconv1|weight" → "net.0.weight"
remapped = {}
for k, v in tch_state.items():
    prefix, suffix = k.split("|")
    pytorch_key = f"{LAYER_MAP[prefix]}.{suffix}"
    remapped[pytorch_key] = v

# Load into PyTorch model and trace
model = Generator()
model.load_state_dict(remapped)
model.eval()

z = torch.randn(1, LATENT_DIM, 1, 1)
traced = torch.jit.trace(model, z)
traced.save("generator_traced.pt")

# Verify
out = traced(z)
print(f"Traced model saved to generator_traced.pt")
print(f"Output shape: {out.shape}")  # should be [1, 1, 64, 64]
