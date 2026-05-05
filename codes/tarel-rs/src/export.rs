use anyhow::{Context, Result};
use std::path::Path;
use tch::{CModule, Device, Kind, Tensor};

/// Export the trained generator as a TorchScript module via tracing.
///
/// The resulting `.pt` file can be loaded in Python:
/// ```python
/// import torch
/// model = torch.jit.load("generator.pt")
/// z = torch.randn(1, 128, 1, 1)
/// fake = model(z)
/// ```
pub fn export_generator(
    vs: &tch::nn::VarStore,
    latent_dim: i64,
    output_path: &Path,
) -> Result<()> {
    // Save the raw weights first (recoverable from Rust)
    let weights_path = output_path.with_extension("weights.pt");
    vs.save(&weights_path)
        .with_context(|| format!("failed to save weights to {}", weights_path.display()))?;

    eprintln!("weights saved to {}", weights_path.display());

    // For TorchScript tracing, we need to go through the CModule API.
    // The practical approach: save weights, then use a small Python script to
    // trace and re-export. We generate that script alongside the weights.
    let script_path = output_path.with_extension("trace.py");
    let script = format!(
        r#"#!/usr/bin/env python3
"""Auto-generated script to trace the logan-rs generator into TorchScript."""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim={latent_dim}):
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
state = torch.load("{weights}", map_location="cpu", weights_only=False)
model = Generator()

# tch-rs saves with flat key names (e.g. "deconv1.weight"),
# while PyTorch expects "net.0.weight" etc. Remap them.
LAYER_MAP = {{
    "deconv1": "net.0",  "bn1": "net.1",
    "deconv2": "net.3",  "bn2": "net.4",
    "deconv3": "net.6",  "bn3": "net.7",
    "deconv4": "net.9",  "bn4": "net.10",
    "deconv5": "net.12",
}}

remapped = {{}}
for k, v in state.items():
    prefix = k.split(".")[0]
    suffix = ".".join(k.split(".")[1:])
    if prefix in LAYER_MAP:
        remapped[f"{{LAYER_MAP[prefix]}}.{{suffix}}"] = v
    else:
        remapped[k] = v

model.load_state_dict(remapped)
model.eval()

# Trace and save
z = torch.randn(1, {latent_dim}, 1, 1)
traced = torch.jit.trace(model, z)
traced.save("{output}")
print(f"TorchScript model saved to {{'{output}'}}")
"#,
        latent_dim = latent_dim,
        weights = weights_path.display(),
        output = output_path.display(),
    );

    std::fs::write(&script_path, script)
        .with_context(|| format!("failed to write trace script to {}", script_path.display()))?;

    eprintln!(
        "trace script written to {}\nrun: python3 {}",
        script_path.display(),
        script_path.display()
    );

    Ok(())
}

/// Load a previously exported TorchScript generator for inference.
pub fn load_generator(path: &Path, device: Device) -> Result<CModule> {
    let model = CModule::load_on_device(path, device)
        .with_context(|| format!("failed to load TorchScript model from {}", path.display()))?;
    Ok(model)
}

/// Generate `n` samples from a traced generator.
pub fn generate_samples(model: &CModule, n: i64, latent_dim: i64, device: Device) -> Tensor {
    let z = Tensor::randn([n, latent_dim, 1, 1], (Kind::Float, device));
    model.forward_ts(&[z]).expect("forward pass failed")
}
