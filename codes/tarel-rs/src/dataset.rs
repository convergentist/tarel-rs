use anyhow::{Context, Result};
use std::path::Path;
use tch::{vision, Kind, Tensor};

/// Loads all images from `dir` as grayscale, resizes to `size×size`,
/// normalises to [-1, 1], and stacks into a single tensor [N, 1, size, size].
pub fn load_recurrence_plots(dir: &Path, size: i64) -> Result<Tensor> {
    let mut tensors: Vec<Tensor> = Vec::new();

    let entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory: {}", dir.display()))?
        .filter_map(Result::ok)
        .filter(|e| {
            let p = e.path();
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("png" | "jpg" | "jpeg" | "bmp")
            )
        })
        .collect();

    anyhow::ensure!(!entries.is_empty(), "no images found in {}", dir.display());

    for entry in &entries {
        // Load as grayscale → [1, H, W]
        let img = vision::image::load(entry.path())
            .with_context(|| format!("failed to load {}", entry.path().display()))?;

        // If RGB, convert to grayscale by averaging channels
        let gray = if img.size()[0] == 3 {
            img.mean_dim(0, true, Kind::Float)
        } else {
            img.to_kind(Kind::Float)
        };

        // Resize to target dimensions
        let resized = gray
            .unsqueeze(0) // [1, 1, H, W]
            .upsample_bilinear2d([size, size], false, None, None)
            .squeeze_dim(0); // [1, size, size]

        tensors.push(resized);
    }

    // Stack → [N, 1, size, size], normalise from [0, 255] to [-1, 1]
    let dataset = Tensor::stack(&tensors, 0);
    let dataset = dataset / 127.5 - 1.0;

    Ok(dataset)
}
