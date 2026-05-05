mod dataset;
mod export;
mod model;
mod train;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tch::Device;

use train::WganConfig;

#[derive(Parser)]
#[command(name = "logan-rs", about = "WGAN-GP on limit order book recurrence plots")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train the WGAN-GP on a directory of recurrence plot images.
    Train {
        /// Directory containing recurrence plot images (png/jpg).
        #[arg(short, long)]
        data: PathBuf,

        /// Path to save the trained generator weights.
        #[arg(short, long, default_value = "generator.pt")]
        output: PathBuf,

        /// Number of training epochs.
        #[arg(long, default_value_t = 200)]
        epochs: i64,

        /// Batch size.
        #[arg(long, default_value_t = 64)]
        batch_size: i64,

        /// Latent vector dimension.
        #[arg(long, default_value_t = 128)]
        latent_dim: i64,

        /// Image resize target (square).
        #[arg(long, default_value_t = 64)]
        img_size: i64,

        /// Path to save training metrics CSV (e.g. metrics.csv).
        #[arg(long)]
        metrics: Option<String>,
    },

    /// Export trained weights + tracing script for HuggingFace deployment.
    Export {
        /// Path to saved generator weights (.weights.pt produced by train).
        #[arg(short, long)]
        weights: PathBuf,

        /// Output path for the TorchScript model.
        #[arg(short, long, default_value = "generator_traced.pt")]
        output: PathBuf,

        /// Latent vector dimension (must match training).
        #[arg(long, default_value_t = 128)]
        latent_dim: i64,
    },

    /// Generate sample images from a traced TorchScript model.
    Generate {
        /// Path to TorchScript model (.pt).
        #[arg(short, long)]
        model: PathBuf,

        /// Number of samples to generate.
        #[arg(short, long, default_value_t = 16)]
        n: i64,

        /// Output directory for generated images.
        #[arg(short, long, default_value = "samples")]
        output: PathBuf,

        /// Latent vector dimension.
        #[arg(long, default_value_t = 128)]
        latent_dim: i64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::cuda_if_available();
    eprintln!("device: {:?}", device);

    match cli.cmd {
        Cmd::Train {
            data,
            output,
            epochs,
            batch_size,
            latent_dim,
            img_size,
            metrics,
        } => {
            let cfg = WganConfig {
                latent_dim,
                img_size,
                batch_size,
                epochs,
                metrics_path: metrics,
                ..Default::default()
            };

            eprintln!("loading recurrence plots from {}", data.display());
            let dataset = dataset::load_recurrence_plots(&data, cfg.img_size)?;
            eprintln!("loaded {} images", dataset.size()[0]);

            let (vs_g, _vs_c, _metrics) = train::train(&dataset, &cfg, device)?;

            export::export_generator(&vs_g, latent_dim, &output)?;
            eprintln!("done.");
        }

        Cmd::Export {
            weights,
            output,
            latent_dim,
        } => {
            // Re-create VarStore, load weights, then export
            let vs_g = tch::nn::VarStore::new(device);
            let _gen = model::Generator::new(&vs_g.root(), latent_dim);
            // vs_g.load() expects the file saved by vs.save()
            // The train command saves to <output>.weights.pt
            let mut vs_g = vs_g;
            vs_g.load(&weights)?;
            export::export_generator(&vs_g, latent_dim, &output)?;
        }

        Cmd::Generate {
            model: model_path,
            n,
            output,
            latent_dim,
        } => {
            std::fs::create_dir_all(&output)?;
            let model = export::load_generator(&model_path, device)?;
            let samples = export::generate_samples(&model, n, latent_dim, device);

            // Denormalise from [-1, 1] to [0, 255] and save as PNG
            let samples = ((samples + 1.0) * 127.5).clamp(0.0, 255.0).to_kind(tch::Kind::Uint8);

            for i in 0..n {
                let img = samples.get(i); // [1, H, W]
                let path = output.join(format!("sample_{i:04}.png"));
                tch::vision::image::save(&img, &path)?;
                eprintln!("saved {}", path.display());
            }
        }
    }

    Ok(())
}
