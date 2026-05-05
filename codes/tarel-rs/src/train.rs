use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::model::{Critic, Generator};

/// WGAN-GP hyperparameters.
pub struct WganConfig {
    pub latent_dim: i64,
    pub img_size: i64,
    pub batch_size: i64,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub lambda_gp: f64,
    pub n_critic: i64,
    pub epochs: i64,
    /// Optional path to write training metrics CSV.
    pub metrics_path: Option<String>,
}

impl Default for WganConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            img_size: 64,
            batch_size: 64,
            lr: 1e-4,
            beta1: 0.0,
            beta2: 0.9,
            lambda_gp: 10.0,
            n_critic: 5,
            epochs: 200,
            metrics_path: None,
        }
    }
}

/// Per-epoch training metrics snapshot.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: i64,
    /// Mean critic loss (Wasserstein + GP).
    pub critic_loss: f64,
    /// Mean generator loss.
    pub generator_loss: f64,
    /// Mean gradient penalty value (before lambda scaling).
    pub gradient_penalty: f64,
    /// Wasserstein distance estimate: E[C(real)] - E[C(fake)].
    pub wasserstein_dist: f64,
    /// Mean critic score on real data.
    pub critic_real: f64,
    /// Mean critic score on fake data.
    pub critic_fake: f64,
    /// Wall-clock seconds for this epoch.
    pub epoch_secs: f64,
}

/// Full training run metrics.
pub struct TrainingMetrics {
    pub epochs: Vec<EpochMetrics>,
    pub total_secs: f64,
}

impl TrainingMetrics {
    fn new() -> Self {
        Self {
            epochs: Vec::new(),
            total_secs: 0.0,
        }
    }

    /// Write all epoch metrics to a CSV file.
    pub fn save_csv(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(
            f,
            "epoch,critic_loss,generator_loss,gradient_penalty,wasserstein_dist,critic_real,critic_fake,epoch_secs"
        )?;
        for m in &self.epochs {
            writeln!(
                f,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3}",
                m.epoch,
                m.critic_loss,
                m.generator_loss,
                m.gradient_penalty,
                m.wasserstein_dist,
                m.critic_real,
                m.critic_fake,
                m.epoch_secs,
            )?;
        }
        eprintln!("metrics saved to {}", path.display());
        Ok(())
    }
}

/// Gradient penalty: enforces Lipschitz constraint on the critic.
fn gradient_penalty(
    critic: &Critic,
    real: &Tensor,
    fake: &Tensor,
    device: Device,
) -> Tensor {
    let batch = real.size()[0];
    let alpha = Tensor::rand([batch, 1, 1, 1], (Kind::Float, device));
    let real_part: Tensor = &alpha * real;
    let fake_part: Tensor = (1.0 - &alpha) * fake;
    let interpolated = (real_part + fake_part).set_requires_grad(true);

    let scores = critic.forward_t(&interpolated);
    let grads = Tensor::run_backward(&[scores.sum(Kind::Float)], &[&interpolated], true, true);
    let grad = &grads[0];

    let grad_norm = grad.view([batch, -1]).norm_scalaropt_dim(2, &[1], false);
    ((grad_norm - 1.0).pow_tensor_scalar(2)).mean(Kind::Float)
}

/// Sample a noise vector z ~ N(0, 1).
fn sample_z(batch: i64, latent_dim: i64, device: Device) -> Tensor {
    Tensor::randn([batch, latent_dim, 1, 1], (Kind::Float, device))
}

/// Run the full WGAN-GP training loop. Returns the trained VarStores and metrics.
pub fn train(
    dataset: &Tensor,
    cfg: &WganConfig,
    device: Device,
) -> Result<(nn::VarStore, nn::VarStore, TrainingMetrics)> {
    let vs_g = nn::VarStore::new(device);
    let vs_c = nn::VarStore::new(device);

    let generator = Generator::new(&vs_g.root(), cfg.latent_dim);
    let crt = Critic::new(&vs_c.root());

    let mut opt_g = nn::Adam::default()
        .beta1(cfg.beta1)
        .beta2(cfg.beta2)
        .build(&vs_g, cfg.lr)?;

    let mut opt_c = nn::Adam::default()
        .beta1(cfg.beta1)
        .beta2(cfg.beta2)
        .build(&vs_c, cfg.lr)?;

    let n_samples = dataset.size()[0];
    let mut metrics = TrainingMetrics::new();
    let train_start = Instant::now();

    for epoch in 0..cfg.epochs {
        let epoch_start = Instant::now();

        // Shuffle indices each epoch
        let perm = Tensor::randperm(n_samples, (Kind::Int64, device));
        let n_batches = n_samples / cfg.batch_size;

        let mut c_loss_acc = 0f64;
        let mut g_loss_acc = 0f64;
        let mut gp_acc = 0f64;
        let mut c_real_acc = 0f64;
        let mut c_fake_acc = 0f64;

        for i in 0..n_batches {
            let idx = perm
                .narrow(0, i * cfg.batch_size, cfg.batch_size)
                .to_device(device);
            let real = dataset.index_select(0, &idx).to_device(device);

            // --- Train critic ---
            for _ in 0..cfg.n_critic {
                let z = sample_z(cfg.batch_size, cfg.latent_dim, device);
                let fake = generator.forward_t(&z).detach();

                let c_real = crt.forward_t(&real).mean(Kind::Float);
                let c_fake = crt.forward_t(&fake).mean(Kind::Float);
                let gp = gradient_penalty(&crt, &real, &fake, device);

                // Wasserstein loss + gradient penalty
                let c_loss = &c_fake - &c_real + cfg.lambda_gp * &gp;

                opt_c.zero_grad();
                c_loss.backward();
                opt_c.step();

                c_loss_acc += f64::try_from(&c_loss).unwrap_or(0.0);
                gp_acc += f64::try_from(&gp).unwrap_or(0.0);
                c_real_acc += f64::try_from(&c_real).unwrap_or(0.0);
                c_fake_acc += f64::try_from(&c_fake).unwrap_or(0.0);
            }

            // --- Train generator ---
            let z = sample_z(cfg.batch_size, cfg.latent_dim, device);
            let fake = generator.forward_t(&z);
            let g_loss = -crt.forward_t(&fake).mean(Kind::Float);

            opt_g.zero_grad();
            g_loss.backward();
            opt_g.step();

            g_loss_acc += f64::try_from(&g_loss).unwrap_or(0.0);
        }

        let epoch_secs = epoch_start.elapsed().as_secs_f64();

        if n_batches > 0 {
            let critic_steps = (n_batches * cfg.n_critic) as f64;
            let gen_steps = n_batches as f64;

            let avg_c = c_loss_acc / critic_steps;
            let avg_g = g_loss_acc / gen_steps;
            let avg_gp = gp_acc / critic_steps;
            let avg_c_real = c_real_acc / critic_steps;
            let avg_c_fake = c_fake_acc / critic_steps;
            let w_dist = avg_c_real - avg_c_fake;

            eprintln!(
                "epoch {epoch:>4}/{} | C: {avg_c:.4} | G: {avg_g:.4} | W: {w_dist:.4} | GP: {avg_gp:.4} | {epoch_secs:.1}s",
                cfg.epochs
            );

            metrics.epochs.push(EpochMetrics {
                epoch,
                critic_loss: avg_c,
                generator_loss: avg_g,
                gradient_penalty: avg_gp,
                wasserstein_dist: w_dist,
                critic_real: avg_c_real,
                critic_fake: avg_c_fake,
                epoch_secs,
            });
        }
    }

    metrics.total_secs = train_start.elapsed().as_secs_f64();
    eprintln!("training complete in {:.1}s", metrics.total_secs);

    // Save metrics CSV if path configured
    if let Some(ref path) = cfg.metrics_path {
        metrics.save_csv(Path::new(path))?;
    }

    Ok((vs_g, vs_c, metrics))
}
