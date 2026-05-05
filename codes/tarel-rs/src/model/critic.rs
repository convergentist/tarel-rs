use tch::{nn, nn::Module, Tensor};

/// LeakyReLU with custom negative slope (tch-rs auto-gen may not expose the slope param).
fn lrelu(x: &Tensor) -> Tensor {
    x.clamp_min(0.0) + 0.2 * x.clamp_max(0.0)
}

/// Convolutional critic (discriminator): single-channel image → scalar score.
///
/// No sigmoid — outputs unbounded Wasserstein score.
///
/// Architecture (for 64×64 input):
///   1×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1
#[derive(Debug)]
pub struct Critic {
    net: nn::Sequential,
}

impl Critic {
    pub fn new(vs: &nn::Path) -> Self {
        let lk = nn::ConvConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let net = nn::seq()
            // 1×64×64 → 64×32×32
            .add(nn::conv2d(vs / "conv1", 1, 64, 4, lk))
            .add_fn(lrelu)
            // 64×32×32 → 128×16×16
            .add(nn::conv2d(vs / "conv2", 64, 128, 4, lk))
            .add_fn(lrelu)
            // 128×16×16 → 256×8×8
            .add(nn::conv2d(vs / "conv3", 128, 256, 4, lk))
            .add_fn(lrelu)
            // 256×8×8 → 512×4×4
            .add(nn::conv2d(vs / "conv4", 256, 512, 4, lk))
            .add_fn(lrelu)
            // 512×4×4 → 1×1×1
            .add(nn::conv2d(
                vs / "conv5",
                512,
                1,
                4,
                nn::ConvConfig {
                    stride: 1,
                    padding: 0,
                    ..Default::default()
                },
            ));

        Self { net }
    }

    /// Returns Wasserstein score per sample. Shape: [batch, 1, 1, 1] → squeezed to [batch].
    pub fn forward_t(&self, img: &Tensor) -> Tensor {
        self.net.forward(img).view([-1])
    }
}
