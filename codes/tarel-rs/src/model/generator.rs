use tch::{nn, nn::ModuleT, Tensor};

/// Transposed-conv generator: latent vector z → single-channel image.
///
/// Architecture (for 64×64 output):
///   z (LATENT_DIM) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 1×64×64
#[derive(Debug)]
pub struct Generator {
    net: nn::SequentialT,
}

impl Generator {
    pub fn new(vs: &nn::Path, latent_dim: i64) -> Self {
        let bn_cfg = nn::BatchNormConfig::default();
        let deconv = |s, p| nn::ConvTransposeConfig {
            stride: s,
            padding: p,
            ..Default::default()
        };

        let net = nn::seq_t()
            .add(nn::conv_transpose2d(vs / "deconv1", latent_dim, 512, 4, deconv(1, 0)))
            .add(nn::batch_norm2d(vs / "bn1", 512, bn_cfg))
            .add_fn(|x| x.relu())
            .add(nn::conv_transpose2d(vs / "deconv2", 512, 256, 4, deconv(2, 1)))
            .add(nn::batch_norm2d(vs / "bn2", 256, bn_cfg))
            .add_fn(|x| x.relu())
            .add(nn::conv_transpose2d(vs / "deconv3", 256, 128, 4, deconv(2, 1)))
            .add(nn::batch_norm2d(vs / "bn3", 128, bn_cfg))
            .add_fn(|x| x.relu())
            .add(nn::conv_transpose2d(vs / "deconv4", 128, 64, 4, deconv(2, 1)))
            .add(nn::batch_norm2d(vs / "bn4", 64, bn_cfg))
            .add_fn(|x| x.relu())
            .add(nn::conv_transpose2d(vs / "deconv5", 64, 1, 4, deconv(2, 1)))
            .add_fn(|x| x.tanh());

        Self { net }
    }

    /// Forward pass. `z` shape: [batch, LATENT_DIM, 1, 1].
    pub fn forward_t(&self, z: &Tensor) -> Tensor {
        self.net.forward_t(z, true) // always train mode for generator
    }
}
