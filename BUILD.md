# Build, Package, and Distribute — Ultradian Binary Release Guide

A paced walkthrough for taking a Rust + CUDA binary built on the **Ultradian** server (x86_64 Linux, NVIDIA GTX 1080 Ti, Pascal `sm_61`) and shipping it to other people via three escalating distribution levels.

Each level produces a usable artifact. Later levels reuse work from earlier ones, so work top-down.

---

## 0. Context, assumptions, and scope

**Build host.** Ultradian, x86_64 Linux, NVIDIA GTX 1080 Ti, Pascal (compute capability 6.1, `sm_61`).

**Toolchain.** Rust (pinned), CUDA Toolkit 12.x. License: **Apache 2.0**.

**Distribution intent.**
1. Recipients **run** the binary on their own Pascal-class (or newer) Linux host with their own CUDA install.
2. Recipients **cite** the work (DOI from Zenodo, Level 2).
3. Rebuilding from source is *not* a supported distribution path. The Apache 2.0 license already disclaims liability for downstream modifications — no further effort required.

**Explicitly out of scope (decisions already made).**
- No `minisign`/`cosign` cryptographic signing — integrity via SHA-256 hosted on the same release page as the artifact.
- No GitHub Actions CI / OIDC provenance — local builds only for now.
- No bundling of `libcudnn.so`, `libcublas.so`, or any other NVIDIA library subject to the cuDNN/cuBLAS EULA. **Recipient installs CUDA Toolkit themselves.**
- No dataset coupling, multi-arch support, bit-reproducibility chasing, anonymization for double-blind review, or glibc-floor analysis.
- No automated rebuild testing on other hardware (later stage).

---

## 1. Pascal CUDA lifetime — read this once, repeat in every release

Pascal (`sm_61`) is **deprecated in CUDA 12** and is expected to be **dropped in CUDA 13**. Practical consequences you should bake into every README, release note, and `BUILDINFO.txt`:

- The CUDA **12.x** line is the last supported toolkit for the GTX 1080 Ti.
- Binaries shipped today have a finite useful life roughly bounded by "how long CUDA 12.x runtimes remain installable on mainstream distros."
- Anyone with Ampere (`sm_86`), Ada (`sm_89`), or Hopper (`sm_90`) hardware will still run a `sm_61`-only build via JIT recompilation of PTX *if* you also embed PTX, but native performance requires a different build with their SM target.

This is not a problem to solve today, but it must be **stated** today so future users aren't surprised.

---

## 2. Pre-flight checklist (do once, applies to all levels)

These steps belong in the source repository and are reused by every level. Run through this list before touching Level 0.

### 2.1 Pin the Rust toolchain

Create `rust-toolchain.toml` at the repository root:

```toml
[toolchain]
channel = "1.79.0"      # adjust to whatever you actually use
components = ["rustfmt", "clippy"]
profile = "minimal"
```

Commit it. Future `cargo` invocations will auto-install and use this exact toolchain.

### 2.2 Tighten the release profile

In `Cargo.toml`:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
strip = "none"          # we'll strip manually so we can keep a debug sidecar
panic = "abort"         # only if you don't rely on unwinding; otherwise leave default
opt-level = 3
debug = 1               # keep enough for symbol resolution; we'll split it out
```

### 2.3 Embed version + commit hash into the binary

`build.rs` at the crate root:

```rust
use std::process::Command;

fn main() {
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let git_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let suffix = if git_dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=GIT_HASH={git_hash}{suffix}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
```

Expose it via your CLI `--version`:

```rust
// in main.rs or wherever your CLI lives
const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("GIT_HASH"),
    ")"
);
```

### 2.4 Commit `Cargo.lock`

For a binary crate, `Cargo.lock` must be committed. Verify:

```bash
git ls-files Cargo.lock   # must print the path
```

### 2.5 License and notice files

At the repo root, ensure:

- `LICENSE` — full Apache 2.0 text.
- `NOTICE` — short attribution file naming the project, copyright holder, and any third-party Apache-licensed code that requires NOTICE propagation.

### 2.6 Naming convention

Every artifact uses this template:

```
<crate>-<version>-x86_64-linux-cuda<major.minor>-sm61
```

Example: `ultradian-0.4.2-x86_64-linux-cuda12.3-sm61`.

Use this for the binary basename, the tarball, and the Docker tag suffix where appropriate.

---

## 3. Level 0 — Release-quality local artifact

**Goal.** Produce a self-contained directory on Ultradian containing the binary, build metadata, runtime requirements, and a local checksum. Ready to be tarred and copied anywhere by hand (`scp`, USB, NFS). **No `.so` files bundled** — recipients install CUDA Toolkit themselves to avoid any NVIDIA redistribution concerns.

This level is also your local pre-release staging area for Level 1.

### 3.1 Set environment for the build

```bash
# pin CUDA explicitly so build scripts and nvcc agree
export CUDA_HOME=/usr/local/cuda-12.3        # adjust to your install
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# tell any nvcc invocation to target Pascal natively + ship PTX as forward-compat fallback
export NVCC_GENCODE="-gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61"
```

If you use a `build.rs` that shells out to `nvcc` (via `cust`, `cudarc` raw build, or hand-rolled), thread `$NVCC_GENCODE` through it. If you use a crate that respects `CUDA_ARCH_FLAGS` or similar, set that.

### 3.2 Build

```bash
cd /path/to/repo
cargo clean                       # for a clean, attestable build
cargo build --release --locked    # --locked = fail if Cargo.lock is out of date
```

Verify the binary exists:

```bash
ls -lh target/release/<crate>
target/release/<crate> --version  # should print version + commit hash
```

### 3.3 Strip and split debug info

```bash
BIN=target/release/<crate>
objcopy --only-keep-debug "$BIN" "${BIN}.debug"
strip --strip-debug --strip-unneeded "$BIN"
objcopy --add-gnu-debuglink="${BIN}.debug" "$BIN"
```

You now have a stripped binary that still knows where its debug symbols live. Keep `${BIN}.debug` next to the binary in the release directory — you'll need it the first time someone hands you a stack trace.

### 3.4 Inspect runtime dependencies

```bash
ldd target/release/<crate>
readelf -d target/release/<crate> | grep -E '(NEEDED|RPATH|RUNPATH|SONAME)'
```

Expected output: dynamic links to `libcuda.so.1` (the driver — always provided by the host's NVIDIA driver, never bundled), `libcudart.so.12`, possibly `libcublas.so.12`, `libcudnn.so.9`, plus libc, libm, libpthread, libdl, libgcc_s.

If you see `libcudnn.so` and you'd rather the recipient install cuDNN separately, that's already the plan — note exactly which cuDNN major version is required.

Save this output to a file; you'll embed it in `BUILDINFO.txt`.

### 3.5 Create the release directory

```bash
VERSION=$(target/release/<crate> --version | awk '{print $2}')   # or read from Cargo.toml
CUDA_VER=$(nvcc --version | awk '/release/ {print $6}' | sed 's/V//;s/,//')
RELDIR="releases/<crate>-${VERSION}-x86_64-linux-cuda${CUDA_VER%.*}-sm61"

mkdir -p "$RELDIR"
cp target/release/<crate>         "$RELDIR/"
cp target/release/<crate>.debug   "$RELDIR/"
cp LICENSE NOTICE                 "$RELDIR/"
```

### 3.6 Write `BUILDINFO.txt`

Place this at `$RELDIR/BUILDINFO.txt`. Template:

```
# Build Information
# Generated: <ISO 8601 timestamp, UTC>

[project]
name      = <crate>
version   = <version>
git_commit = <12-char hash>
git_tag   = <tag if any, else "none">
license   = Apache-2.0

[build_host]
hostname  = ultradian
os        = <uname -a>
kernel    = <uname -r>
distro    = <cat /etc/os-release | grep PRETTY_NAME>

[toolchain]
rustc     = <rustc -vV output, full block>
cargo     = <cargo --version>
nvcc      = <nvcc --version output>
cuda_home = <$CUDA_HOME>
gencode   = -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

[gpu_at_build_time]
nvidia_smi = <nvidia-smi output, first 30 lines>

[dynamic_linkage]
ldd_output = <full ldd target/release/<crate>>
needed_libs = <readelf -d ... | grep NEEDED>

[notes]
- libcuda.so.1 is NOT bundled — it is provided by the NVIDIA driver on the host.
- libcudnn.so / libcublas.so are NOT bundled — install via CUDA Toolkit + cuDNN.
- This binary targets sm_61 (Pascal). It will JIT on newer SMs via embedded PTX
  but native binaries for Ampere/Ada/Hopper are not provided.
- Pascal is deprecated in CUDA 12 and expected to be removed in CUDA 13.
```

You can either write this by hand for each release or wrap it in a `make buildinfo` recipe — your call. Hand-written is fine for now.

### 3.7 Write `RUNTIME_REQUIREMENTS.md`

Place this at `$RELDIR/RUNTIME_REQUIREMENTS.md`. Template:

```markdown
# Runtime Requirements

This binary was built on x86_64 Linux against CUDA 12.x and targets NVIDIA
Pascal GPUs (compute capability 6.1, e.g. GTX 1080 Ti, Tesla P100, Titan Xp).

## Required on the host

- **NVIDIA driver:** R525 or newer (CUDA 12.x compatibility).
- **CUDA Toolkit:** 12.x — install yourself from
  https://developer.nvidia.com/cuda-toolkit-archive
- **cuDNN:** v9.x for CUDA 12 — install yourself from
  https://developer.nvidia.com/cudnn (free account required)
- **glibc:** 2.35 or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+, RHEL 9+).

We do not redistribute cuDNN or cuBLAS. You must accept NVIDIA's EULA and
install them locally.

## Verify your host

    nvidia-smi                                  # driver + GPU sanity check
    nvcc --version                              # CUDA toolkit version
    ldconfig -p | grep -E 'libcudnn|libcublas'  # cuDNN/cuBLAS visible to dynamic linker

## Running

    ./<crate> --version
    ./<crate> --help

If you hit `error while loading shared libraries: libcudnn.so.9: cannot open
shared object file`, set `LD_LIBRARY_PATH` to your CUDA install:

    export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

## GPU notes

- The binary will run natively on Pascal SMs.
- On newer GPUs (Ampere/Ada/Hopper) the embedded PTX will be JIT-compiled on
  first launch, which adds a one-time startup cost. Native builds for those
  architectures are not currently provided.
- Pascal is deprecated in CUDA 12 and will likely be removed in CUDA 13. The
  CUDA 12.x line is the last supported toolkit for this hardware.
```

### 3.8 Hash everything in the directory

```bash
cd "$RELDIR"
sha256sum * > SHA256SUMS
cd -
```

`SHA256SUMS` lists every file (including itself if you re-run, so re-run only once or exclude it explicitly). Anyone with the directory can run `sha256sum -c SHA256SUMS` to detect corruption.

### 3.9 Mark the directory immutable (optional but cheap)

```bash
chmod -R a-w "$RELDIR"        # filesystem perms — convention
# optional, requires root: chattr +i recursively to make it truly immutable
```

### 3.10 Off-host backup

The Ultradian server is your only build host. If it goes down, the artifact is gone unless copied. **Pick one** of:

- `rsync -avzP "$RELDIR" you@nas:/path/to/releases/`
- `restic backup "$RELDIR"` to your backup repo
- A tarball copied to S3/B2/iCloud with versioned object naming

Do this *before* moving to Level 1. Level 1 (GitHub) is itself a backup, but the gap between "build done" and "uploaded" is the dangerous one.

### 3.11 Smoke test on the build host

Before declaring Level 0 done, run the binary from inside `$RELDIR`:

```bash
cd "$RELDIR"
./<crate> --version
./<crate> --help
# whatever minimal end-to-end invocation makes sense
./<crate> <small test input>
```

If it fails, fix it now. A bad Level 0 artifact poisons every level above.

### 3.12 Level 0 done — checklist

- [ ] `rust-toolchain.toml` committed
- [ ] `Cargo.lock` committed
- [ ] `LICENSE` and `NOTICE` at repo root
- [ ] `build.rs` embeds version + commit
- [ ] Built with `cargo build --release --locked`
- [ ] Binary stripped, `.debug` sidecar kept
- [ ] `ldd` output captured
- [ ] `$RELDIR` populated with binary, `.debug`, `LICENSE`, `NOTICE`, `BUILDINFO.txt`, `RUNTIME_REQUIREMENTS.md`, `SHA256SUMS`
- [ ] `$RELDIR` marked read-only
- [ ] Off-host backup taken
- [ ] Smoke test passed

---

## 4. Level 1 — GitHub Release tarball

**Goal.** Make the Level 0 artifact publicly downloadable via a versioned GitHub Release with an integrity-checkable checksum and an SBOM. No cryptographic signing for now — the SHA256SUMS file on the release page protects against accidental corruption but not against an attacker who can replace the release. That's an acceptable tradeoff for the current stage.

### 4.1 Sanity-check what you're about to ship

```bash
ls -la "$RELDIR"
cat "$RELDIR/BUILDINFO.txt"
cat "$RELDIR/RUNTIME_REQUIREMENTS.md"
sha256sum -c "$RELDIR/SHA256SUMS"   # all OK
```

### 4.2 Tag the commit

```bash
git tag -a v0.4.2 -m "Release v0.4.2 — <one-line summary>"
git push origin v0.4.2
```

Use [semver](https://semver.org). For `0.x.y` releases, breaking changes are allowed at every minor — call them out in the release notes.

### 4.3 Generate an SBOM

Two SBOMs tell complementary stories — ship both.

```bash
# Rust dependency graph from Cargo.lock
cargo install --locked cargo-cyclonedx
cargo cyclonedx --format json --output-pattern bom
mv bom.json "$RELDIR/SBOM-rust-cyclonedx.json"

# Whole-binary view via syft (catches C/CUDA libs by SONAME)
# https://github.com/anchore/syft
syft "$RELDIR/<crate>" -o cyclonedx-json > "$RELDIR/SBOM-binary-syft.json"
```

If you skip either, document which one you skipped and why in `BUILDINFO.txt`.

### 4.4 Create the tarball

```bash
TARBALL="<crate>-v0.4.2-x86_64-linux-cuda12.3-sm61.tar.zst"
tar --owner=0 --group=0 --numeric-owner --sort=name \
    --mtime='@0' \
    -I 'zstd -19' \
    -cf "$TARBALL" -C "$(dirname "$RELDIR")" "$(basename "$RELDIR")"

sha256sum "$TARBALL" > "$TARBALL.sha256"
```

Flags explained:
- `--owner=0 --group=0 --numeric-owner` — strip build-host user info.
- `--sort=name --mtime='@0'` — reduce nondeterminism (won't get full bit-reproducibility but eliminates the easy diffs).
- `zstd -19` — high compression, fast decompression on the recipient side.

### 4.5 Smoke test the tarball on a different directory

The point is to catch "I baked an absolute path into the binary" mistakes.

```bash
mkdir /tmp/smoke && cd /tmp/smoke
tar -I zstd -xf /path/to/$TARBALL
cd <crate>-*-sm61
sha256sum -c SHA256SUMS
./<crate> --version
./<crate> --help
```

If possible, also run the smoke test on a second user account on the same machine (no shared shell config, fresh `LD_LIBRARY_PATH`). This catches "it only works because *my* shell has the right exports" surprises.

### 4.6 Write release notes

A short Markdown blob you'll paste into the GitHub Release form. Template:

```markdown
## <crate> v0.4.2

**Target:** x86_64 Linux · NVIDIA Pascal (`sm_61`) · CUDA 12.x · glibc ≥ 2.35

### Changes
- <bullet>
- <bullet>

### Verification
After downloading, verify integrity with:

    sha256sum -c <crate>-v0.4.2-x86_64-linux-cuda12.3-sm61.tar.zst.sha256

Then extract and read `RUNTIME_REQUIREMENTS.md` before running.

### Pascal lifetime notice
This binary targets NVIDIA Pascal GPUs. Pascal is deprecated in CUDA 12 and is
expected to be removed in CUDA 13. CUDA 12.x is the last supported toolkit for
this hardware.

### License
Apache 2.0. See LICENSE inside the tarball.
```

### 4.7 Publish the release

Web UI path:
1. GitHub repo → Releases → "Draft a new release".
2. Choose tag `v0.4.2` (already pushed).
3. Title: `v0.4.2 — <one-line summary>`.
4. Paste the release notes from §4.6.
5. Attach assets:
   - `<crate>-v0.4.2-x86_64-linux-cuda12.3-sm61.tar.zst`
   - `<crate>-v0.4.2-x86_64-linux-cuda12.3-sm61.tar.zst.sha256`
   - `BUILDINFO.txt` (copy from `$RELDIR`)
   - `RUNTIME_REQUIREMENTS.md` (copy from `$RELDIR`)
   - `SBOM-rust-cyclonedx.json`
   - `SBOM-binary-syft.json`
6. **Do not** mark as pre-release for stable cuts; do for `vX.Y.Z-rc.N`.
7. Publish.

CLI path (`gh` CLI, often faster):

```bash
gh release create v0.4.2 \
  --title "v0.4.2 — <summary>" \
  --notes-file RELEASE_NOTES.md \
  "$TARBALL" \
  "$TARBALL.sha256" \
  "$RELDIR/BUILDINFO.txt" \
  "$RELDIR/RUNTIME_REQUIREMENTS.md" \
  "$RELDIR/SBOM-rust-cyclonedx.json" \
  "$RELDIR/SBOM-binary-syft.json"
```

### 4.8 Update repo README

In the repository `README.md`, link to the latest release and explain the
verification workflow:

```markdown
## Releases

Pre-built x86_64 Linux binaries for NVIDIA Pascal GPUs (CUDA 12.x) are
available on the [Releases page](../../releases).

Download the `.tar.zst` and the matching `.sha256`, then:

    sha256sum -c <name>.tar.zst.sha256
    tar -I zstd -xf <name>.tar.zst
    cd <name> && cat RUNTIME_REQUIREMENTS.md
```

Also link to a `BUILDING.md` (or extend this section) that briefly describes:
- The pinned toolchain (`rust-toolchain.toml`).
- The `cargo build --release --locked` invocation.
- The strip/objcopy/tar recipe used to produce the release.
- That builds are intentionally not bit-reproducible at this stage.

### 4.9 Level 1 done — checklist

- [ ] Tag pushed
- [ ] SBOMs generated (both Rust + binary)
- [ ] Tarball built with stable flags
- [ ] Tarball SHA-256 computed
- [ ] Tarball smoke-tested in a fresh directory
- [ ] Release notes written
- [ ] GitHub Release published with all assets
- [ ] Repo README updated with download/verify instructions

---

## 5. Level 2 — Docker image (GHCR) + Zenodo DOI

**Goal.** Ship a container image that bundles a known-good CUDA runtime (legal under NVIDIA's container licensing) so recipients can `docker run --gpus all` without installing CUDA themselves. Then archive each GitHub Release to Zenodo and obtain a citable DOI.

### 5.1 Confirm `nvidia-container-toolkit` on Ultradian

You weren't sure whether it's installed. Run:

```bash
# is the nvidia runtime registered with Docker?
docker info 2>/dev/null | grep -i 'runtimes\|nvidia'

# is the CLI present?
which nvidia-ctk && nvidia-ctk --version

# end-to-end test: a CUDA image should see your GPU
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

**If the last command prints the GTX 1080 Ti's `nvidia-smi` output**, you're good — skip §5.2.

**If it fails**, install the toolkit (Ubuntu/Debian instructions; adjust for your distro):

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### 5.2 Choose and pin base images

You need two — a build-stage image (has nvcc, headers) and a runtime-stage image (small, just enough to run).

**Build stage.** `nvidia/cuda:12.3.2-devel-ubuntu22.04` — ~5 GB, includes nvcc/headers.

**Runtime stage.** Pick the smallest image that still has what your binary dynamically links to. Options, smallest first:

- `nvidia/cuda:12.3.2-base-ubuntu22.04` — ~150 MB. **Use this if your binary only needs `libcudart`** and the driver. Pure CUDA, no cuBLAS/cuDNN.
- `nvidia/cuda:12.3.2-runtime-ubuntu22.04` — ~2 GB. Adds `libcublas`, `libcurand`, `libcufft`, etc.
- `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04` — ~3 GB. Adds cuDNN 9 on top of runtime. **Use this if your binary links `libcudnn.so`.**

Pick the smallest one whose `lib64/` directory contains every SONAME from your Level 0 `ldd` output.

**Pin by digest.** Tags are mutable. Query the digest once and pin:

```bash
docker pull nvidia/cuda:12.3.2-devel-ubuntu22.04
docker inspect --format='{{index .RepoDigests 0}}' nvidia/cuda:12.3.2-devel-ubuntu22.04
# nvidia/cuda@sha256:abc123...

docker pull nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04   # or whichever runtime you chose
docker inspect --format='{{index .RepoDigests 0}}' nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

Record both `@sha256:...` strings; you'll paste them into the Dockerfile.

### 5.3 Multi-stage `Dockerfile`

Place at the repo root or under `docker/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1.7

# -------- build stage --------
FROM nvidia/cuda@sha256:<BUILD_DIGEST_FROM_5.2> AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:${PATH} \
    NVCC_GENCODE="-gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config git \
    && rm -rf /var/lib/apt/lists/*

# pinned rustup install — same channel as rust-toolchain.toml
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain none --profile minimal

WORKDIR /src
COPY rust-toolchain.toml ./
RUN rustup show   # installs the channel pinned by rust-toolchain.toml

# Cache deps separately from source for faster rebuilds
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs && cargo fetch
COPY . .
RUN cargo build --release --locked

RUN strip --strip-debug --strip-unneeded /src/target/release/<crate>

# -------- runtime stage --------
FROM nvidia/cuda@sha256:<RUNTIME_DIGEST_FROM_5.2> AS runtime

LABEL org.opencontainers.image.title="<crate>"
LABEL org.opencontainers.image.description="Ultradian binary, Pascal-targeted, CUDA 12.x"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/<you>/<repo>"
# fill these from CI vars or pass via --label at build time
LABEL org.opencontainers.image.version="0.4.2"
LABEL org.opencontainers.image.revision="<git-sha>"

COPY --from=builder /src/target/release/<crate> /usr/local/bin/<crate>
COPY LICENSE NOTICE /usr/share/doc/<crate>/

# tini for proper signal handling — optional but nice
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/<crate>"]
CMD ["--help"]
```

### 5.4 Build, label, and locally test the image

```bash
GIT_SHA=$(git rev-parse --short=12 HEAD)
VERSION=0.4.2
IMAGE=ghcr.io/<you>/<crate>

docker build \
  --label "org.opencontainers.image.version=${VERSION}" \
  --label "org.opencontainers.image.revision=${GIT_SHA}" \
  -t "${IMAGE}:${VERSION}" \
  -t "${IMAGE}:latest" \
  .

# smoke test against your GPU
docker run --rm --gpus all "${IMAGE}:${VERSION}" --version
docker run --rm --gpus all "${IMAGE}:${VERSION}" --help

# capture the digest for the release notes
docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE}:${VERSION}" 2>/dev/null || \
  echo "no digest yet — only available after push"
```

### 5.5 Push to GHCR

One-time setup: create a GitHub Personal Access Token (classic) with scopes
`write:packages`, `read:packages`, `delete:packages`. Store it somewhere
recoverable (password manager). Then:

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u <your-github-username> --password-stdin
docker push "${IMAGE}:${VERSION}"
docker push "${IMAGE}:latest"

# now capture the immutable digest from the registry
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE}:${VERSION}")
echo "$DIGEST"
# ghcr.io/<you>/<crate>@sha256:abc123...
```

In the GitHub UI, find the package under your profile/org → Packages →
`<crate>` → mark **Public** (Apache 2.0, no reason to gate it).

### 5.6 Update the GitHub Release notes

Edit the v0.4.2 release notes to add a Docker section. Use the **digest**
(not the tag) so the citation is immutable:

```markdown
### Docker image

    docker run --rm --gpus all \
      ghcr.io/<you>/<crate>@sha256:abc123... \
      --help

Requires NVIDIA driver R525+ and `nvidia-container-toolkit` on the host.
The image is pinned to its build-time digest; the `:0.4.2` tag points to the
same content but tags are theoretically mutable, so the digest is the
canonical reference.

Base image: `nvidia/cuda@sha256:<runtime-digest>`
(`12.3.2-cudnn9-runtime-ubuntu22.04` at time of build).
```

### 5.7 Zenodo DOI via GitHub webhook

One-time setup:

1. Sign in to [zenodo.org](https://zenodo.org) with your ORCID. Connect your
   GitHub account in Zenodo settings → Linked accounts.
2. Go to Zenodo → [GitHub integration](https://zenodo.org/account/settings/github/).
3. Find `<you>/<repo>` in the list, flip the switch to **On**.

From now on, **every published GitHub Release** auto-creates a new Zenodo
deposit and mints a versioned DOI. You also get a "concept DOI" that always
points to the latest version (useful for citation in evolving work).

Pre-fill metadata so each release deposit is correct without manual fixup —
create `.zenodo.json` at the repository root:

```json
{
  "title": "<crate>: <one-line description>",
  "description": "Rust + CUDA implementation accompanying the TAReL paper. Targets NVIDIA Pascal GPUs (sm_61) on x86_64 Linux with CUDA 12.x.",
  "creators": [
    {
      "name": "Me, Francisco",
      "affiliation": "<your university or affiliation>",
      "orcid": "<your ORCID, no URL prefix>"
    }
  ],
  "keywords": [
    "rust", "cuda", "gpu", "quantitative finance",
    "limit order book", "recurrence plots", "generative adversarial networks"
  ],
  "license": "Apache-2.0",
  "upload_type": "software",
  "access_right": "open",
  "related_identifiers": [
    {
      "identifier": "https://github.com/<you>/<repo>",
      "relation": "isSupplementTo",
      "scheme": "url"
    }
  ]
}
```

Commit this file and cut a new release (or re-publish v0.4.2) to trigger the
first archival. The DOI appears on Zenodo within a minute or two.

After it appears, add the DOI badge to the repository `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

### 5.8 Caveats to record in `BUILDINFO.txt` and release notes

- The Zenodo deposit archives the **GitHub repo snapshot at the tag**, not the
  Docker image or the binary. The image lives in GHCR, the binary lives in
  the GitHub Release assets. The DOI cites the *source* — useful for academic
  reference but not a replacement for the GHCR/Releases artifacts.
- If you ever need the *binary* itself archived under a DOI (some funders or
  journals require this), attach the tarball as a Zenodo upload in the
  Zenodo UI after the auto-deposit appears.

### 5.9 Level 2 done — checklist

- [ ] `nvidia-container-toolkit` confirmed working on Ultradian
- [ ] Base image digests pinned for both build and runtime
- [ ] Multi-stage Dockerfile committed
- [ ] Image built, tagged with version + git sha labels
- [ ] Image smoke-tested with `--gpus all`
- [ ] Image pushed to GHCR, marked public
- [ ] Release notes updated with image digest
- [ ] `.zenodo.json` committed
- [ ] Zenodo GitHub integration enabled
- [ ] First DOI minted
- [ ] DOI badge added to README

---

## 6. Per-release runbook (after the one-time setup is done)

Once Levels 0–2 are all set up once, each new release becomes mechanical. Order
of operations:

1. Bump version in `Cargo.toml`. Update `CHANGELOG.md` if you keep one.
2. Commit, push.
3. `git tag -a vX.Y.Z -m "…"` then `git push origin vX.Y.Z`.
4. On Ultradian, run Level 0 §3.1–§3.11 → produces `$RELDIR`.
5. Run Level 1 §4.3–§4.7 → tarball + SBOM + GitHub Release published.
6. Run Level 2 §5.4–§5.6 → image pushed, release notes amended with digest.
7. Zenodo deposit auto-creates from the GitHub Release. Verify the DOI in the
   Zenodo UI. Update the README badge if it's a major version.

If you want to wrap any of this in a `justfile` or `Makefile` later, that's a
nice quality-of-life improvement but is not required to ship.

---

## 7. Known limitations of this plan (state them, don't hide them)

These are intentional. Document them in `BUILDINFO.txt` and any "Limitations"
section of papers using this artifact.

- **No cryptographic signatures.** Integrity is via SHA-256 hosted on the same
  release page. An attacker who replaces both the asset and the checksum can
  serve a tampered binary. Acceptable for the current stage; add `minisign`
  or `cosign` later when distribution scale warrants it.
- **No GitHub Actions CI.** Builds happen on Ultradian only. There is no
  independently verifiable build provenance. Acceptable for now; SLSA L2+
  via GitHub Actions OIDC is a future upgrade.
- **No bit-reproducibility.** Rust binaries currently embed some non-deterministic
  metadata. Builds produced from the same commit on the same host will differ
  byte-for-byte across runs. The SBOM + `BUILDINFO.txt` partially compensate.
- **No glibc-floor analysis.** Recipients on distros older than the build host
  (Ubuntu 22.04 → glibc 2.35) may see `GLIBC_2.35 not found` errors and have
  no recourse other than the Docker image.
- **No cross-hardware testing matrix.** The artifact is exercised only on the
  Ultradian GTX 1080 Ti. Other Pascal cards (P100, Titan Xp, GTX 1070, etc.)
  *should* work but are untested.
- **Pascal is on a finite clock.** CUDA 12.x is the last toolkit line that
  supports `sm_61`. When CUDA 13+ becomes ubiquitous, this binary requires
  recipients to keep an older CUDA install around.
- **NeurIPS double-blind anonymity is not preserved.** GitHub identity, GHCR
  namespace, and Zenodo author records all leak authorship. Use these levels
  only post-submission or for non-anonymous distribution.

---

## 8. Quick reference: file inventory per level

| File                            | Level 0 | Level 1 (in tarball) | Level 1 (as separate release asset) | Level 2 (in image) |
|---------------------------------|:-------:|:--------------------:|:-----------------------------------:|:------------------:|
| `<crate>` (stripped binary)     |   ✓     |          ✓           |                                     |     /usr/local/bin |
| `<crate>.debug` (sidecar)       |   ✓     |          ✓           |                                     |                    |
| `LICENSE`                       |   ✓     |          ✓           |                                     |  /usr/share/doc/   |
| `NOTICE`                        |   ✓     |          ✓           |                                     |  /usr/share/doc/   |
| `BUILDINFO.txt`                 |   ✓     |          ✓           |                  ✓                  |                    |
| `RUNTIME_REQUIREMENTS.md`       |   ✓     |          ✓           |                  ✓                  |                    |
| `SHA256SUMS`                    |   ✓     |          ✓           |                                     |                    |
| `<tarball>.tar.zst`             |         |                      |                  ✓                  |                    |
| `<tarball>.tar.zst.sha256`      |         |                      |                  ✓                  |                    |
| `SBOM-rust-cyclonedx.json`      |         |                      |                  ✓                  |                    |
| `SBOM-binary-syft.json`         |         |                      |                  ✓                  |                    |

Zenodo deposit contains: the GitHub source tarball at the tagged commit, plus
optionally the manually-uploaded release artifacts.

---

## 9. References

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- CUDA Toolkit archive (12.x downloads): https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN download: https://developer.nvidia.com/cudnn
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- CycloneDX for Rust: https://github.com/CycloneDX/cyclonedx-rust-cargo
- Syft (SBOM generator): https://github.com/anchore/syft
- Zenodo–GitHub integration: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- Semantic Versioning: https://semver.org
