//! See logan-rs/build.rs for the rationale (torch-sys 0.18 + tch 0.18
//! drop torch_cuda/c10_cuda from the NEEDED list under --as-needed).
fn main() {
    println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg-bins=-ltorch_cuda");
    println!("cargo:rustc-link-arg-bins=-lc10_cuda");
    println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
}
