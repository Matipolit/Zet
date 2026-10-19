//! zet rendering lab. The shell lives in `shell/`; the renderer is `renderer.rs`.
//!
//! ```text
//! cargo run -p lab                                       # window with the overlay
//! cargo run -p lab -- --adapter llvmpipe                 # force the CPU Vulkan driver
//! cargo run -p lab -- --frames 64 --capture captures/out.png
//! cargo run -p lab -- --bench clear --frames 300         # writes bench/clear.csv
//! ```

mod renderer;
mod shell;

use clap::Parser;
use std::path::PathBuf;

/// The renderer the shell drives. Change this when you replace `ClearRenderer`.
type ActiveRenderer = renderer::ClearRenderer;

#[derive(Parser, Debug, Clone)]
#[command(name = "lab", about = "zet rendering lab: a window shell around the renderer in src/renderer.rs")]
pub struct Args {
    #[arg(long, default_value_t = 1280)]
    pub width: u32,
    #[arg(long, default_value_t = 720)]
    pub height: u32,
    /// Substring of the adapter name or driver to use, e.g. `Iris` or `llvmpipe`.
    #[arg(long)]
    pub adapter: Option<String>,
    /// Exit after this many frames. With --bench, counts recorded frames after warm-up.
    #[arg(long)]
    pub frames: Option<u64>,
    /// Record per-pass GPU timings and write bench/<name>.csv on exit (default 300 frames).
    #[arg(long)]
    pub bench: Option<String>,
    /// Frames to discard before recording in bench mode.
    #[arg(long, default_value_t = 30)]
    pub warmup: u64,
    /// Save the renderer output to this .png (sRGB) or .exr (linear) on exit.
    #[arg(long)]
    pub capture: Option<PathBuf>,
    /// Present with vsync. Off by default so timings are not capped by the display.
    #[arg(long)]
    pub vsync: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();
    shell::run::<ActiveRenderer>(Args::parse())
}
