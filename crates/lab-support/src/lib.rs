//! Measurement tooling for the zet rendering lab.
//!
//! Nothing in here renders anything. It exists so the renderer you write can be
//! timed, captured and compared against references with one call each:
//!
//! - [`gpu`]: adapter selection and device creation, headless or for a window.
//! - [`gpu_timer`]: per-pass GPU timestamps with a non-blocking readback ring.
//! - [`capture`]: read a texture back into a linear `f32` image.
//! - [`imgdiff`]: image metrics (MSE, relative MSE, PSNR), heatmaps, sRGB-aware I/O.
//! - [`bench`]: benchmark samples, their CSV form, summaries and regression checks.
//!
//! Convention used throughout: images in memory are **linear light**, `f32` RGBA.
//! PNG files are sRGB-encoded 8-bit; EXR files are linear. Loading and saving
//! convert accordingly.

pub mod bench;
pub mod capture;
pub mod gpu;
pub mod gpu_timer;
pub mod imgdiff;
