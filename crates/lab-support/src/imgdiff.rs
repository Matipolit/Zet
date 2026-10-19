//! Image comparison on linear-light `f32` RGBA images, and sRGB-aware file I/O.
//!
//! `rel_mse` is the relative mean squared error used in rendering papers:
//! `mean((test - ref)^2 / (ref^2 + 0.01))`, over RGB. PSNR assumes a peak of 1.0,
//! so it is meaningful for LDR comparisons only.

use anyhow::{ensure, Context};
use image::{ColorType, DynamicImage, Rgb, RgbImage, Rgba, Rgba32FImage, RgbaImage};
use std::path::Path;

pub const REL_MSE_EPSILON: f64 = 1e-2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub mse: f64,
    pub rel_mse: f64,
    pub psnr: f64,
}

fn same_size(a: &Rgba32FImage, b: &Rgba32FImage) -> anyhow::Result<()> {
    ensure!(
        a.dimensions() == b.dimensions(),
        "image sizes differ: {:?} vs {:?}",
        a.dimensions(),
        b.dimensions()
    );
    Ok(())
}

pub fn metrics(test: &Rgba32FImage, reference: &Rgba32FImage) -> anyhow::Result<Metrics> {
    same_size(test, reference)?;
    let mut se = 0.0f64;
    let mut rse = 0.0f64;
    let mut n = 0usize;
    for (t, r) in test.pixels().zip(reference.pixels()) {
        for c in 0..3 {
            let (t, r) = (t.0[c] as f64, r.0[c] as f64);
            let d2 = (t - r) * (t - r);
            se += d2;
            rse += d2 / (r * r + REL_MSE_EPSILON);
            n += 1;
        }
    }
    let mse = se / n as f64;
    let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (1.0 / mse).log10() };
    Ok(Metrics { mse, rel_mse: rse / n as f64, psnr })
}

/// Black → red → yellow → white.
fn ramp(t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0) * 3.0;
    let (r, g, b) = if t < 1.0 {
        (t, 0.0, 0.0)
    } else if t < 2.0 {
        (1.0, t - 1.0, 0.0)
    } else {
        (1.0, 1.0, t - 2.0)
    };
    Rgb([(r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8])
}

/// Per-pixel mean absolute RGB difference, divided by `scale`, on a colour ramp.
pub fn heatmap(a: &Rgba32FImage, b: &Rgba32FImage, scale: f32) -> anyhow::Result<RgbImage> {
    same_size(a, b)?;
    ensure!(scale > 0.0, "heatmap scale must be positive");
    let (w, h) = a.dimensions();
    let mut out = RgbImage::new(w, h);
    for (x, y, p) in out.enumerate_pixels_mut() {
        let (pa, pb) = (a.get_pixel(x, y), b.get_pixel(x, y));
        let diff = (0..3).map(|c| (pa.0[c] - pb.0[c]).abs()).sum::<f32>() / 3.0;
        *p = ramp(diff / scale);
    }
    Ok(out)
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Loads an image as linear-light RGBA `f32`. Float formats (EXR) are taken as
/// linear; integer formats (PNG and friends) are decoded from sRGB.
pub fn load_linear(path: &Path) -> anyhow::Result<Rgba32FImage> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(match img.color() {
        ColorType::Rgb32F | ColorType::Rgba32F => img.into_rgba32f(),
        _ => {
            let rgba16 = img.into_rgba16();
            let (w, h) = rgba16.dimensions();
            let mut out = Rgba32FImage::new(w, h);
            for (src, dst) in rgba16.pixels().zip(out.pixels_mut()) {
                let f = |v: u16| v as f32 / 65535.0;
                *dst = Rgba([
                    srgb_to_linear(f(src.0[0])),
                    srgb_to_linear(f(src.0[1])),
                    srgb_to_linear(f(src.0[2])),
                    f(src.0[3]),
                ]);
            }
            out
        }
    })
}

/// Saves a linear-light image. `.exr` keeps it linear `f32`; anything else is
/// written as sRGB-encoded 8-bit with values clamped to `[0, 1]`.
pub fn save_linear(img: &Rgba32FImage, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let is_exr = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr"));
    if is_exr {
        DynamicImage::ImageRgba32F(img.clone())
            .save(path)
            .with_context(|| format!("saving {}", path.display()))?;
        return Ok(());
    }
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (src, dst) in img.pixels().zip(out.pixels_mut()) {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        *dst = Rgba([
            q(linear_to_srgb(src.0[0])),
            q(linear_to_srgb(src.0[1])),
            q(linear_to_srgb(src.0[2])),
            q(src.0[3]),
        ]);
    }
    out.save(path).with_context(|| format!("saving {}", path.display()))
}
