//! Reading textures back into linear-light `f32` images.
//!
//! The texture needs `TextureUsages::COPY_SRC`. Supported formats: `Rgba8Unorm`,
//! `Rgba8UnormSrgb`, `Bgra8Unorm`, `Bgra8UnormSrgb`, `Rgba16Float`, `Rgba32Float`.
//! sRGB formats are decoded to linear; the others are assumed linear already.

use crate::gpu::Gpu;
use crate::imgdiff::srgb_to_linear;
use anyhow::{anyhow, bail};
use image::{Rgba, Rgba32FImage};
use std::sync::mpsc;
use wgpu::TextureFormat as F;

pub fn read_texture(gpu: &Gpu, texture: &wgpu::Texture) -> anyhow::Result<Rgba32FImage> {
    let format = texture.format();
    let bytes_per_pixel: u32 = match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb | F::Bgra8Unorm | F::Bgra8UnormSrgb => 4,
        F::Rgba16Float => 8,
        F::Rgba32Float => 16,
        other => bail!("read_texture: unsupported format {other:?}"),
    };
    let size = texture.size();
    let (w, h) = (size.width, size.height);
    let unpadded = w * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture staging"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let index = gpu.queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
        .map_err(|e| anyhow!("waiting for capture: {e:?}"))?;
    rx.recv()
        .map_err(|_| anyhow!("map callback never fired"))?
        .map_err(|e| anyhow!("mapping capture buffer: {e:?}"))?;

    let data = staging.slice(..).get_mapped_range().map_err(|e| anyhow!("{e:?}"))?;
    let mut out = Rgba32FImage::new(w, h);
    for y in 0..h {
        let row = &data[(y * padded) as usize..(y * padded + unpadded) as usize];
        for x in 0..w {
            let px = &row[(x * bytes_per_pixel) as usize..((x + 1) * bytes_per_pixel) as usize];
            out.put_pixel(x, y, decode(format, px));
        }
    }
    drop(data);
    staging.unmap();
    Ok(out)
}

fn decode(format: F, px: &[u8]) -> Rgba<f32> {
    let u8f = |v: u8| v as f32 / 255.0;
    match format {
        F::Rgba8Unorm => Rgba([u8f(px[0]), u8f(px[1]), u8f(px[2]), u8f(px[3])]),
        F::Bgra8Unorm => Rgba([u8f(px[2]), u8f(px[1]), u8f(px[0]), u8f(px[3])]),
        F::Rgba8UnormSrgb => Rgba([
            srgb_to_linear(u8f(px[0])),
            srgb_to_linear(u8f(px[1])),
            srgb_to_linear(u8f(px[2])),
            u8f(px[3]),
        ]),
        F::Bgra8UnormSrgb => Rgba([
            srgb_to_linear(u8f(px[2])),
            srgb_to_linear(u8f(px[1])),
            srgb_to_linear(u8f(px[0])),
            u8f(px[3]),
        ]),
        F::Rgba16Float => {
            let h = |i: usize| half::f16::from_le_bytes([px[2 * i], px[2 * i + 1]]).to_f32();
            Rgba([h(0), h(1), h(2), h(3)])
        }
        F::Rgba32Float => {
            let f = |i: usize| f32::from_le_bytes(px[4 * i..4 * i + 4].try_into().unwrap());
            Rgba([f(0), f(1), f(2), f(3)])
        }
        _ => unreachable!("format checked by read_texture"),
    }
}
