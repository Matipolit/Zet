//! Tests that need a GPU (or lavapipe). They skip with a message when no adapter exists.

use lab_support::capture::read_texture;
use lab_support::gpu::Gpu;
use lab_support::gpu_timer::GpuTimer;

fn gpu() -> Option<Gpu> {
    match Gpu::headless(None) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: no GPU adapter available ({e})");
            None
        }
    }
}

fn texture(gpu: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[test]
fn capture_reads_back_rgba8_texture_with_row_padding() {
    let Some(gpu) = gpu() else { return };
    // 5 px wide: 20 bytes per row, which forces the 256-byte row padding path.
    let (w, h) = (5u32, 3u32);
    let tex = texture(&gpu, w, h, wgpu::TextureFormat::Rgba8Unorm);
    let bytes: Vec<u8> = (0..(w * h * 4) as u8).collect();
    gpu.queue.write_texture(
        tex.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None },
        tex.size(),
    );

    let img = read_texture(&gpu, &tex).unwrap();
    assert_eq!((img.width(), img.height()), (w, h));
    for (i, v) in img.as_raw().iter().enumerate() {
        assert!((v - bytes[i] as f32 / 255.0).abs() < 1e-6, "byte {i}");
    }
}

#[test]
fn capture_reads_back_rgba16f_as_linear_f32() {
    let Some(gpu) = gpu() else { return };
    let tex = texture(&gpu, 3, 2, wgpu::TextureFormat::Rgba16Float);
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.125).collect();
    let halves: Vec<u8> = values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    gpu.queue.write_texture(
        tex.as_image_copy(),
        &halves,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(3 * 8), rows_per_image: None },
        tex.size(),
    );

    let img = read_texture(&gpu, &tex).unwrap();
    assert_eq!(img.as_raw(), &values);
}

#[test]
fn gpu_timer_reports_every_scope_of_every_frame_after_flush() {
    let Some(gpu) = gpu() else { return };
    let mut timer = GpuTimer::new(&gpu, 8);
    if !timer.enabled() {
        eprintln!("skipping: adapter has no timestamp queries");
        return;
    }

    let mut early = Vec::new();
    for frame in 0..5u64 {
        early.extend(timer.begin_frame(&gpu, frame));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let _pass = timer.compute_pass(&mut encoder, "noop");
        }
        {
            let _pass = timer.compute_pass(&mut encoder, "noop2");
        }
        timer.end_frame(&mut encoder);
        let index = gpu.queue.submit([encoder.finish()]);
        timer.after_submit(&gpu, index);
    }
    let mut all = early;
    all.extend(timer.flush(&gpu));

    let frames: Vec<u64> = all.iter().map(|t| t.frame).collect();
    assert_eq!(frames, vec![0, 1, 2, 3, 4]);
    for t in &all {
        let labels: Vec<&str> = t.scopes.iter().map(|(l, _)| *l).collect();
        assert_eq!(labels, vec!["noop", "noop2"]);
        assert!(t.scopes.iter().all(|(_, ms)| *ms >= 0.0 && ms.is_finite()));
    }
}
