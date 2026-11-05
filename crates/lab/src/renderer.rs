//! Your renderer. Everything in this file is yours to replace.
//!
//! The shell only needs the [`Renderer`] trait. It calls `render` once per frame
//! with a command encoder, then presents whatever texture `output` returns and
//! captures that same texture for `--capture` and F12. The texture is treated as
//! **linear light**: write linear radiance into it and let the shell apply the
//! sRGB transfer function on display and on PNG export.
//!
//! [`ClearRenderer`] is a placeholder that proves the shell works. Stage one of
//! `docs/learning-path.md` replaces it with a compute shader that writes the
//! output texture directly, so the recommended usage is
//! `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC` on an `Rgba16Float` texture.

use crate::shell::Frame;
use lab_support::gpu::Gpu;

pub trait Renderer {
    fn new(gpu: &Gpu, size: [u32; 2]) -> Self;

    /// The swapchain changed size. Recreate anything sized to the screen.
    fn resize(&mut self, gpu: &Gpu, size: [u32; 2]);

    /// Record this frame's GPU work. Use `frame.compute_pass(label)` and
    /// `frame.render_pass(label, ..)` so each pass is timed under `label`.
    fn render(&mut self, gpu: &Gpu, frame: &mut Frame<'_>);

    /// The texture the shell shows and captures. Needs
    /// `TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC`.
    fn output(&self) -> &wgpu::Texture;

    /// Widgets for the overlay window. Sliders go here.
    fn ui(&mut self, _ui: &mut egui::Ui) {}
}

/// Placeholder: clears the output to a colour you can pick in the overlay.
pub struct ClearRenderer {
    output: wgpu::Texture,
    color: [f32; 3],
}

impl Renderer for ClearRenderer {
    fn new(gpu: &Gpu, size: [u32; 2]) -> Self {
        Self {
            output: output_texture(gpu, size),
            color: [0.10, 0.20, 0.35],
        }
    }

    fn resize(&mut self, gpu: &Gpu, size: [u32; 2]) {
        self.output = output_texture(gpu, size);
    }

    fn render(&mut self, _gpu: &Gpu, frame: &mut Frame<'_>) {
        let view = self.output.create_view(&Default::default());
        let [r, g, b] = self.color.map(f64::from);
        let _pass = frame.render_pass(
            "clear",
            &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            None,
        );
    }

    fn output(&self) -> &wgpu::Texture {
        &self.output
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("ClearRenderer: replace me in src/renderer.rs");
        ui.horizontal(|ui| {
            ui.label("clear colour");
            ui.color_edit_button_rgb(&mut self.color);
        });
    }
}

pub struct WeekendRenderer {
    output: wgpu::Texture,
    pipeline: wgpu::RenderPipeline,
}

impl Renderer for WeekendRenderer {
    fn new(gpu: &Gpu, size: [u32; 2]) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("weekend-1.wgsl"));

        let render_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Ray Tracing W1 Pipeline"),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });

        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        Self {
            output: output_texture(gpu, size),
            pipeline: render_pipeline,
        }
    }

    fn resize(&mut self, gpu: &Gpu, size: [u32; 2]) {
        self.output = output_texture(gpu, size);
    }

    fn render(&mut self, gpu: &Gpu, frame: &mut Frame<'_>) {
        let view = self.output.create_view(&Default::default());
        let mut pass = frame.render_pass(
            "clear",
            &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            None,
        );
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }

    fn output(&self) -> &wgpu::Texture {
        &self.output
    }
}

pub fn output_texture(gpu: &Gpu, size: [u32; 2]) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("renderer output"),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
