//! Event loop, surface, debug overlay, benchmark and capture modes.

use crate::renderer::Renderer;
use crate::shell::blit::Blit;
use crate::shell::input::Input;
use crate::Args;
use anyhow::Context;
use lab_support::bench::{render_table, summarize, to_csv, Sample};
use lab_support::capture::read_texture;
use lab_support::gpu::Gpu;
use lab_support::gpu_timer::{FrameTimings, GpuTimer};
use lab_support::imgdiff::save_linear;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Everything a renderer gets for one frame.
#[allow(dead_code)] // consumed by the renderer you write, not by the shell
pub struct Frame<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub timer: &'a mut GpuTimer,
    /// Frames since the application started.
    pub index: u64,
    /// Seconds since the application started.
    pub time: f32,
    /// Seconds since the previous frame.
    pub dt: f32,
    pub input: &'a Input,
    /// Swapchain size in pixels.
    pub size: [u32; 2],
}

#[allow(dead_code)] // consumed by the renderer you write, not by the shell
impl Frame<'_> {
    /// A compute pass whose GPU time shows up in the overlay and in benchmarks under `label`.
    pub fn compute_pass(&mut self, label: &'static str) -> wgpu::ComputePass<'_> {
        self.timer.compute_pass(self.encoder, label)
    }

    /// A render pass whose GPU time shows up in the overlay and in benchmarks under `label`.
    pub fn render_pass(
        &mut self,
        label: &'static str,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment<'_>>,
    ) -> wgpu::RenderPass<'_> {
        self.timer.render_pass(self.encoder, label, color_attachments, depth_stencil_attachment)
    }
}

pub fn run<R: Renderer>(args: Args) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::<R>::new(args);
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

struct Gfx<R> {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    timer: GpuTimer,
    blit: Blit,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    renderer: R,
}

struct App<R: Renderer> {
    args: Args,
    gfx: Option<Gfx<R>>,
    input: Input,
    start: Instant,
    last_frame: Instant,
    frame: u64,
    recorded: Vec<Sample>,
    latest: Option<FrameTimings>,
    cpu_ms: f32,
    show_overlay: bool,
    finished: bool,
    error: Option<anyhow::Error>,
}

impl<R: Renderer> App<R> {
    fn new(args: Args) -> Self {
        Self {
            args,
            gfx: None,
            input: Input::default(),
            start: Instant::now(),
            last_frame: Instant::now(),
            frame: 0,
            recorded: Vec::new(),
            latest: None,
            cpu_ms: 0.0,
            show_overlay: true,
            finished: false,
            error: None,
        }
    }

    fn bench(&self) -> bool {
        self.args.bench.is_some()
    }

    /// Total frames to run before exiting, if a limit was requested.
    fn frame_limit(&self) -> Option<u64> {
        let frames = self.args.frames.or(if self.bench() { Some(300) } else { None })?;
        Some(if self.bench() { frames + self.args.warmup } else { frames })
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<Gfx<R>> {
        let attrs = Window::default_attributes()
            .with_title("zet lab")
            .with_inner_size(PhysicalSize::new(self.args.width, self.args.height));
        let window = Arc::new(event_loop.create_window(attrs)?);

        let instance = Gpu::instance_with_display(Some(window.clone()));
        let surface = instance.create_surface(window.clone())?;
        let gpu = Gpu::from_instance(instance, self.args.adapter.as_deref(), Some(&surface))?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .context("surface is not supported by this adapter")?;
        // The blit shader applies the sRGB transfer function itself, so a plain
        // 8-bit format is preferred: egui expects one, and the conversion is
        // visible in blit.wgsl rather than hidden in the swapchain format.
        let caps = surface.get_capabilities(&gpu.adapter);
        let preferred = [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm];
        if let Some(format) = preferred
            .iter()
            .find(|f| caps.formats.contains(f))
            .or_else(|| caps.formats.iter().find(|f| !f.is_srgb()))
        {
            config.format = *format;
        }
        config.present_mode = if self.args.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&gpu.device, &config);
        log::info!("surface {}x{} {:?} {:?}", config.width, config.height, config.format, config.present_mode);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, config.format, Default::default());
        let blit = Blit::new(&gpu.device, config.format);
        let timer = GpuTimer::new(&gpu, 64);
        let renderer = R::new(&gpu, [config.width, config.height]);

        Ok(Gfx { window, surface, config, gpu, timer, blit, egui_state, egui_renderer, renderer })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        if size.width == 0 || size.height == 0 {
            return;
        }
        gfx.config.width = size.width;
        gfx.config.height = size.height;
        gfx.surface.configure(&gfx.gpu.device, &gfx.config);
        gfx.renderer.resize(&gfx.gpu, [size.width, size.height]);
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.cpu_ms = if self.frame == 0 { dt * 1000.0 } else { self.cpu_ms * 0.9 + dt * 100.0 };

        let Some(gfx) = self.gfx.as_mut() else { return };
        let surface_texture = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
                return;
            }
            other => {
                log::warn!("no surface texture this frame: {other:?}");
                return;
            }
        };

        if let Some(timings) = gfx.timer.begin_frame(&gfx.gpu, self.frame) {
            if self.args.bench.is_some() && timings.frame >= self.args.warmup {
                let frame = timings.frame - self.args.warmup;
                self.recorded.extend(
                    timings.scopes.iter().map(|(scope, ms)| Sample { frame, scope: scope.to_string(), ms: *ms }),
                );
            }
            self.latest = Some(timings);
        }

        let mut encoder = gfx.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut frame = Frame {
                encoder: &mut encoder,
                timer: &mut gfx.timer,
                index: self.frame,
                time: now.duration_since(self.start).as_secs_f32(),
                dt,
                input: &self.input,
                size: [gfx.config.width, gfx.config.height],
            };
            gfx.renderer.render(&gfx.gpu, &mut frame);
        }

        let output_view = gfx.renderer.output().create_view(&Default::default());
        let surface_view = surface_texture.texture.create_view(&Default::default());
        gfx.blit.draw(&gfx.gpu.device, &mut gfx.timer, &mut encoder, &output_view, &surface_view);

        // Debug overlay.
        let raw_input = gfx.egui_state.take_egui_input(&gfx.window);
        let overlay = OverlayData {
            adapter: gfx.gpu.info().name.clone(),
            size: [gfx.config.width, gfx.config.height],
            cpu_ms: self.cpu_ms,
            latest: self.latest.clone(),
            frame: self.frame,
            bench: self.args.bench.clone(),
            recorded_frames: self.frame.saturating_sub(self.args.warmup),
        };
        let show_overlay = self.show_overlay;
        let renderer = &mut gfx.renderer;
        let ctx = gfx.egui_state.egui_ctx().clone();
        ctx.begin_pass(raw_input);
        if show_overlay {
            overlay.show(&ctx, renderer);
        }
        let full = ctx.end_pass();
        gfx.egui_state.handle_platform_output(&gfx.window, full.platform_output);
        let paint_jobs = gfx.egui_state.egui_ctx().tessellate(full.shapes, full.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gfx.config.width, gfx.config.height],
            pixels_per_point: full.pixels_per_point,
        };
        let mut textures_delta = full.textures_delta;
        for (id, deltas) in std::mem::take(&mut textures_delta.set) {
            for delta in &deltas {
                gfx.egui_renderer.update_texture(&gfx.gpu.device, &gfx.gpu.queue, id, delta);
            }
        }
        gfx.egui_renderer.update_buffers(&gfx.gpu.device, &gfx.gpu.queue, &mut encoder, &paint_jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            gfx.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in std::mem::take(&mut textures_delta.free) {
            gfx.egui_renderer.free_texture(&id);
        }

        gfx.timer.end_frame(&mut encoder);
        let index = gfx.gpu.queue.submit([encoder.finish()]);
        gfx.timer.after_submit(&gfx.gpu, index);
        gfx.window.pre_present_notify();
        gfx.gpu.queue.present(surface_texture);

        self.input.end_frame();
        self.frame += 1;

        if self.frame_limit().is_some_and(|limit| self.frame >= limit) {
            self.finish();
            event_loop.exit();
        }
    }

    /// Writes the benchmark CSV and the requested capture. Runs once.
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let Some(gfx) = self.gfx.as_mut() else { return };

        for timings in gfx.timer.flush(&gfx.gpu) {
            if self.args.bench.is_some() && timings.frame >= self.args.warmup {
                let frame = timings.frame - self.args.warmup;
                self.recorded.extend(
                    timings.scopes.iter().map(|(scope, ms)| Sample { frame, scope: scope.to_string(), ms: *ms }),
                );
            }
        }

        if let Some(name) = &self.args.bench {
            let path = PathBuf::from("bench").join(format!("{name}.csv"));
            let result = std::fs::create_dir_all("bench").and_then(|_| std::fs::write(&path, to_csv(&self.recorded)));
            match result {
                Ok(()) => {
                    println!(
                        "# {} · {}x{} · {} recorded frames after {} warm-up\n",
                        gfx.gpu.info().name,
                        gfx.config.width,
                        gfx.config.height,
                        self.recorded.iter().map(|s| s.frame).max().map_or(0, |f| f + 1),
                        self.args.warmup
                    );
                    print!("{}", render_table(&summarize(&self.recorded), None));
                    println!("\nwrote {}", path.display());
                }
                Err(e) => self.error = Some(anyhow::anyhow!("writing {}: {e}", path.display())),
            }
        }

        if let Some(path) = self.args.capture.clone() {
            if let Err(e) = capture_to(gfx, &path) {
                self.error = Some(e);
            }
        }
    }
}

fn capture_to<R: Renderer>(gfx: &Gfx<R>, path: &std::path::Path) -> anyhow::Result<()> {
    let image = read_texture(&gfx.gpu, gfx.renderer.output())?;
    save_linear(&image, path)?;
    println!("captured {}", path.display());
    Ok(())
}

impl<R: Renderer> ApplicationHandler for App<R> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_none() {
            match self.init(event_loop) {
                Ok(gfx) => self.gfx = Some(gfx),
                Err(e) => {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        let response = gfx.egui_state.on_window_event(&gfx.window, &event);
        if !response.consumed {
            self.input.window_event(&event);
        }
        match event {
            WindowEvent::CloseRequested => {
                self.finish();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, repeat: false, .. },
                ..
            } if !response.consumed => match code {
                KeyCode::Escape => {
                    self.finish();
                    event_loop.exit();
                }
                KeyCode::F1 => self.show_overlay = !self.show_overlay,
                KeyCode::F12 => {
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs());
                    let path = PathBuf::from("captures").join(format!("{stamp}.png"));
                    if let Some(gfx) = self.gfx.as_ref() {
                        if let Err(e) = capture_to(gfx, &path) {
                            log::error!("capture failed: {e:#}");
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        self.input.device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }
}

struct OverlayData {
    adapter: String,
    size: [u32; 2],
    cpu_ms: f32,
    latest: Option<FrameTimings>,
    frame: u64,
    bench: Option<String>,
    recorded_frames: u64,
}

impl OverlayData {
    fn show<R: Renderer>(&self, ctx: &egui::Context, renderer: &mut R) {
        egui::Window::new("lab")
            .default_pos([8.0, 8.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("{} · {}x{}", self.adapter, self.size[0], self.size[1]));
                ui.label(format!("frame {}  cpu {:.2} ms", self.frame, self.cpu_ms));
                match &self.latest {
                    Some(t) => {
                        egui::Grid::new("gpu").num_columns(2).show(ui, |ui| {
                            for (label, ms) in &t.scopes {
                                ui.label(*label);
                                ui.label(format!("{ms:.3} ms"));
                                ui.end_row();
                            }
                            ui.strong("gpu total");
                            ui.strong(format!("{:.3} ms", t.total_ms()));
                            ui.end_row();
                        });
                    }
                    None => {
                        ui.label("gpu timings: waiting");
                    }
                }
                if let Some(name) = &self.bench {
                    ui.label(format!("benchmark `{name}`: {} frames recorded", self.recorded_frames));
                }
                ui.separator();
                renderer.ui(ui);
                ui.separator();
                ui.small("F1 overlay · F12 capture · Esc quit");
            });
    }
}
