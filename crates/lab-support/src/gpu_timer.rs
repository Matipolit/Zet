//! Per-pass GPU timestamps with a non-blocking readback ring.
//!
//! Usage per frame: `begin_frame` → any number of `compute_pass` / `render_pass`
//! → `end_frame(encoder)` before `finish()` → `after_submit(index)`.
//! `begin_frame` hands back the timings of the frame that used the same ring
//! slot `SLOTS` frames ago, so results arrive with a small, fixed delay and no
//! stall. `flush` drains everything that is still pending.
//!
//! Timestamps are written at pass boundaries, which only needs
//! `Features::TIMESTAMP_QUERY`. Without it the timer is inert and passes are
//! created without timestamps.

use crate::gpu::Gpu;
use std::sync::{Arc, Mutex};

pub const SLOTS: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct FrameTimings {
    pub frame: u64,
    /// `(label, milliseconds)` in the order the passes were recorded.
    pub scopes: Vec<(&'static str, f64)>,
}

impl FrameTimings {
    pub fn total_ms(&self) -> f64 {
        self.scopes.iter().map(|(_, ms)| ms).sum()
    }
}

type MapResult = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

struct Slot {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    labels: Vec<&'static str>,
    frame: u64,
    pending: Option<(wgpu::SubmissionIndex, MapResult)>,
}

pub struct GpuTimer {
    period_ns: f32,
    max_scopes: u32,
    slots: Vec<Slot>,
    current: usize,
    warned_full: bool,
}

impl GpuTimer {
    pub fn new(gpu: &Gpu, max_scopes_per_frame: u32) -> Self {
        let enabled = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let slots = if enabled {
            (0..SLOTS).map(|i| Slot::new(gpu, i, max_scopes_per_frame)).collect()
        } else {
            log::warn!("adapter has no timestamp queries; GPU timings disabled");
            Vec::new()
        };
        Self {
            period_ns: gpu.queue.get_timestamp_period(),
            max_scopes: max_scopes_per_frame,
            slots,
            current: 0,
            warned_full: false,
        }
    }

    pub fn enabled(&self) -> bool {
        !self.slots.is_empty()
    }

    /// Starts recording `frame` and returns the timings of the frame that last
    /// used this ring slot, if any.
    pub fn begin_frame(&mut self, gpu: &Gpu, frame: u64) -> Option<FrameTimings> {
        if !self.enabled() {
            return None;
        }
        let period = self.period_ns;
        let slot = &mut self.slots[self.current];
        let previous = slot.pending.take().map(|p| slot.read(gpu, p, period));
        slot.labels.clear();
        slot.frame = frame;
        previous
    }

    fn alloc(&mut self, label: &'static str) -> Option<(u32, u32)> {
        if !self.enabled() {
            return None;
        }
        let slot = &mut self.slots[self.current];
        if slot.labels.len() as u32 >= self.max_scopes {
            if !self.warned_full {
                log::warn!("GpuTimer: more than {} scopes in a frame; extra passes are untimed", self.max_scopes);
                self.warned_full = true;
            }
            return None;
        }
        let i = slot.labels.len() as u32;
        slot.labels.push(label);
        Some((2 * i, 2 * i + 1))
    }

    pub fn compute_pass<'e>(
        &mut self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &'static str,
    ) -> wgpu::ComputePass<'e> {
        let indices = self.alloc(label);
        let timestamp_writes = indices.map(|(begin, end)| wgpu::ComputePassTimestampWrites {
            query_set: &self.slots[self.current].query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        });
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes })
    }

    pub fn render_pass<'e>(
        &mut self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &'static str,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment<'_>>,
    ) -> wgpu::RenderPass<'e> {
        let indices = self.alloc(label);
        let timestamp_writes = indices.map(|(begin, end)| wgpu::RenderPassTimestampWrites {
            query_set: &self.slots[self.current].query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Records the query resolve and copy. Call before `encoder.finish()`.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled() {
            return;
        }
        let slot = &self.slots[self.current];
        let count = 2 * slot.labels.len() as u32;
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&slot.query_set, 0..count, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.staging, 0, Some(count as u64 * 8));
    }

    /// Kicks off the readback of this frame's queries and advances the ring.
    pub fn after_submit(&mut self, _gpu: &Gpu, index: wgpu::SubmissionIndex) {
        if !self.enabled() {
            return;
        }
        let slot = &mut self.slots[self.current];
        if !slot.labels.is_empty() {
            let bytes = slot.labels.len() as u64 * 16;
            let result: MapResult = Arc::new(Mutex::new(None));
            let sink = result.clone();
            slot.staging.slice(..bytes).map_async(wgpu::MapMode::Read, move |r| {
                *sink.lock().unwrap() = Some(r);
            });
            slot.pending = Some((index, result));
        }
        self.current = (self.current + 1) % SLOTS;
    }

    /// Waits for and returns every pending frame, oldest first.
    pub fn flush(&mut self, gpu: &Gpu) -> Vec<FrameTimings> {
        let period = self.period_ns;
        let mut out: Vec<FrameTimings> = self
            .slots
            .iter_mut()
            .filter_map(|slot| slot.pending.take().map(|p| slot.read(gpu, p, period)))
            .collect();
        out.sort_by_key(|t| t.frame);
        out
    }
}

impl Slot {
    fn new(gpu: &Gpu, i: usize, max_scopes: u32) -> Self {
        let count = 2 * max_scopes;
        let bytes = count as u64 * 8;
        let label = |what: &str| Some(format!("gpu timer slot {i} {what}")).map(String::leak).map(|s| &*s);
        Self {
            query_set: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: label("queries"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: label("resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: label("staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            labels: Vec::with_capacity(max_scopes as usize),
            frame: 0,
            pending: None,
        }
    }

    fn read(&mut self, gpu: &Gpu, (index, result): (wgpu::SubmissionIndex, MapResult), period_ns: f32) -> FrameTimings {
        gpu.device
            .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .expect("waiting for GPU timer readback");
        // The wait above invokes callbacks; a non-blocking poll covers any backend that defers them.
        while result.lock().unwrap().is_none() {
            gpu.device.poll(wgpu::PollType::Poll).expect("polling GPU timer readback");
        }
        let bytes = self.labels.len() as u64 * 16;
        let scopes = match result.lock().unwrap().take() {
            Some(Ok(())) => {
                let data = self.staging.slice(..bytes).get_mapped_range().expect("mapped range");
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                self.labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let dt = ticks[2 * i + 1].saturating_sub(ticks[2 * i]);
                        (*label, dt as f64 * period_ns as f64 / 1e6)
                    })
                    .collect()
            }
            other => {
                log::warn!("GPU timer readback failed: {other:?}");
                Vec::new()
            }
        };
        self.staging.unmap();
        FrameTimings { frame: self.frame, scopes }
    }
}
