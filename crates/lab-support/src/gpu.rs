//! Adapter selection and device creation.
//!
//! Vulkan only, by design: this lab is the on-ramp to a Vulkan renderer. Set
//! `WGPU_BACKEND` to override. The adapter is chosen by preference (discrete,
//! integrated, virtual, CPU) unless a name filter such as `llvmpipe` or `Iris`
//! is given.

use anyhow::{anyhow, Context};

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Features the shell asks for when the adapter has them. Add to this as the
/// renderer grows; missing ones are silently skipped, so check
/// `gpu.device.features()` before relying on one.
pub const OPTIONAL_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

impl Gpu {
    pub fn instance() -> wgpu::Instance {
        Self::instance_with_display::<std::sync::Arc<winit_free::NoDisplay>>(None)
    }

    /// Like [`Gpu::instance`], but tells wgpu which display the surface will be
    /// created on. Pass the window (or event loop) when there is one.
    pub fn instance_with_display<D>(display: Option<D>) -> wgpu::Instance
    where
        D: wgpu::rwh::HasDisplayHandle + std::fmt::Debug + Send + Sync + 'static,
    {
        let mut desc = match display {
            Some(display) => wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display)),
            None => wgpu::InstanceDescriptor::new_without_display_handle(),
        };
        desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::VULKAN);
        wgpu::Instance::new(desc.with_env())
    }

    /// A device with no window. Used by tests and offline tools.
    pub fn headless(adapter_filter: Option<&str>) -> anyhow::Result<Self> {
        Self::from_instance(Self::instance(), adapter_filter, None)
    }

    pub fn from_instance(
        instance: wgpu::Instance,
        adapter_filter: Option<&str>,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> anyhow::Result<Self> {
        let adapter = pick_adapter(&instance, adapter_filter, compatible_surface)?;
        let info = adapter.get_info();
        log::info!(
            "adapter: {} ({:?}, {} {})",
            info.name,
            info.device_type,
            info.driver,
            info.driver_info
        );

        let required_features = adapter.features() & OPTIONAL_FEATURES;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("lab"),
            required_features,
            required_limits: adapter.limits(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        }))
        .context("requesting device")?;

        Ok(Self { instance, adapter, device, queue })
    }

    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

fn rank(ty: wgpu::DeviceType) -> u8 {
    match ty {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        wgpu::DeviceType::Other => 4,
    }
}

fn pick_adapter(
    instance: &wgpu::Instance,
    filter: Option<&str>,
    surface: Option<&wgpu::Surface<'_>>,
) -> anyhow::Result<wgpu::Adapter> {
    let mut candidates: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .filter(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
        .collect();
    if candidates.is_empty() {
        return Err(anyhow!("no compatible GPU adapter found"));
    }

    if let Some(filter) = filter {
        let needle = filter.to_lowercase();
        candidates.retain(|a| {
            let info = a.get_info();
            info.name.to_lowercase().contains(&needle) || info.driver.to_lowercase().contains(&needle)
        });
        if candidates.is_empty() {
            return Err(anyhow!("no adapter matches `{filter}`"));
        }
    }

    candidates.sort_by_key(|a| rank(a.get_info().device_type));
    Ok(candidates.remove(0))
}

mod winit_free {
    /// Placeholder type parameter for [`super::Gpu::instance`]; never instantiated.
    #[derive(Debug)]
    pub struct NoDisplay;
    impl wgpu::rwh::HasDisplayHandle for NoDisplay {
        fn display_handle(&self) -> Result<wgpu::rwh::DisplayHandle<'_>, wgpu::rwh::HandleError> {
            Err(wgpu::rwh::HandleError::Unavailable)
        }
    }
}
