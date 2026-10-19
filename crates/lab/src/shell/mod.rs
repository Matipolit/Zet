//! The window shell. You should not need to edit anything in this directory to
//! do the work in `docs/learning-path.md`; it owns the window, the surface, the
//! debug UI, timings, capture and benchmark modes, and presents whatever
//! texture your renderer returns from `Renderer::output`.

pub mod app;
pub mod blit;
pub mod input;

pub use app::{run, Frame};
