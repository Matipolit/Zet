//! Keyboard and mouse state for the frame, fed by winit events the UI did not consume.

use std::collections::HashSet;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default, Debug)]
pub struct Input {
    down: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>,
    mouse_down: [bool; 3],
    /// Mouse movement this frame, in device units (raw motion, not cursor position).
    pub mouse_delta: [f32; 2],
    /// Scroll this frame, in lines.
    pub scroll: f32,
    /// Cursor position in physical pixels, if the cursor is over the window.
    pub cursor: Option<[f32; 2]>,
}

#[allow(dead_code)] // queried by the renderer you write, not by the shell
impl Input {
    /// Key is currently held.
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.down.contains(&key)
    }

    /// Key went down this frame.
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    /// Left, right, middle.
    pub fn mouse_down(&self, button: usize) -> bool {
        self.mouse_down.get(button).copied().unwrap_or(false)
    }

    pub(crate) fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if self.down.insert(code) {
                                self.pressed.insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.down.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let i = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                self.mouse_down[i] = *state == ElementState::Pressed;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some([position.x as f32, position.y as f32]);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::Focused(false) => {
                self.down.clear();
                self.mouse_down = [false; 3];
            }
            _ => {}
        }
    }

    pub(crate) fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_delta[0] += delta.0 as f32;
            self.mouse_delta[1] += delta.1 as f32;
        }
    }

    pub(crate) fn end_frame(&mut self) {
        self.pressed.clear();
        self.mouse_delta = [0.0; 2];
        self.scroll = 0.0;
    }
}
