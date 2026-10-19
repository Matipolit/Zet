// Fullscreen triangle that copies the renderer's output texture to the swapchain.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle covering the clip-space square: (-1,-1), (3,-1), (-1,3).
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// For plain (non-sRGB) swapchain formats: encode on the way out.
@fragment
fn fs_encode(in: VsOut) -> @location(0) vec4<f32> {
    let rgb = clamp(textureSample(src, samp, in.uv).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(linear_to_srgb(rgb), 1.0);
}

// For sRGB swapchain formats: the hardware encodes, so pass linear through.
@fragment
fn fs_passthrough(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(src, samp, in.uv).rgb, 1.0);
}
