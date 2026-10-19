# Zet learning path

Date: 2026-10-18
Status: active. This is the plan that is being followed; the architecture spec in
`superpowers/specs/2026-10-18-zet-renderer-architecture-design.md` is the destination it leads to.

## Why this path

The architecture spec describes techniques that large teams shipped over years each. It is a
good map of where real-time ray tracing is; it is not a route from a first renderer to there.
This document is the route. Each stage ends with a picture on screen and a measurement, and
each one is a toy-scale version of a stage in the spec, so by the end the spec reads as a
checklist rather than a wall.

The Vulkan tutorials feel exhausting because they teach two things at once: the GPU
programming model, and Vulkan's vocabulary for it. This path separates them. The model is
learned in wgpu, a Rust API that runs on Vulkan underneath and needs a few hundred lines for
a window plus a compute shader. Vulkan proper comes at stage five, when every object it
introduces already has a familiar job.

## Ground rules

1. **Hand-write the renderer.** Shaders, the BVH builder, camera and sampling math, the
   temporal and denoising logic, and later the Vulkan device layer are written by hand.
   The learning is in writing them.
2. **Delegate what you would copy from a tutorial anyway.** Benchmark harnesses, image
   diff tools, PNG dumping, file loaders, build scripts, test scaffolding, and code review
   are jobs for AI. Rule of thumb: hand-write anything whose behaviour you will need to
   reason about later; delegate the rest.
3. **CPU for tests, GPU for pictures.** Keep the core math, ray–triangle intersection, BVH
   traversal and sampling routines, as plain Rust functions with unit tests, then
   transliterate to the shader. The spec's reference mode grows from this CPU copy.
4. **WGSL now, Slang at stage five.** Rust-GPU is tempting and is a toolchain detour.
5. **Measure every stage.** The lab tooling exists so that "it feels faster" never has to be
   said. Every stage's exit criterion is a number or an image comparison.
6. **Shelve the spec until stage six.** Read it for curiosity, not as a to-do list.

## The workspace

```
crates/lab            the app: window shell (shell/) + your renderer (renderer.rs)
crates/lab-support    tooling library: GPU timers, capture, image metrics, benchmark records
crates/lab-tools      CLI: imgdiff, bench-report
bench/                benchmark CSVs (ignored by git)
captures/             screenshots (ignored by git)
docs/                 this file and the architecture spec
```

Yours: `crates/lab/src/renderer.rs`, `crates/lab/src/main.rs` (one line names the active
renderer), and any shader files you add under `crates/lab/src/`. The shell in
`crates/lab/src/shell/` and the two tooling crates are maintained for you.

Commands:

```
cargo run -p lab                                            # window with the overlay (F1 toggles it)
cargo run -p lab -- --adapter llvmpipe                      # force the CPU Vulkan driver
cargo run -p lab -- --frames 64 --capture captures/out.png  # render 64 frames, save, exit
cargo run -p lab -- --frames 64 --capture captures/out.exr  # same, linear HDR
cargo run -p lab -- --bench spheres --frames 300            # per-pass GPU ms -> bench/spheres.csv
cargo run -p lab-tools -- imgdiff captures/out.png refs/out.png --threshold 1e-3 --heatmap captures/diff.png
cargo run -p lab-tools -- bench-report bench/spheres.csv --baseline bench/spheres-before.csv
cargo test --workspace
```

Conventions the tooling assumes: the renderer's output texture holds linear light; the
shell applies sRGB on display and on PNG export; EXR files stay linear; `imgdiff` compares in
linear space and reports MSE, relative MSE and PSNR. Every pass created through
`frame.compute_pass(label)` or `frame.render_pass(label, ..)` is timed under that label.

## Stages

| Stage | You build | Exit criterion | Spec counterpart |
|---|---|---|---|
| 1 | Book-one path tracer in a compute shader | Converged spheres image matches a CPU render of the same scene within rel-MSE 1e-3 | primary visibility, materials |
| 2 | Triangles, a BVH built on the CPU, traversed on the GPU | Bistro renders; primary rays per second on the Iris Xe recorded in `bench/` | tier C tracer, `zet-bvh` |
| 3 | Lights, shadow rays, MIS, GGX | Matches the CPU reference on Bistro within rel-MSE 1e-2 at equal samples | direct lighting, sampling conventions |
| 4 | One sample per pixel per frame, moving camera, reprojection, denoiser | Stable image under camera motion at one sample per pixel; ghosting visible when history is forced on | temporal stage, `ZetDenoiser` |
| 5 | Host layer ported to ash, shaders to Slang | Stage-four image bit-identical (same seeds) and per-pass times within 5% of the wgpu version | `zet-vk`, `zet-shaders` |
| 6 | Read the spec with its review; decide milestone M0 | Written list of spec changes | M0 |

Rough sizes assume evenings and weekends: stage one is a weekend or two, two and three are a
few weeks each, four is a month, five is two to three weeks.

### Stage 1: book one, on the GPU

Build: a compute shader that writes the output texture. Camera from book one, spheres in a
storage buffer, the three materials, antialiasing by jittering, depth of field, progressive
accumulation so the image refines while you watch. A reset button or key clears the
accumulation when a slider changes.

Keep the CPU version: the same scene rendered by a plain Rust port of book one, written as
you go. It is the reference image for the exit criterion and the debugging aid when the GPU
version is wrong. `imgdiff` compares the two.

What you learn: storage buffers and textures, bind groups, workgroups, why there is no
recursion, how a random number generator works on a GPU, and what one sample per pixel
looks like.

Read: Ray Tracing in One Weekend, Learn Wgpu (the compute chapter), the WGSL spec as a reference.

### Stage 2: triangles and a BVH

Build: a glTF loader (delegate it), a binned-SAH BVH builder in Rust written by hand, a flat
node array uploaded to the GPU, and a stack-based traversal loop in the shader. Load Sponza
first, Bistro after. Measure primary-ray throughput at 1080p on the Iris Xe and record it in
`bench/`; that one number sets expectations for everything after. Then compare your
builder against the `obvhs` crate on build time and traversal time, which is a job for the
benchmark tooling.

What you learn: why the acceleration structure dominates ray tracing cost, what SAH means,
watertightness, self-intersection offsets, and that incoherent rays are the expensive ones.

Read: Jacco Bikker's "How to build a BVH" series, Ray Tracing Gems chapter 6 on avoiding
self-intersection.

### Stage 3: book three, on the GPU

Build: emissive triangles and a light list, next-event estimation with one shadow ray per
sample, multiple importance sampling between BSDF and light sampling, and a GGX microfacet
material. The CPU reference gets the same features.

What you learn: the probability-density bookkeeping that every later technique depends on.
MegaLights and ReSTIR are both "pick a better light sample", and this is where you learn
what a sample is worth.

Read: Ray Tracing in One Weekend book three, PBRT fourth edition chapters on Monte Carlo
integration and sampling, Boksansky's "Crash Course in BRDF Implementation".

### Stage 4: from progressive to real-time

Build: one sample per pixel per frame with a flying camera, which breaks everything. Add
motion vectors, reproject last frame's result, reject history on depth and normal
mismatch, add an à-trous spatial filter with variance guidance, and a simple TAA. Keep a
toggle for every piece so each artefact can be shown on demand.

What you learn: the vocabulary becomes yours. Ghosting, disocclusion, boiling, variance
clamping, and why the spec spends a whole section on one temporal stage.

Read: the SVGF paper, Alain Galvan's denoising overview.

### Stage 5: port the host layer to ash

By now wgpu's edges show: bindless resources, buffer device addresses, timeline semaphores,
hardware ray queries if a card is available. Each is a reason a Vulkan feature exists.
Follow vkguide.dev rather than vulkan-tutorial; it draws with compute first and uses the
modern API surface the spec assumes. Port the shaders to Slang mechanically. The exit
criterion is that nothing visible changed and the timings match.

### Stage 6: the spec, again

Read the architecture spec and its review together, with four stages of experience. Write
down what the spec gets wrong for this hardware and this developer, and decide what M0 is.
That becomes the first real milestone plan.

## Where AI fits, concretely

- Review each stage's code before moving on. Ask for the review to name the production
  change that would break each test, not for compliments.
- Build the benchmark and image-diff tooling, and extend it when a stage needs more.
- Explain validation-layer and compiler errors; derive a PDF or a Jacobian step by step
  when the math stalls. Ask it to derive, not to paste.
- Write loaders, scripts and scaffolding.

## Performance expectations

Recorded here so later measurements have something to be compared against.

The governing arithmetic: pixels × rays per pixel × frames per second must fit inside the
hardware's ray throughput, with about a third of the frame left for everything that is not
tracing. On the Iris Xe, with no ray-tracing hardware, a good software traversal kernel is
expected to land near 100 million coherent primary rays per second and 30 to 60 million
incoherent rays per second, which makes native 1080p impossible for the full pipeline and
puts the realistic target at about 30 frames per second near 540p internal resolution in a
Bistro-class scene. The Steam Deck has hardware ray queries and is expected to land near
30 frames per second at 800p with upscaling. Stage two replaces these guesses with
measurements.

## Changing this document

Edit it when a stage's exit criterion turns out wrong, when a stage is finished (record the
measurement next to it), or when the order changes. It is the plan, not a record of intent.
