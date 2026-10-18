# Zet renderer: architecture design

Date: 2026-10-18
Status: draft for review
Scope: the whole renderer. Each milestone in section 16 gets its own implementation plan.

## 1. Purpose and goals

Zet is a 100% ray-traced renderer written in Rust on raw Vulkan, intended as the rendering foundation for the author's own game engine. It must render real game content: streamed open worlds, skinned characters, foliage, many lights.

Goals, in priority order:

1. **Maximum performance on ray-tracing hardware.** Nothing in the design may cost performance on tier A to make tiers B or C possible.
2. **Graceful fallback.** Without ray-tracing hardware the same scene renders with the same lighting system at lower ray budgets. There is exactly one lighting system.
3. **Production content.** Foliage, alpha-tested geometry, skinning, many instances, many lights and large worlds are design inputs, not later additions.
4. **Measurable.** Every technique in this document is either measured in a cited source or marked as a bet to be validated by a benchmark before it is kept.

Non-goals:

- Any rasterization path. Primary visibility is ray traced.
- Metal or DirectX 12 back-ends. Vulkan on Linux and Windows only.
- Neural shading inside ray-tracing shaders (cooperative vectors are NVIDIA-only on Vulkan as of this writing).
- Spectral rendering.
- Editor, tooling UI, networking. Zet is a renderer plus an asset cooker.

## 2. Hardware tiers

Detected once at device creation. A tier never changes while running.

| Tier | Requirement | Optional extensions used |
|---|---|---|
| **A** | `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline` | `VK_EXT_ray_tracing_invocation_reorder` (only if `rayTracingInvocationReorderReorderingHint` says reordering is real), `VK_EXT_opacity_micromap`, `VK_KHR_ray_tracing_position_fetch`, `VK_NV_cluster_acceleration_structure`, `VK_NV_partitioned_acceleration_structure` |
| **B** | `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` | `VK_KHR_ray_tracing_position_fetch` |
| **C** | Vulkan 1.3 compute only: buffer device address, descriptor indexing, subgroup operations, 16-bit storage | `VK_EXT_shader_object` |

Nothing depends on `VK_KHR_cooperative_matrix` or `VK_NV_cooperative_vector` on any tier.

Development targets: tier C on the author's Intel Iris Xe (RPL-U) laptop; tier A/B correctness through lavapipe, which emulates both ray-tracing extensions on the CPU; tier A performance on a discrete RTX, RDNA2+ or Arc GPU.

## 3. Core principles

These four invariants hold on every tier. The tiers differ only in which implementation is bound to each stage interface and in one ray-budget constant.

### 3.1 Rays never shade

Every ray from every pass returns the same 16-byte `HitRecord` (section 5.1) and either resumes into the next trace stage or terminates into a cache cell. Shading runs later, as material-sorted indirect dispatches over deduplicated cache cells. Consequences:

- Traversal kernels carry no material code, so their register pressure stays low. This matters on AMD, where dynamic register allocation exists only for non-inlined ray-tracing pipelines, and on tier C, where the traversal kernel is our own.
- Shading cost scales with unique surface area touched, not with ray count.
- This is the software equivalent of shader execution reordering (SER). On tier A, `hitObjectRecordFromQueryEXT` lets the same record also feed hardware reordering.

Sources: id Tech 8 (SIGGRAPH 2025), Dragon Age: The Veilguard (REAC 2025), Lumen's sorted deferred hit lighting (SIGGRAPH 2022).

### 3.2 One trace chain with a "resume at t" contract

Every world-space query runs through a chain of stages: screen-space trace → near-field BVH → far-field BVH → cache → sky. Each stage records how far the ray got; the next one resumes from there. Rays are compacted between stages, preserving order. Removing ray-tracing hardware removes a stage *implementation*, never an effect.

Sources: Lumen, Snowdrop (GDC 2024), Assassin's Creed Shadows (SIGGRAPH 2025). Snowdrop reports GI plus reflections going from 15 ms to 7 ms on Series X from this chain; Lumen reports 21% from order-preserving compaction over naive atomic compaction.

### 3.3 The world-space cache is the hub

A hash-grid cache keyed by position (at view-distance-adaptive level of detail), normal direction and surface class serves every one of these roles, and they share one key:

1. Radiance storage and ray terminator. Ray length is tied to the cell size at the hit, which bounds tail cost and reduces noise (Frostbite reports −60% tracing time; Snowdrop and AC Shadows use 2× probe spacing).
2. The shading point for 3.1.
3. A representative `HitRecord` per cell, so lights can change without new rays ("trace once, relight many": Snowdrop re-traces about 15% of probes per frame).
4. Per-cell lighting scale, to keep precision across a wide exposure range (AC Shadows).
5. Guiding store for tier A path guiding (ReSTIR PG), as a side table.
6. Irradiance SH per cell for the tier B/C gather.

### 3.4 One temporal stage that everything consumes

Temporal-gradient prepass → reprojection → per-pixel ray-rate map → load balancer → compacted ray lists. It runs *before* any lighting rays and decides where the frame's fixed ray budget goes. The denoiser's accumulation is the back half of the same stage. Every temporal consumer (denoiser history, light-list reprojection, future ReSTIR reuse) uses this single reprojection.

Source: Call of Duty variable-rate ray tracing (SIGGRAPH 2026 Advances), which reports "no variance clamping, yet no ghosting".

## 4. System overview

```
async compute queue                    main queue (per frame)
-------------------                    ----------------------
scene update                           1  primary visibility   1 ray/px, always
  culling groups -> instance buffer         -> visibility buffer (HitRecord per pixel)
  BLAS queue, budgeted by triangles         -> delta chains -> up to 3 stable planes
  one TLAS per ray class                2  resolve   material-sorted -> G-buffer, motion vectors
  (tier C: own builders)                3  temporal  gradients -> reproject -> ray-rate map
                                              -> load balance -> compacted ray lists
cache update                            4  direct    pick lights -> shadow rays via chain
  per-material shading dispatches       5  indirect  1 guided ray/px via chain -> cache
  relight-only cells                    6  specular  roughness-tiered
  multi-bounce update rays              7  gather    lighting from cache (SH / reservoirs)
  guiding lobe update (tier A)          8  denoise   accumulate -> spatial -> upscale
                                        9  composite stochastic transparency, particles
```

Step 2 is the one consequence of "100% ray traced": a material-sorted resolve pass replaces G-buffer rasterization. A G-buffer still exists; the screen-space trace stage, the denoiser guides and the temporal stage need it.

### 4.1 Crate layout

Cargo workspace:

| Crate | Responsibility | Depends on |
|---|---|---|
| `zet-vk` | `ash` device, queues, `gpu-allocator`, bindless descriptor heap, buffer device address helpers, timeline-semaphore render graph with automatic barriers and async-compute scheduling, tier detection | ash, gpu-allocator |
| `zet-contracts` | `#[repr(C)]` Rust mirrors of every GPU struct in `shaders/contracts.slang`, with size and offset assertions checked against Slang reflection in tests | bytemuck |
| `zet-scene` | Cooked-scene loading, instance management, culling groups, BLAS/TLAS build queue, light table, streaming hooks | zet-vk, zet-contracts |
| `zet-bvh` | Tier C acceleration structures: builders (via `obvhs`), node formats, serialization | obvhs, zet-contracts |
| `zet-shaders` | `build.rs` that compiles Slang modules to SPIR-V per tier (specialization through Slang interfaces), hot reload in debug builds | slang (shader-slang crate or CLI) |
| `zet-render` | The frame pipeline of section 4: every stage, the cache, the temporal stage, the denoiser | all of the above |
| `zet-cook` | Offline asset cooker (section 13) | zet-bvh, zet-contracts |
| `zet-viewer` | winit application: loads a cooked scene, exposes debug views and benchmark mode | zet-render |

Rule: a crate never reaches into another's internals. `zet-render` consumes `zet-scene` through buffers and handles only.

### 4.2 Shader organization

Slang modules under `shaders/`:

- `contracts.slang`: the structs of section 5. Single source of truth; Rust mirrors are generated or asserted against it.
- `tracer/`: `ISceneTracer` with `HwQueryTracer` (tiers A/B) and `SwCwbvhTracer` (tier C). Each stage kernel is specialized at compile time, so no tier pays for another's branches.
- `chain/`: `ITraceStage` implementations: `ScreenTrace`, `BvhTrace<T : ISceneTracer>`, `CacheTerminate`, `SkyTerminate`.
- `lighting/`: light selection (`ILightSampler`: `VisibleListSampler`, later `RestirDiSampler`), BSDF (bounded VNDF), guiding (`IGuide`: `BsdfOnly`, later `VmfGridGuide`).
- `cache/`: hashing, insertion, shading dispatch, relight, SH encode.
- `temporal/`: gradient prepass, reprojection, ray-rate, load balancer, trim.
- `denoise/`: the denoiser of section 10 (`IDenoiser`: `ZetDenoiser`, `DlssRrBridge`).
- `resolve/`: visibility resolve, stable planes, transparency composite.

Shader-side rules:

- Every ray query is initialized and advanced once, unconditionally, in straight-line code (required for Mali's compiler; harmless elsewhere).
- Live state across a trace call is kept minimal: fp16 radiance and throughput, 16-bit counters, octahedral-packed directions. Indiana Jones reports that these cuts, not SER itself, doubled SER's gain.
- Radiance accumulators are fp16 or fp32, never plain `R11G11B10` (hue shift, no pure white). The only `R11G11B10` radiance is the denoiser history of section 10.1, which is written with stochastic quantization.

## 5. Data contracts

These three structs are what every stage agrees on. Changing one is an architecture change.

### 5.1 `HitRecord` (16 bytes)

```slang
struct HitRecord {
    uint  instance_geom;  // [31:8] instance index (16M), [7:0] geometry index within instance
    uint  primitive;      // [23:0] primitive index within geometry, [31:24] flags
    uint  barycentrics;   // u, v as unorm16 x 2
    float t;              // distance along the ray; +inf = miss
};
// instance_geom == 0xFFFFFFFF   miss
// flags: bit 24 back-facing, bit 25 procedural (not triangle), bits 26..31 reserved
```

Identical on every tier. Material id is looked up from `(instance, geometry)` through the bindless instance table. A future cluster format may redefine `primitive` as (cluster, triangle); the flag bits are reserved for that.

### 5.2 `RayState` (32 bytes)

```slang
struct RayState {
    float3 origin;        // original origin, never moved
    uint   dir_oct;       // direction, octahedral 2 x snorm16
    float  t_resume;      // where the next stage starts; 0 at launch
    float  t_max;         // ray length, set by class and cache cell size at launch
    uint   source;        // [27:0] pixel index or cache slot, [31:28] ray class
    uint   flags;         // stages visited, last-bounce hint, cull mode, transmission
};
```

Ray classes: `primary`, `shadow`, `indirect`, `specular`, `cache_update`, `gradient`. The class selects the stage list (section 6.2) and the `t_max` policy. Path throughput and radiance for multi-bounce paths live in a separate per-path array, not in `RayState`.

### 5.3 `CacheCell` (64 bytes)

```slang
struct CacheCell {
    uint64_t  key;               // 0 = empty. Position cell at LOD | normal direction (3 bits) | surface class (3 bits)
    HitRecord hit;               // representative surface, used to relight without tracing
    float3    radiance;          // running estimate
    uint      sample_count;
    uint      last_touched;      // frame index, for aging and eviction
    uint      scale_flags;       // [15:0] lighting scale as f16, [31:16] flags (dirty, relight-only, leaking suspect)
    uint2     sh_l1;             // irradiance: L1 luma + L0 chroma in YCoCg, 64 bits
    uint2     reserved;
};
```

- Key: position quantized to `base_cell * 2^lod`, with `lod = floor(log2(distance_to_camera / lod_distance))`. Base cell 25 cm (id Tech 8). Normal stored as one of 6 axis directions. Surface class separates BSDF families so guiding and SH don't mix glass with diffuse.
- Table: open addressing, linear probing, at most 8 probes, then the insert is dropped (id Tech 8 does the same). Size is a memory budget per tier: 2^20 cells (64 MB) on tier C, 2^22 (256 MB) on tier A at most; section 12.
- Side tables keyed by the same slot: shading list membership (section 8.3) on all tiers; vMF guiding lobes (tier A, later milestone).

### 5.4 Per-pixel buffers

| Buffer | Content | Precision |
|---|---|---|
| Visibility | `HitRecord` | 16 B |
| G-buffer | depth, octahedral normal, roughness + metalness, albedo, specular albedo (pre-integrated BRDF, not F0), motion vector, flags (instance absent from secondary BVH, transmission) | depth f32, rest packed to 16 B |
| Stable planes ×3 | per delta branch: normal, roughness, albedo, depth, virtual motion vector, throughput, radiance | allocated only when the scene has delta materials |
| Ray-rate map | requested rays this frame, as a fixed-point rate | 8 bit |
| Gradient map | at 1/8 resolution per axis | f16 |
| RNG index | per-pixel sample index, reprojected with point sampling, incremented by rays actually cast | u32 |
| History | demodulated diffuse and specular, moments, history length | 2× R11G11B10 with stochastic quantization; full-precision debug variant |

## 6. Tracing

### 6.1 `ISceneTracer`

```slang
interface ISceneTracer {
    // Closest hit against one acceleration structure (one ray class → one TLAS).
    bool trace(RayState r, uint tlas_id, uint cull_flags, out HitRecord h);
    // Any hit, for shadow rays. Honors opacity policy of section 7.4.
    bool occluded(RayState r, uint tlas_id);
};
```

- `HwQueryTracer`: inline `RayQuery` in compute. Used for every pass on tiers A and B except the divergent multi-bounce path-tracing pass, which uses a ray-tracing pipeline on tier A so SER and (on RDNA4) dynamic register allocation apply.
- `SwCwbvhTracer`: 8-wide compressed BVH (CWBVH) traversal in compute, two-level. Kernel rules: traversal stack in shared memory with overflow to global memory; octant-ordered partial node fetch, never whole-node fetch (register spills reported by UBVH authors); a traversal iteration cap where "cap reached" is reported as an opaque hit with zero radiance (returning the partial hit leaked light in The Matrix Awakens); watertight triangle test.

### 6.2 The trace chain

```slang
interface ITraceStage {
    // Returns Hit (h filled), Continue (r.t_resume advanced, stage bit set) or Terminated (cache/sky).
    StageResult run(inout RayState r, inout HitRecord h);
};
```

Stage lists per ray class:

| Class | Stages | `t_max` |
|---|---|---|
| primary | near BVH → far BVH | camera far plane |
| shadow | screen (≤ 20 cm) → near BVH → far BVH | distance to light |
| indirect | screen → near BVH → far BVH → cache | 2× cache cell size at the origin's LOD (set at launch); see cache rejection below |
| specular | screen → near BVH → far BVH → cache (rough) | by roughness |
| cache_update | near BVH → far BVH → cache | 2× cell size |
| gradient | identical to the ray it re-traces | identical |

Cache rejection: `CacheTerminate` looks up the cell at the ray's end point using the *end point's* LOD. If the cell's normal disagrees with the surface the ray would have continued toward, or the cell is empty and the ray has not yet been extended, the stage returns `Continue` with `t_max` doubled, and the ray is re-queued once for the BVH stages in the next compaction round. A ray is extended at most once; the second rejection terminates into the sky stage's radiance with a "leak suspect" flag for the debug view.

Between stages, surviving rays are compacted preserving order: Z-order within a 128×128 pixel group, wave prefix sum, one global atomic per group.

Screen-space stage rules (from Lumen and MegaLights): hierarchical-Z traversal, never linear steps; thin-surface assumption, so a ray going behind a surface steps back to the last unoccluded point and continues into the BVH rather than reporting a hit; depth from `min(point, bilinear)`; raised thickness where the G-buffer flag says the instance is absent from the secondary BVH.

Mismatch between primary detail and secondary BVH detail (AC Shadows measured up to 50 cm): the BVH stage starts the world ray at `t_resume` from the screen stage; where the screen stage made no progress, it first casts a short epsilon ray with back-face culling, then the full ray without culling.

Far field: a second TLAS of merged, static, low-detail geometry (section 7.2). Its ray starts slightly before the near-field ray ended. Shadow rays trace the cheaper far field first.

## 7. Scene and acceleration structures

### 7.1 GPU-driven scene

- The instance buffer and every TLAS instance array are written by compute from GPU-resident scene data. No CPU readback for particles or skinned transforms.
- Culling for ray tracing is by *ray relevance* only: distance, projected solid angle, and the maximum ray reach of the frame. Never by frustum; rays need off-screen geometry. Culling groups (all parts of a building) are culled together so objects do not partially vanish from reflections.
- Culling is time-sliced over the frame on async compute; the TLAS builds overlap graphics work (The Matrix Awakens: 2 ms of build at 0.3 ms effective cost).

### 7.2 One TLAS per ray class

Instance masks are not used for class separation; The Matrix Awakens measured 44% slower near-field traversal from masking a far field inside a shared TLAS. Classes:

| TLAS | Content | Rebuild |
|---|---|---|
| near | full-detail static and dynamic, within near radius | every frame |
| far | merged HLOD meshes, static | on streaming events |
| alpha | alpha-tested geometry; ray classes that need it (shadow, specular) trace it as a second query with `t_max` = the opaque hit distance, classes that use the opaque approximation of 7.4 find it in `near` instead | every frame |
| translucent | particles, blended geometry, queried only with `t_max` = main hit distance | every frame |

Partitioned top level: instances are grouped into spatial partitions, each with its own structure rebuilt only when dirty, and a small top structure over partitions. Tier A uses `VK_NV_partitioned_acceleration_structure` where present; tiers B and C emulate it in software (per-partition BVH plus top BVH). A "global" partition holds fast-moving instances.

### 7.3 BLAS policy

| Content | Policy |
|---|---|
| static | Cooked offline (section 13). Tier A/B: driver build with `PREFER_FAST_TRACE`, compacted, serialized per device with a compatibility check and cached. Tier C: spatial splits plus reinsertion in `obvhs`, judged by SAH and end-point overlap. |
| vegetation, cloth that returns to rest | Build once with `ALLOW_UPDATE | ALLOW_COMPACTION`, compact, then refit forever; never rebuild a compacted BLAS (Indiana Jones: 1027 → 606 MB). |
| skinned characters | Refit every frame, rebuild when the refit quality metric degrades past a threshold (tracked per BLAS). |
| cluster LOD (tier A, later) | Cluster acceleration structures rebuilt from clusters each frame; cannot be refit. |

The build queue is budgeted by triangles per frame, not BLAS count, with priority aging so no request starves. Batched builds pay barriers once (The Matrix Awakens: about one third of the update time). The uncompacted BLAS is used until compaction lands; in-flight compactions are capped to bound peak memory.

Tier C builders: H-PLOC with fused collapsing to the wide format for dynamic content (HPG 2025, 1.4–1.6× faster rebuilds); offline builds for static content. Node format starts as CWBVH (80-byte nodes). The 64-byte merged-node format with 128-byte meshlet leaves (HPG 2026) is a bet: it is adopted only if a benchmark against the CWBVH kernel on real GPUs confirms the simulated 32% gain.

Hygiene, enforced by the cooker and a debug view: meshes rotated to local axes, large meshes split, tiny meshes merged, detached parts not left in a parent BLAS, and a traversal heatmap checked periodically.

### 7.4 Opacity

| Tier | Diffuse / indirect rays | Shadow and specular rays |
|---|---|---|
| A | Opacity micromaps, forced 2-state (no any-hit) | Opacity micromaps, 4-state |
| B | Opaque approximation: triangles shrunk by the mesh's average opacity, forced opaque (AC Shadows: 60% of full cost, visually close for GI) | Alpha test in any-hit, capped at 4 invocations |
| C | Same as B | Alpha test in the software traversal, capped at 4; software opacity micromaps with subtree-merging compression as a later upgrade |

Stochastic dithering of alpha is not used: AC Shadows measured no gain. MegaLights' continuation-ray scheme (cheap opaque trace first, any-hit continuation only when the hit instance needs it) is the B/C path for shadow rays once alpha-heavy content exists.

Micromap cooking: baked offline, 4-state, unknown states promoted to opaque, subdivision at most 6, per-asset limit 0.5 MB, per-level pool 128 MB, bakes longer than 30 minutes skipped. Debug view: forced 2-state plus any-hit iteration heatmap, to find meshes missing micromaps.

### 7.5 Level of detail

Milestone scope: discrete per-instance LOD for the secondary BVHs, with the mismatch mitigations of 6.2. Cluster acceleration structures (tier A) and software cluster LOD (tier C, per Benthin and Peters, HPG 2023) are a later milestone. When they arrive, temporal reuse must become LOD-aware (SIGGRAPH 2026), because a cluster switch moves the surface to a different mesh.

### 7.6 Massive animated geometry

Tetrahedral cages (AMD, HPG 2026) are the intended path for foliage and crowds: only a coarse cage animates, each tetrahedron is an affine instance of a static mini-BLAS, and the per-frame work is a top-level build over tetrahedra. It maps onto the two-level traversal already present on every tier. Not in the first milestones; the instance table reserves a "cage" instance kind.

## 8. Lighting

### 8.1 Direct lighting: visible-light lists

The first-shipped direct-lighting path, on every tier, follows MegaLights (Epic, SIGGRAPH 2025), which reports 941 shadowed area lights at 1 sample per pixel in 5.5 ms on PS5 at 1080p:

- Per 8×8 tile, a list of up to 32 lights that produced visible samples last frame, sorted with `WaveActiveMin`, read with a stochastic bilinear lookup across the 4 nearest tiles.
- Lights not on the list are sampled from a separate reservoir whose weight is capped at 20% of the total, raised to 50% where reprojection failed. A strong but occluded light can never dominate.
- Light weights are `log2(luminance(BRDF · L)) + 1`; samples below an exposure-relative threshold are dropped. The directional light is capped at 50% of samples when local lights matter.
- One spatiotemporal blue-noise value per pixel drives the whole selection loop, by warping the random number back to [0,1] after each weighted-reservoir step.
- Duplicate shadow rays (same origin, penumbra smaller than a pixel) are skipped. A 2×2 visibility bitmask per area light records which quarters were visible.
- Sampling may run at 1/4 rate (4-rooks pattern) with a depth- and normal-weighted stochastic bilinear upsample; that is the tier C default.
- Transmission (leaves, paper): flip the bias, offset toward the light, skip the screen stage, use hit distance as thickness.

Emissive meshes and the sky are reached through BSDF-sampled indirect rays, blended against the light-list result by roughness² ("poor man's MIS", Cyberpunk 2077).

### 8.2 Direct lighting, tier A upgrade: ReSTIR DI

Layered on top, as a second `ILightSampler` implementation, only after the list sampler ships and only on tier A. It uses the visible-light lists as initial candidates and includes from day one:

- Fractional visibility per reservoir, averaged with the resampling weights (Tokuyoshi, SIGGRAPH 2024 talk: error 80% → 60% at identical cost).
- Compatibility-guided neighbour selection (HPG 2026 best paper: 6–29% lower error for 2–5% cost).
- Decorrelation: duplication maps that lower the temporal confidence cap per pixel, and optionally one Metropolis mutation per reservoir, because correlated sample clumps degrade the denoiser.
- Feeding spatial samples back into the temporal loop; 8 spatial samples on disocclusion, 2 otherwise; a fixed ratio of initial candidates to lights when tuning (Cyberpunk 2077).

Decision rule: ReSTIR DI stays only if it beats the list sampler at equal frame time on the benchmark scenes at 2 or more rays per pixel. Nothing else in the design depends on it.

### 8.3 Indirect lighting

- One guided ray per pixel at the rate the ray-rate map assigns, direction from a mix of BSDF sampling (bounded VNDF, section 11) and, on tier A later, the cache's vMF guiding lobes (ReSTIR PG, SIGGRAPH Asia 2025).
- The ray runs the indirect chain and terminates into the cache. Its `HitRecord` is hashed into a cell; if the cell is new or dirty, the cell's slot is appended to the per-material shading list.
- **Cache update** (async compute): one indirect dispatch per material type over its list, budgeted at N cells per frame (id Tech 8 shades about 20k). Each entry either re-traces its multi-bounce update rays (default 4 bounces, querying the cache at depth 2, as DOOM: The Dark Ages) or, if marked relight-only, re-shades its stored `HitRecord` against the current lights without tracing. A cell is relight-only when its geometry hash has not changed; it is re-traced on a fixed cadence (every 2 s) or when flagged dirty.
- **Gather:** tiers B and C read the cell's irradiance SH and resolve it with the full-resolution normal at the upscale, so normal-map detail survives at quarter-rate sampling (id Tech 8, AC Shadows). Tier A later adds ReSTIR GI/PT reservoirs over the same hits.
- Leak control: the cache is queried only at the end of a ray, never at the origin; the cell LOD used is the hit's, not the origin's; rays continue past a cache hit when the cell's normal disagrees with the hit normal.
- Compensation for geometry missing from the BVH: an ambient-occlusion term from hit distance, scaled per the AC Shadows "probe AO" formula, applied before any screen-space AO; emissive fireflies clamped against the wave-average luminance.

### 8.4 Specular

- Roughness-tiered: below a threshold (0.2 initially), half-resolution reflection rays through the specular chain with dual-source reprojection (surface and reflected point). Above it, the diffuse SH or cache is evaluated instead of tracing (AC Shadows: 20–30% fewer specular rays).
- Delta surfaces (mirrors, glass): stable planes. The primary pass follows delta lobes up to 3 planes, storing per plane a normal, roughness, albedo, depth, throughput and a virtual motion vector (Householder product for flat mirrors, curvature-corrected otherwise). Each plane is denoised as if it were a primary surface and recombined with its throughput (NVIDIA RTXPT).

### 8.5 Sky and sun

The sun is a directional light in the light table with the 50% cap of 8.1. Sky radiance is the `SkyTerminate` stage; it writes motion vectors from camera rotation so the denoiser does not trail.

## 9. The temporal stage

Runs before any lighting ray, after the primary resolve.

1. **Gradient prepass.** At 1/8 resolution per axis, one pixel per 8×8 stratum re-traces *last frame's exact* ray (same RNG index, same origin, against this frame's scene) and compares radiance. The normalized difference is the gradient. Any approximation here produces persistent false positives (Call of Duty).
2. **Reprojection.** Per-texel rejection on depth, normal and instance id; interpolated depth or normals are not used; texels are never weighted by history length (causes swimming). Produces the mapping every later pass uses.
3. **Ray-rate map.** Above 1 ray per pixel where history is invalid or the gradient fired; below 1 where history is full and variance is low, as a fractional probability thresholded with blue noise and bucketed by variance.
4. **Load balancer.** One single-lane pass shifts per-bucket thresholds (plus optional foveated ring thresholds) so the total equals the tier's ray budget.
5. **Trim.** Writes indirect dispatch arguments and a compacted thread list (wave prefix sum plus one groupshared atomic). The RNG index is reprojected with point sampling and incremented by the number of rays actually cast, so no ray is ever repeated.

On a camera cut at less than 1 ray per pixel, the spatial filter fills holes with variance weights dropped, and spatial output is not fed back into history.

Reservoir splatting (SIGGRAPH 2025), if adopted for tier A reuse on foliage and fur, replaces step 2's back-projection with forward-projected primary hits; it still produces the same mapping for everyone else.

## 10. Denoising and upscaling

### 10.1 `ZetDenoiser` (all tiers; the only denoiser on AMD and Intel)

AMD's FSR Ray Regeneration is DirectX 12, Windows 11 and RDNA4-only, so on Vulkan this denoiser *is* the non-NVIDIA path. Recipe, from MegaLights and Call of Duty:

- Inputs: demodulated diffuse and specular, the G-buffer guides of 5.4, hit distance (itself denoised in a pre-pass before use for tracking), the reprojection of section 9.
- Temporal: 5×5 YCoCg variance clip loaded through groupshared memory, with a confidence term proportional to how far history falls outside the box; it *shortens* maximum history rather than only clamping.
- Disocclusion: for the first 4 frames assume constant variance, use more spatial taps, accumulate in tonemapped space.
- Spatial: one sparse kernel, rotated per pixel, applied after the temporal pass (not recurrent), only where relative variance exceeds a threshold. The upscaler cleans the rest.
- Specular: blend surface and virtual-point reprojection by the specular-dominant factor, with local-curvature correction; specular albedo is the pre-integrated BRDF.
- Precision: history in 2× R11G11B10 with stochastic quantization; low-precision history that stops moving is pushed 1 ULP toward the new value; a full-precision debug path exists.
- Passthrough: where this frame's samples cover most of the energy ("shading confidence"), the signal bypasses denoising and goes straight to the upscaler, keeping specular sharp.

### 10.2 Tier A: DLSS Ray Reconstruction

Through NGX on Vulkan (the `vk_denoise_dlssrr` sample is the reference). Guide buffers include the stable-plane layers as multi-layer guides. Selected at runtime when available; `ZetDenoiser` remains the fallback and the ground truth for debugging.

### 10.3 Upscaling

Zet's own temporal upscaler first; DLSS Super Resolution (NVIDIA) and the FSR 3.1 upscaler (Vulkan-capable) through FFI later. The denoiser runs at render resolution, so the upscaler is told which pixels were denoised versus passed through.

## 11. Sampling conventions

- **Sequences:** FAST spatiotemporal textures (EA SEED, I3D 2024) optimized for `ZetDenoiser`'s actual filters, plus the sample-warping trick of 8.1. Known gap: variable rays per pixel (section 9) and precomputed sequences do not combine cleanly; the fallback is a per-ray hash when the rate is not 1.
- **BSDF:** bounded VNDF sampling for GGX (Eto and Tokuyoshi, SIGGRAPH Asia 2023), with its PDF used consistently in MIS and any resampling Jacobian.
- **Textures:** stochastic texture filtering (I3D 2024/2025): one texel per lookup, chosen by filter weight, with the denoiser integrating. Ray cones provide the level of detail (slightly underestimated spread for path tracing). Wave-cooperative sharing of decoded texels is used only in coherent passes (primary resolve), never in divergent hit shading.
- **Transparency:** stochastic transparency with spatio-temporal dithering (EGSR 2025): one fragment accepted per ray by Russian roulette, with the motion-vector rectification for dithered pixels. Particles and blended geometry live in the translucent TLAS and are queried with `t_max` equal to the opaque hit distance.

## 12. Memory budgets

The frame rate collapses past about 80% of VRAM in use (DOOM: The Dark Ages), so a budget manager owns these allocations and degrades in this order: cache table size → micromap subdivision → history precision → ray budget.

| Allocation | Tier C (Iris Xe class) | Tier A (12 GB class) |
|---|---|---|
| Cache table | 2^20 cells, 64 MB | 2^22 cells, 256 MB |
| Micromap pool | none | 128 MB |
| G-buffer, visibility, temporal, history | about 60 MB at 1080p | about 120 MB at 1440p |
| Stable planes (3 layers) | allocated only with delta materials | about 100 MB at 1440p |
| Acceleration structures | scene dependent, tracked | scene dependent, tracked |

Numbers are initial budgets, validated at milestone M3.

## 13. Asset cooker (`zet-cook`)

Input: glTF 2.0 plus a Zet manifest (materials' surface class, LOD assignments, culling groups, cage definitions later). Output: a cooked scene directory loaded by `zet-scene` with no CPU-side processing.

Offline work, in order of milestone:

1. Mesh quantization, per-triangle shading records (face normal plus packed UVs, one 16-byte fetch per hit, as AC Shadows), material table, instance table, light table.
2. Static BLAS pre-builds for tier C (splits plus reinsertion), with SAH and end-point overlap reported. Mesh hygiene (axis alignment, split, merge).
3. Opacity micromap baking with the limits of 7.4.
4. Far-field HLOD merging.
5. Cluster LOD and tetrahedral cages (later milestones).

Tier A/B acceleration structures are built by the driver on first load and cached serialized per device.

## 14. Tier matrix

| Slot | A | B | C |
|---|---|---|---|
| `ISceneTracer` | `HwQueryTracer`; ray-tracing pipeline for the multi-bounce path-tracing pass | `HwQueryTracer` | `SwCwbvhTracer` |
| Reordering | SER with live-state diet and a 1-bit "last bounce" hint | material-sorted shading only | same |
| Opacity | micromaps 4-state / forced 2-state | opaque-shrink plus capped any-hit | same, software micromaps later |
| `ILightSampler` | `VisibleListSampler`, later `RestirDiSampler` | `VisibleListSampler` | `VisibleListSampler` at 1/4 rate |
| Indirect gather | cache + SH, later ReSTIR GI/PT with ReSTIR PG | cache + SH | cache + SH, lower rate |
| `IDenoiser` | `DlssRrBridge` when available, else `ZetDenoiser` | `ZetDenoiser` | `ZetDenoiser` |
| Ray budget (per pixel per frame, all classes) | 2–3 | about 1 | below 1, managed by the ray-rate map |
| TLAS partitions | `VK_NV_partitioned_acceleration_structure` or software | software | software |

Startup also accepts an override (`--tier`) for testing, and a per-scene hint for a software far-field representation where instance overlap is extreme (Lumen measured 0.94 ms with a merged SDF against 10 ms with hardware RT in such content). That representation is a later milestone.

## 15. Testing and validation

- **Reference mode.** A brute-force path tracer built on the same `ISceneTracer`, with no cache, no reuse and no denoiser, converging over frames. Every lighting feature has an automated image comparison against it on the benchmark scenes (Sponza, Bistro, San Miguel) with a relative-MSE threshold, plus a split-screen view in the viewer.
- **Bypass toggles** per stage: reuse radius 0, cache disabled, screen stage disabled, denoiser disabled, forced tier. Each toggle has a test that the image changes in the expected direction.
- **Tier parity.** The same scene on tiers A, B and C must match the reference within the same threshold; the tiers may differ in noise, never in mean.
- **Correctness CI.** lavapipe runs the tier A/B kernels on the CPU. Tier C runs on the Iris Xe laptop.
- **Acceleration structures.** Watertightness (random rays through shared edges hit exactly once), SAH and end-point overlap regression per cooked mesh, refit-versus-rebuild quality metric tests.
- **Contracts.** A test compares `zet-contracts` sizes and offsets against Slang reflection output for `contracts.slang`.
- **Determinism.** Fixed seeds produce identical images across runs on the same device; used by every image test.
- **Performance.** A benchmark mode records per-stage GPU timings on the benchmark scenes; regressions above 5% fail. Profiling follows the Nsight live-state tab and active-threads-per-warp, or Radeon Raytracing Analyzer for instance overlap.
- **Debug views.** Traversal heatmap, forced 2-state micromap heatmap, cache cell occupancy and age, ray-rate map, gradient map, stage reached per ray, history length, passthrough mask.

## 16. Milestones

Each milestone is its own spec-plan-implementation cycle. Exit criteria are testable.

| M | Content | Exit criterion |
|---|---|---|
| M0 | Workspace, `zet-vk` device and render graph, tier detection, Slang build, `zet-contracts` test, viewer window | Clears to a color on tiers A (lavapipe) and C (Iris Xe); contracts test passes |
| M1 | `zet-cook` step 1, `zet-scene` loading, primary visibility through `ISceneTracer` on all tiers, material-sorted resolve, G-buffer, `SwCwbvhTracer` kernel, CWBVH benchmark harness | Bistro visibility buffer identical across tiers; tier C primary-visibility time on Bistro at 1080p recorded on the Iris Xe as the baseline that sets the M3 ray budget |
| M2 | Trace chain with compaction, cache with shading lists and relight, visible-light-list direct lighting, indirect ray and SH gather, reference mode | Direct plus one-bounce indirect matches reference within threshold on all tiers |
| M3 | Temporal stage, `ZetDenoiser`, own upscaler, memory budget manager | Stable image under camera motion at tier C's ray budget; no variance clamping in the pipeline |
| M4 | Specular tiers, stable planes, stochastic transparency, translucent TLAS, far-field TLAS | Mirrors, glass and particles match reference |
| M5 | Production content: opacity policy per tier, micromap baking, vegetation refit policy, skinning, build queue, partitions | Foliage-heavy scene at target budget on tiers A and C |
| M6 | Tier A upgrades: SER, ray-tracing pipeline path, DLSS RR bridge, ReSTIR DI with the decision rule of 8.2, ReSTIR PG | Measured win over M5 on tier A, or feature removed |
| M7 | Scale: streaming, HLOD far field, cluster acceleration structures, tetrahedral cages, software far-field representation | Open-world test scene streams at target budget |

## 17. Decisions deferred to measurement

1. Tier C node format: CWBVH versus 64-byte merged nodes (M1 benchmark).
2. ReSTIR DI versus the list sampler at 2+ rays per pixel (M6).
3. Variable rays per pixel versus precomputed blue-noise sequences (M3).
4. Reservoir splatting for tier A temporal reuse on foliage (M6 or later).
5. Stable-plane memory at 4K (M4).
6. Cluster acceleration structures versus discrete LOD for secondary rays (M7).
7. FSR 3.1 upscaler on Vulkan as the non-NVIDIA upscaler (M3).

## 18. Key sources

Production:

- Sousa, "Fast as Hell: idTech8 Global Illumination", SIGGRAPH 2025 Advances. https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf
- Narkowicz, Costa, "MegaLights: Stochastic Direct Lighting in UE5", SIGGRAPH 2025 Advances. https://advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf
- Leblanc, Conte, "Ray Tracing the World of Assassin's Creed Shadows", SIGGRAPH 2025 Advances.
- Kuenlin, "Raytracing in Snowdrop", GDC 2024. https://media.gdcvault.com/gdc2024/Slides/GDC+slide+presentations/Kuenlib_Quentin_Raytracing_In_Snowdrop.pdf
- Wright, Narkowicz, Kelly, "Lumen: Real-time GI in UE5", SIGGRAPH 2022 Advances.
- Netzel, Costa, "Ray Tracing Open Worlds in UE5", SIGGRAPH 2022 Advances.
- Apers, "Shipping Dynamic GI in Frostbite", SIGGRAPH 2024 Advances.
- Olejnik, "Variable Rate Ray Tracing in Call of Duty", SIGGRAPH 2026 Advances. https://advances.realtimerendering.com/s2026/
- Bavoil, Indiana Jones path-tracing optimization posts, NVIDIA 2025. https://developer.nvidia.com/blog/path-tracing-optimization-in-indiana-jones-shader-execution-reordering-and-live-state-reductions/
- Khan, Stack, "A Beautiful Hell: Path Tracing in DOOM: The Dark Ages", Graphics Programming Conference 2025.
- Kozlowski, De Francesco, "Cyberpunk 2077 RT: Overdrive, ReSTIR Integration", SIGGRAPH 2023 ReSTIR course.
- NVIDIA RTXPT stable planes. https://github.com/NVIDIA-RTX/RTXPT

Research:

- Junkins et al., "Compatibility-Guided Neighbor Selection for ReSTIR", HPG 2026. https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/
- Tokuyoshi, "Efficient Visibility Reuse for Real-time ReSTIR", SIGGRAPH 2024 talk. https://gpuopen.com/download/Efficient_Visibility_Reuse_for_Real-time_ReSTIR.pdf
- Eto, Tokuyoshi, "Bounded VNDF Sampling for Smith-GGX Reflections", SIGGRAPH Asia 2023. https://doi.org/10.1145/3610543.3626163
- Zeng et al., "ReSTIR PG", SIGGRAPH Asia 2025. https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/
- Liu et al., "Reservoir Splatting", SIGGRAPH 2025. https://dqlin.xyz/pubs/2025-sig-RSP/
- Sawhney et al., "Decorrelating ReSTIR Samplers via MCMC Mutations", TOG 2024.
- Donnelly et al., "Filter-Adapted Spatio-Temporal Sampling", I3D 2024. https://github.com/electronicarts/fastnoise
- Pharr et al., "Filtering After Shading With Stochastic Texture Filtering", I3D 2024. https://arxiv.org/abs/2407.06107
- Brüll, Kern, Grosch, "Stochastic Transparency with Spatio-Temporal Dithering", EGSR 2025.
- Haydel et al., "Memory-Efficient BVHs with Merged Nodes", HPG 2026. https://graphics.cs.utah.edu/research/projects/bvh-merged-nodes/
- Barbier, Paulin, "Fused Collapsing", HPG 2025. https://wbrbr.org/publications/FusedCollapsing/
- Gruen et al., "Tetrahedral Cages for Massive Animated Geometry", HPG 2026. https://gpuopen.com/learn/ray-tracing-massive-amounts-animated-geometry/
- Benthin, Peters, cluster LOD ray tracing, HPG 2023. https://momentsingraphics.de/HPG2023.html
- Khronos, `VK_EXT_ray_tracing_invocation_reorder` proposal. https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_ray_tracing_invocation_reorder.html
- AMD, FSR Ray Regeneration requirements. https://gpuopen.com/amd-fsr-rayregeneration/
