# zet

A ray-traced renderer in Rust, built as a learning project. Two things live here:

- `docs/learning-path.md`: the plan being followed, stage by stage, with exit criteria.
- `docs/superpowers/specs/`: the architecture the project is heading toward.
- `crates/`: the lab. A window shell around a renderer you write, plus measurement tooling.

```
cargo run -p lab                   # opens a window; F1 toggles the overlay, F12 captures, Esc quits
cargo test --workspace             # tooling tests; the GPU ones use whatever adapter is present
```

See `docs/learning-path.md` for the full command list, the crate layout and the conventions
the tooling assumes.
