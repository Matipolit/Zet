//! Command-line tools for the lab: `imgdiff` and `bench-report`.
//!
//! Both exit with status 1 when a threshold is crossed, so they work as CI gates.

use anyhow::Context;
use clap::{Parser, Subcommand};
use lab_support::bench::{compare, from_csv, regressions, render_table, summarize};
use lab_support::imgdiff::{heatmap, load_linear, metrics};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "lab-tools", about = "Measurement tools for the zet rendering lab")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare a test image against a reference. PNG is read as sRGB, EXR as linear.
    Imgdiff {
        test: PathBuf,
        reference: PathBuf,
        /// Exit 1 when relative MSE exceeds this value.
        #[arg(long)]
        threshold: Option<f64>,
        /// Write a difference heatmap (PNG) here.
        #[arg(long)]
        heatmap: Option<PathBuf>,
        /// Absolute per-channel difference shown as white in the heatmap.
        #[arg(long, default_value_t = 0.1)]
        heatmap_scale: f32,
    },
    /// Summarize a benchmark CSV written by `lab --bench`, optionally against a baseline.
    BenchReport {
        current: PathBuf,
        /// Baseline CSV to compare medians against.
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Exit 1 when any scope's median is slower than the baseline by more than this percent.
        #[arg(long, default_value_t = 5.0)]
        threshold: f64,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(ok) => {
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

/// Returns `Ok(false)` when a threshold was crossed.
fn run(cli: Cli) -> anyhow::Result<bool> {
    match cli.cmd {
        Cmd::Imgdiff { test, reference, threshold, heatmap: heatmap_path, heatmap_scale } => {
            let a = load_linear(&test)?;
            let b = load_linear(&reference)?;
            let m = metrics(&a, &b)?;
            println!(
                "mse {:.6e}  rel_mse {:.6e}  psnr {:.2} dB  ({}x{})",
                m.mse,
                m.rel_mse,
                m.psnr,
                a.width(),
                a.height()
            );
            if let Some(path) = heatmap_path {
                heatmap(&a, &b, heatmap_scale)?
                    .save(&path)
                    .with_context(|| format!("saving {}", path.display()))?;
                println!("heatmap written to {}", path.display());
            }
            if let Some(t) = threshold {
                if m.rel_mse > t {
                    println!("FAIL: rel_mse {:.6e} exceeds threshold {:.6e}", m.rel_mse, t);
                    return Ok(false);
                }
                println!("ok: rel_mse within threshold {:.6e}", t);
            }
            Ok(true)
        }
        Cmd::BenchReport { current, baseline, threshold } => {
            let cur = from_csv(&std::fs::read_to_string(&current)?)
                .with_context(|| format!("reading {}", current.display()))?;
            let summaries = summarize(&cur);
            let Some(baseline) = baseline else {
                print!("{}", render_table(&summaries, None));
                return Ok(true);
            };
            let base = from_csv(&std::fs::read_to_string(&baseline)?)
                .with_context(|| format!("reading {}", baseline.display()))?;
            let deltas = compare(&base, &cur);
            print!("{}", render_table(&summaries, Some(&deltas)));
            let slow = regressions(&deltas, threshold);
            if slow.is_empty() {
                println!("ok: no scope slower than baseline by more than {threshold}%");
                return Ok(true);
            }
            for d in &slow {
                println!(
                    "FAIL: {} median {:.3} ms vs baseline {:.3} ms ({:+.1}%)",
                    d.scope,
                    d.current_median,
                    d.baseline_median.unwrap_or(f64::NAN),
                    d.percent.unwrap_or(f64::NAN)
                );
            }
            Ok(false)
        }
    }
}
