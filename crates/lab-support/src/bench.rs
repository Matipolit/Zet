//! Benchmark records: one `Sample` per (frame, scope), their CSV form, summaries
//! per scope, and comparison against a baseline run.
//!
//! CSV layout is `frame,scope,ms`, one sample per line. Scope labels must not
//! contain commas.

use anyhow::{bail, ensure, Context};
use std::fmt::Write as _;

pub const CSV_HEADER: &str = "frame,scope,ms";

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub frame: u64,
    pub scope: String,
    pub ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScopeSummary {
    pub scope: String,
    pub count: usize,
    pub median: f64,
    /// Nearest-rank 95th percentile.
    pub p95: f64,
    pub mean: f64,
    pub min: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    pub scope: String,
    pub baseline_median: Option<f64>,
    pub current_median: f64,
    /// Percent change of the median versus the baseline; positive means slower.
    pub percent: Option<f64>,
}

pub fn to_csv(samples: &[Sample]) -> String {
    let mut out = String::with_capacity(samples.len() * 24 + 16);
    out.push_str(CSV_HEADER);
    out.push('\n');
    for s in samples {
        debug_assert!(!s.scope.contains(','), "scope labels must not contain commas");
        writeln!(out, "{},{},{}", s.frame, s.scope, s.ms).unwrap();
    }
    out
}

pub fn from_csv(text: &str) -> anyhow::Result<Vec<Sample>> {
    let mut lines = text.lines();
    let header = lines.next().map(str::trim).unwrap_or("");
    ensure!(header == CSV_HEADER, "expected header `{CSV_HEADER}`, found `{header}`");

    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let line_no = i + 2;
        let mut parts = line.splitn(3, ',');
        let (Some(frame), Some(scope), Some(ms)) = (parts.next(), parts.next(), parts.next()) else {
            bail!("line {line_no}: expected `frame,scope,ms`, found `{line}`");
        };
        out.push(Sample {
            frame: frame.parse().with_context(|| format!("line {line_no}: bad frame `{frame}`"))?,
            scope: scope.to_string(),
            ms: ms.parse().with_context(|| format!("line {line_no}: bad ms `{ms}`"))?,
        });
    }
    Ok(out)
}

/// Groups sample values by scope, preserving the order in which scopes first appear.
fn group(samples: &[Sample]) -> Vec<(String, Vec<f64>)> {
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for s in samples {
        match groups.iter_mut().find(|(name, _)| *name == s.scope) {
            Some((_, values)) => values.push(s.ms),
            None => groups.push((s.scope.clone(), vec![s.ms])),
        }
    }
    groups
}

pub fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Nearest-rank percentile: the value at rank `ceil(p/100 * n)`, 1-based.
pub fn percentile_nearest_rank(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

pub fn summarize(samples: &[Sample]) -> Vec<ScopeSummary> {
    group(samples)
        .into_iter()
        .map(|(scope, mut values)| {
            values.sort_by(f64::total_cmp);
            let count = values.len();
            ScopeSummary {
                scope,
                count,
                median: median(&values),
                p95: percentile_nearest_rank(&values, 95.0),
                mean: values.iter().sum::<f64>() / count as f64,
                min: values[0],
            }
        })
        .collect()
}

pub fn compare(baseline: &[Sample], current: &[Sample]) -> Vec<Delta> {
    let baseline = summarize(baseline);
    summarize(current)
        .into_iter()
        .map(|cur| {
            let baseline_median = baseline.iter().find(|b| b.scope == cur.scope).map(|b| b.median);
            let percent = baseline_median
                .filter(|b| *b > 0.0)
                .map(|b| (cur.median - b) / b * 100.0);
            Delta { scope: cur.scope, baseline_median, current_median: cur.median, percent }
        })
        .collect()
}

/// Deltas whose median slowed down by more than `threshold_percent`.
pub fn regressions(deltas: &[Delta], threshold_percent: f64) -> Vec<&Delta> {
    deltas
        .iter()
        .filter(|d| d.percent.is_some_and(|p| p > threshold_percent))
        .collect()
}

/// A Markdown table of the summaries, with a delta column when a baseline comparison is given.
pub fn render_table(summaries: &[ScopeSummary], deltas: Option<&[Delta]>) -> String {
    let mut out = String::new();
    match deltas {
        Some(_) => {
            out.push_str("| scope | n | median ms | p95 ms | mean ms | min ms | vs baseline |\n");
            out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
        }
        None => {
            out.push_str("| scope | n | median ms | p95 ms | mean ms | min ms |\n");
            out.push_str("|---|---:|---:|---:|---:|---:|\n");
        }
    }
    for s in summaries {
        write!(
            out,
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} |",
            s.scope, s.count, s.median, s.p95, s.mean, s.min
        )
        .unwrap();
        if let Some(deltas) = deltas {
            let cell = deltas
                .iter()
                .find(|d| d.scope == s.scope)
                .and_then(|d| d.percent)
                .map(|p| format!("{p:+.1}%"))
                .unwrap_or_else(|| "new".to_string());
            write!(out, " {cell} |").unwrap();
        }
        out.push('\n');
    }
    out
}
