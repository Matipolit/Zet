use lab_support::bench::{compare, from_csv, regressions, summarize, to_csv, Delta, Sample};

fn s(frame: u64, scope: &str, ms: f64) -> Sample {
    Sample { frame, scope: scope.to_string(), ms }
}

#[test]
fn csv_round_trips_samples() {
    let samples = vec![s(0, "trace", 1.5), s(0, "blit", 0.25), s(1, "trace", 1.75)];
    let csv = to_csv(&samples);
    assert!(csv.starts_with("frame,scope,ms\n"), "csv was: {csv}");
    assert_eq!(from_csv(&csv).unwrap(), samples);
}

#[test]
fn from_csv_rejects_a_foreign_header() {
    assert!(from_csv("a,b,c\n0,x,1\n").is_err());
}

#[test]
fn summarize_reports_median_p95_mean_min_in_first_seen_order() {
    let mut samples = vec![s(0, "b", 3.0), s(1, "b", 1.0)];
    for i in 1..=100 {
        samples.push(s(i, "a", i as f64));
    }
    let sum = summarize(&samples);
    assert_eq!(sum.len(), 2);

    assert_eq!(sum[0].scope, "b");
    assert_eq!(sum[0].count, 2);
    assert_eq!(sum[0].median, 2.0);
    assert_eq!(sum[0].p95, 3.0);
    assert_eq!(sum[0].mean, 2.0);
    assert_eq!(sum[0].min, 1.0);

    assert_eq!(sum[1].scope, "a");
    assert_eq!(sum[1].count, 100);
    assert_eq!(sum[1].median, 50.5);
    assert_eq!(sum[1].p95, 95.0);
    assert_eq!(sum[1].mean, 50.5);
    assert_eq!(sum[1].min, 1.0);
}

#[test]
fn compare_reports_percent_change_of_medians() {
    let baseline: Vec<Sample> = (0..5).map(|f| s(f, "trace", 10.0)).collect();
    let mut current: Vec<Sample> = (0..5).map(|f| s(f, "trace", 11.0)).collect();
    current.push(s(0, "blit", 0.5));

    let deltas = compare(&baseline, &current);
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0].scope, "trace");
    assert_eq!(deltas[0].baseline_median, Some(10.0));
    assert_eq!(deltas[0].current_median, 11.0);
    assert!((deltas[0].percent.unwrap() - 10.0).abs() < 1e-9);
    assert_eq!(deltas[1].scope, "blit");
    assert_eq!(deltas[1].baseline_median, None);
    assert_eq!(deltas[1].percent, None);
}

#[test]
fn regressions_flags_only_slowdowns_above_threshold() {
    let d = |scope: &str, percent: Option<f64>| Delta {
        scope: scope.to_string(),
        baseline_median: Some(1.0),
        current_median: 1.0,
        percent,
    };
    let deltas = vec![
        d("slower", Some(10.0)),
        d("noise", Some(4.0)),
        d("faster", Some(-20.0)),
        d("new", None),
    ];
    let flagged: Vec<&str> = regressions(&deltas, 5.0).iter().map(|d| d.scope.as_str()).collect();
    assert_eq!(flagged, vec!["slower"]);
}

#[test]
fn table_lists_each_scope_with_median_and_p95() {
    use lab_support::bench::render_table;
    let sum = summarize(&[s(0, "trace", 2.0), s(1, "trace", 4.0)]);
    let table = render_table(&sum, None);
    assert!(table.contains("trace"), "{table}");
    assert!(table.contains("3.000"), "{table}");
    assert!(table.contains("4.000"), "{table}");
}
