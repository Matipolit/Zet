use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lab-tools-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lab-tools")).args(args).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

fn solid_png(path: &Path, value: u8) {
    image::RgbaImage::from_pixel(4, 4, image::Rgba([value, value, value, 255])).save(path).unwrap();
}

#[test]
fn imgdiff_exits_zero_when_within_threshold_and_one_above_it() {
    let dir = tmp("imgdiff");
    let (a, b, heat) = (dir.join("a.png"), dir.join("b.png"), dir.join("heat.png"));
    solid_png(&a, 128);
    solid_png(&b, 160);

    let (code, text) = run(&[
        "imgdiff", a.to_str().unwrap(), a.to_str().unwrap(), "--threshold", "0.001",
    ]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("rel_mse"), "{text}");

    let (code, text) = run(&[
        "imgdiff", a.to_str().unwrap(), b.to_str().unwrap(),
        "--threshold", "0.001", "--heatmap", heat.to_str().unwrap(),
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(heat.exists(), "heatmap not written");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bench_report_exits_one_on_regression_and_zero_otherwise() {
    let dir = tmp("bench");
    let (base, cur) = (dir.join("base.csv"), dir.join("cur.csv"));
    std::fs::write(&base, "frame,scope,ms\n0,trace,10\n1,trace,10\n2,trace,10\n").unwrap();
    std::fs::write(&cur, "frame,scope,ms\n0,trace,11\n1,trace,11\n2,trace,11\n").unwrap();

    let (code, text) = run(&[
        "bench-report", cur.to_str().unwrap(), "--baseline", base.to_str().unwrap(), "--threshold", "5",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("trace"), "{text}");
    assert!(text.contains("+10.0%"), "{text}");

    let (code, text) = run(&[
        "bench-report", cur.to_str().unwrap(), "--baseline", base.to_str().unwrap(), "--threshold", "20",
    ]);
    assert_eq!(code, 0, "{text}");

    let (code, text) = run(&["bench-report", cur.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");

    std::fs::remove_dir_all(&dir).ok();
}
