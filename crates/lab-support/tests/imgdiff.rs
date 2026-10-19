use image::{Rgba, Rgba32FImage};
use lab_support::imgdiff::{heatmap, load_linear, metrics, srgb_to_linear};

fn solid(w: u32, h: u32, v: f32) -> Rgba32FImage {
    Rgba32FImage::from_pixel(w, h, Rgba([v, v, v, 1.0]))
}

#[test]
fn identical_images_have_zero_error_and_infinite_psnr() {
    let a = solid(4, 4, 0.5);
    let m = metrics(&a, &a).unwrap();
    assert_eq!(m.mse, 0.0);
    assert_eq!(m.rel_mse, 0.0);
    assert!(m.psnr.is_infinite());
}

#[test]
fn constant_offset_gives_known_metrics() {
    let test = solid(8, 8, 0.6);
    let reference = solid(8, 8, 0.5);
    let m = metrics(&test, &reference).unwrap();
    assert!((m.mse - 0.01).abs() < 1e-6, "mse {}", m.mse);
    // relative MSE uses the reference: (0.1^2) / (0.5^2 + 0.01)
    assert!((m.rel_mse - 0.01 / 0.26).abs() < 1e-6, "rel_mse {}", m.rel_mse);
    assert!((m.psnr - 20.0).abs() < 1e-4, "psnr {}", m.psnr);
}

#[test]
fn size_mismatch_is_an_error() {
    assert!(metrics(&solid(4, 4, 0.5), &solid(3, 4, 0.5)).is_err());
}

#[test]
fn heatmap_is_black_where_equal_and_white_at_full_scale() {
    let a = solid(2, 1, 0.5);
    let mut b = solid(2, 1, 0.5);
    b.put_pixel(1, 0, Rgba([0.75, 0.75, 0.75, 1.0]));
    let map = heatmap(&a, &b, 0.25).unwrap();
    assert_eq!(map.get_pixel(0, 0).0, [0, 0, 0]);
    assert_eq!(map.get_pixel(1, 0).0, [255, 255, 255]);
}

#[test]
fn srgb_decoding_matches_reference_values() {
    assert_eq!(srgb_to_linear(0.0), 0.0);
    assert_eq!(srgb_to_linear(1.0), 1.0);
    assert!((srgb_to_linear(0.5) - 0.21404).abs() < 1e-4);
}

#[test]
fn load_linear_decodes_png_as_srgb_and_exr_as_linear() {
    let dir = std::env::temp_dir().join(format!("lab-imgdiff-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let png = dir.join("grey.png");
    image::RgbaImage::from_pixel(1, 1, Rgba([128, 128, 128, 255])).save(&png).unwrap();
    let decoded = load_linear(&png).unwrap();
    let expected = srgb_to_linear(128.0 / 255.0);
    assert!((decoded.get_pixel(0, 0).0[0] - expected).abs() < 1e-3);

    let exr = dir.join("quarter.exr");
    solid(1, 1, 0.25).save(&exr).unwrap();
    let decoded = load_linear(&exr).unwrap();
    assert!((decoded.get_pixel(0, 0).0[0] - 0.25).abs() < 1e-6);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn save_linear_round_trips_through_png_and_exr() {
    use lab_support::imgdiff::save_linear;
    let dir = std::env::temp_dir().join(format!("lab-save-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let img = solid(2, 2, 0.25);

    let png = dir.join("q.png");
    save_linear(&img, &png).unwrap();
    let back = load_linear(&png).unwrap();
    assert!((back.get_pixel(1, 1).0[1] - 0.25).abs() < 3e-3, "png gave {:?}", back.get_pixel(1, 1));

    let exr = dir.join("q.exr");
    save_linear(&img, &exr).unwrap();
    let back = load_linear(&exr).unwrap();
    assert!((back.get_pixel(1, 1).0[1] - 0.25).abs() < 1e-6);

    std::fs::remove_dir_all(&dir).ok();
}
