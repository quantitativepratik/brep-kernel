use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::math::{Vec2, Vec3};
use brep_kernel::predicates::{orient2d, orient3d, RobustSign};

#[test]
fn run_regression_corpus() {
    for case in [
        include_str!("../corpus/regression/cube_minus_cylinder.case"),
        include_str!("../corpus/regression/near_collinear_orient2d.case"),
        include_str!("../corpus/regression/near_coplanar_orient3d.case"),
    ] {
        run_case(case);
    }
}

fn run_case(case: &str) {
    let kv = parse_case(case);
    match kv.get("type").map(String::as_str) {
        Some("boolean_cube_minus_cylinder") => {
            let size = parse_f64(&kv, "size");
            let radius = parse_f64(&kv, "radius");
            let segments = parse_usize(&kv, "segments");
            let report = subtract_cube_cylinder(size, radius, segments).unwrap();
            assert_eq!(report.solid.boundary_halfedge_count(), 0);
            assert_eq!(
                report.euler_characteristic,
                parse_isize(&kv, "expect_euler")
            );
            assert_eq!(report.genus, Some(parse_isize(&kv, "expect_genus")));
        }
        Some("near_collinear_orient2d") => {
            let sign = orient2d(
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(2.0, 2.0 + f64::EPSILON),
            );
            assert_eq!(sign, RobustSign::Uncertain);
        }
        Some("near_coplanar_orient3d") => {
            let sign = orient3d(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.25, 0.25, 0.0),
            );
            assert_eq!(sign, RobustSign::Uncertain);
        }
        other => panic!("unknown regression type: {other:?}"),
    }
}

fn parse_case(case: &str) -> std::collections::HashMap<String, String> {
    case.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

fn parse_f64(kv: &std::collections::HashMap<String, String>, key: &str) -> f64 {
    kv[key].parse().unwrap()
}

fn parse_usize(kv: &std::collections::HashMap<String, String>, key: &str) -> usize {
    kv[key].parse().unwrap()
}

fn parse_isize(kv: &std::collections::HashMap<String, String>, key: &str) -> isize {
    kv[key].parse().unwrap()
}
