mod common;

use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::topology::Solid;
use common::{assert_closed_manifold, rectangular_box};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn random_rectangular_boxes_are_closed_genus_zero(
        sx in positive_dimension(),
        sy in positive_dimension(),
        sz in positive_dimension(),
    ) {
        let solid = rectangular_box([sx, sy, sz]);
        assert_closed_manifold(&solid);
        prop_assert_eq!(solid.euler_characteristic(), 2);
        prop_assert_eq!(solid.genus(), Some(0));
        prop_assert!(solid.signed_volume() > 0.0);
        prop_assert!(solid.volume() >= 0.0);

        let expected_volume = sx * sy * sz;
        let expected_area = 2.0 * (sx * sy + sx * sz + sy * sz);
        prop_assert!((solid.volume() - expected_volume).abs() <= 1.0e-8);
        prop_assert!((solid.surface_area() - expected_area).abs() <= 1.0e-8);
    }

    #[test]
    fn random_cube_minus_cylinders_are_closed_genus_one(
        cube_size in 0.5_f64..10.0,
        radius_fraction in 0.05_f64..0.45,
        segments in 8_usize..128,
    ) {
        let radius = cube_size * radius_fraction;
        let report = subtract_cube_cylinder(cube_size, radius, segments).expect("valid boolean inputs");
        let solid = report.solid;
        assert_closed_manifold(&solid);
        prop_assert_eq!(solid.euler_characteristic(), 0);
        prop_assert_eq!(solid.genus(), Some(1));
        prop_assert!(solid.signed_volume() > 0.0);
        prop_assert!(solid.volume() >= 0.0);
        prop_assert!(solid.volume() < cube_size.powi(3));
    }

    #[test]
    fn generated_valid_solids_preserve_general_closed_manifold_invariants(case in valid_solid_case()) {
        let (solid, expected_genus) = case.into_solid();
        assert_closed_manifold(&solid);
        prop_assert_eq!(solid.genus(), Some(expected_genus));
        prop_assert_eq!(solid.euler_characteristic(), 2 - 2 * expected_genus);
        prop_assert!(solid.volume() >= 0.0);
    }
}

#[derive(Clone, Debug)]
enum SolidCase {
    Box {
        sx: f64,
        sy: f64,
        sz: f64,
    },
    CubeMinusCylinder {
        cube_size: f64,
        radius_fraction: f64,
        segments: usize,
    },
}

impl SolidCase {
    fn into_solid(self) -> (Solid, isize) {
        match self {
            Self::Box { sx, sy, sz } => (rectangular_box([sx, sy, sz]), 0),
            Self::CubeMinusCylinder {
                cube_size,
                radius_fraction,
                segments,
            } => {
                let radius = cube_size * radius_fraction;
                let report = subtract_cube_cylinder(cube_size, radius, segments)
                    .expect("strategy emits valid boolean inputs");
                (report.solid, 1)
            }
        }
    }
}

fn positive_dimension() -> impl Strategy<Value = f64> {
    (0.05_f64..10.0).prop_map(|value| (value * 1000.0).round() / 1000.0)
}

fn valid_solid_case() -> impl Strategy<Value = SolidCase> {
    prop_oneof![
        (
            positive_dimension(),
            positive_dimension(),
            positive_dimension()
        )
            .prop_map(|(sx, sy, sz)| { SolidCase::Box { sx, sy, sz } }),
        (0.5_f64..10.0, 0.05_f64..0.45, 8_usize..128).prop_map(
            |(cube_size, radius_fraction, segments)| SolidCase::CubeMinusCylinder {
                cube_size,
                radius_fraction,
                segments,
            },
        ),
    ]
}
