use brep_kernel::geometry::{Line, Plane};
use brep_kernel::intersection::{
    intersect_curve_plane, intersect_line_plane, intersect_nurbs_surfaces,
    intersect_plane_nurbs_surface, intersect_plane_plane, HitKind, LinePlaneIntersection,
    PlanePlaneIntersection,
};
use brep_kernel::math::{Point3, Vec2, Vec3};
use brep_kernel::nurbs::{NurbsCurve, NurbsSurface};
use brep_kernel::predicates::{incircle2d, orient2d, orient2d_fast, orient3d, RobustSign};
use brep_kernel::topology::{EdgeCurve3D, TrimCurve2D};

#[test]
fn interval_orientation_certifies_easy_cases() {
    assert_eq!(
        orient2d(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0)
        ),
        RobustSign::Positive
    );
    assert_eq!(
        orient3d(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        ),
        RobustSign::Negative
    );
}

#[test]
fn interval_orientation_refuses_near_collinear_guess() {
    let a = Vec2::new(0.0, 0.0);
    let b = Vec2::new(1.0, 1.0);
    let c = Vec2::new(2.0, 2.0 + f64::EPSILON);
    assert!(orient2d_fast(a, b, c).abs() <= 4.0 * f64::EPSILON);
    assert_eq!(orient2d(a, b, c), RobustSign::Uncertain);
}

#[test]
fn incircle_certifies_inside_point() {
    assert_eq!(
        incircle2d(
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 0.0)
        ),
        RobustSign::Positive
    );
}

#[test]
fn line_plane_and_plane_plane_intersections_work() {
    let plane = Plane::new(Point3::ZERO, Vec3::new(0.0, 0.0, 1.0));
    let line = Line::new(Point3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0));
    let LinePlaneIntersection::Point {
        point, residual, ..
    } = intersect_line_plane(line, plane, 1.0e-12)
    else {
        panic!("expected a point");
    };
    assert!(point.distance(Point3::ZERO) < 1.0e-12);
    assert!(residual.abs() < 1.0e-12);

    let other = Plane::new(Point3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let PlanePlaneIntersection::Line(axis) = intersect_plane_plane(plane, other, 1.0e-12) else {
        panic!("expected a line");
    };
    assert!(axis.direction.cross(Vec3::new(0.0, 1.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn curve_plane_intersection_classifies_crossing() {
    let curve = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
        vec![1.0, 1.0],
    )
    .unwrap();
    let plane = Plane::new(Point3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let hits = intersect_curve_plane(&curve, plane, 8, 1.0e-12);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].kind, HitKind::Crossing);
    assert!((hits[0].u - 0.5).abs() < 1.0e-12);
}

#[test]
fn plane_nurbs_surface_marching_finds_contour_segments() {
    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, -1.0, 1.0)],
        [Point3::new(-1.0, 1.0, -1.0), Point3::new(1.0, 1.0, 1.0)],
    ]);
    let plane = Plane::new(Point3::ZERO, Vec3::new(0.0, 0.0, 1.0));
    let polylines = intersect_plane_nurbs_surface(plane, &surface, 12, 4, 1.0e-10);
    assert!(!polylines.is_empty());
    assert!(polylines.iter().all(|p| p.points.len() == 2));
    assert!(polylines.iter().all(|p| p.max_residual < 1.0e-8));
}

#[test]
fn nurbs_nurbs_surface_intersection_finds_curved_curve() {
    let a = parabolic_x_surface();
    let b = complementary_y_surface();
    let curves = intersect_nurbs_surfaces(&a, &b, 24, 24, 1.0e-7);
    assert!(!curves.is_empty());
    assert!(curves.iter().any(|curve| curve.points.len() >= 8));
    assert!(curves.iter().all(|curve| curve.max_residual < 1.0e-5));

    let total_points: usize = curves.iter().map(|curve| curve.points.len()).sum();
    assert!(total_points >= 20);
    for curve in &curves {
        assert_trim_ready_curve(curve);
    }
    for sample in curves.iter().flat_map(|curve| &curve.points) {
        let point = sample.point;
        assert!((point.x * point.x + point.y * point.y - 0.25).abs() < 1.0e-3);
        assert!((point.z - point.x * point.x).abs() < 1.0e-3);
        assert!((point.z - (0.25 - point.y * point.y)).abs() < 1.0e-3);
        assert!((0.0..=1.0).contains(&sample.a_uv.x));
        assert!((0.0..=1.0).contains(&sample.a_uv.y));
        assert!((0.0..=1.0).contains(&sample.b_uv.x));
        assert!((0.0..=1.0).contains(&sample.b_uv.y));
    }
}

fn assert_trim_ready_curve(curve: &brep_kernel::intersection::TrimReadyIntersectionCurve) {
    assert_eq!(curve.to_polyline().points.len(), curve.points.len());
    match &curve.edge_curve {
        EdgeCurve3D::LineSegment { start, end } => {
            assert_eq!(curve.points.len(), 2);
            assert_eq!(*start, curve.points[0].point);
            assert_eq!(*end, curve.points[1].point);
        }
        EdgeCurve3D::Polyline { points } => {
            assert_eq!(points.len(), curve.points.len());
            assert_eq!(points[0], curve.points[0].point);
        }
        other => panic!("unexpected trim-ready edge curve: {other:?}"),
    }
    assert_pcurve_matches_samples(&curve.a_pcurve, &curve.points, true);
    assert_pcurve_matches_samples(&curve.b_pcurve, &curve.points, false);
}

fn assert_pcurve_matches_samples(
    pcurve: &TrimCurve2D,
    samples: &[brep_kernel::intersection::SurfaceSurfaceIntersectionPoint],
    first_surface: bool,
) {
    let uv = |index: usize| {
        if first_surface {
            samples[index].a_uv
        } else {
            samples[index].b_uv
        }
    };
    match pcurve {
        TrimCurve2D::LineSegment { start, end } => {
            assert_eq!(samples.len(), 2);
            assert_eq!(*start, uv(0));
            assert_eq!(*end, uv(1));
        }
        TrimCurve2D::Polyline { points } => {
            assert_eq!(points.len(), samples.len());
            assert_eq!(points[0], uv(0));
            assert_eq!(points[points.len() - 1], uv(samples.len() - 1));
        }
        other => panic!("unexpected trim-ready p-curve: {other:?}"),
    }
}

fn parabolic_x_surface() -> NurbsSurface {
    quadratic_surface(|x_square, _y_square| x_square)
}

fn complementary_y_surface() -> NurbsSurface {
    quadratic_surface(|_x_square, y_square| 0.25 - y_square)
}

fn quadratic_surface(z_control: impl Fn(f64, f64) -> f64) -> NurbsSurface {
    let linear = [-1.0, 0.0, 1.0];
    let square = [1.0, -1.0, 1.0];
    let mut points = Vec::with_capacity(9);
    for j in 0..3 {
        for i in 0..3 {
            points.push(Point3::new(
                linear[i],
                linear[j],
                z_control(square[i], square[j]),
            ));
        }
    }
    NurbsSurface::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        3,
        3,
        points,
        vec![1.0; 9],
    )
    .unwrap()
}
