use brep_kernel::geometry::{Line, Plane};
use brep_kernel::intersection::{
    intersect_curve_plane, intersect_line_plane, intersect_nurbs_surfaces,
    intersect_plane_nurbs_surface, intersect_plane_plane, HitKind, LinePlaneIntersection,
    PlanePlaneIntersection,
};
use brep_kernel::math::{Point3, Vec2, Vec3};
use brep_kernel::nurbs::{NurbsCurve, NurbsSurface};
use brep_kernel::predicates::{incircle2d, orient2d, orient2d_fast, orient3d, RobustSign};

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
    let polylines = intersect_nurbs_surfaces(&a, &b, 24, 24, 1.0e-7);
    assert!(!polylines.is_empty());
    assert!(polylines.iter().any(|polyline| polyline.points.len() >= 8));
    assert!(polylines
        .iter()
        .all(|polyline| polyline.max_residual < 1.0e-5));

    let total_points: usize = polylines.iter().map(|polyline| polyline.points.len()).sum();
    assert!(total_points >= 20);
    for point in polylines.iter().flat_map(|polyline| &polyline.points) {
        assert!((point.x * point.x + point.y * point.y - 0.25).abs() < 1.0e-3);
        assert!((point.z - point.x * point.x).abs() < 1.0e-3);
        assert!((point.z - (0.25 - point.y * point.y)).abs() < 1.0e-3);
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
