use brep_kernel::intersection::intersect_nurbs_surfaces;
use brep_kernel::math::{Point3, Vec2};
use brep_kernel::nurbs::NurbsSurface;
use brep_kernel::topology::{EdgeCurve3D, FaceSurface, Solid, TopologyError, TrimCurve2D};

#[test]
fn split_faces_with_curves_stages_shared_edge_without_changing_shell_counts() {
    let mut cube = Solid::cube(2.0).unwrap();
    let before_counts = cube.topology_counts();
    let before_euler = cube.euler_characteristic();
    let edge_curve = EdgeCurve3D::Polyline {
        points: vec![
            Point3::new(-0.25, -0.25, 0.0),
            Point3::new(0.0, 0.1, 0.0),
            Point3::new(0.25, 0.25, 0.0),
        ],
    };
    let a_pcurve = TrimCurve2D::Polyline {
        points: vec![
            Vec2::new(0.1, 0.2),
            Vec2::new(0.4, 0.55),
            Vec2::new(0.8, 0.9),
        ],
    };
    let b_pcurve = TrimCurve2D::Polyline {
        points: vec![
            Vec2::new(1.0, 0.2),
            Vec2::new(0.6, 0.5),
            Vec2::new(0.2, 0.85),
        ],
    };

    let report = cube
        .split_faces_with_curves(
            0,
            2,
            edge_curve.clone(),
            a_pcurve.clone(),
            b_pcurve.clone(),
            1.0e-9,
        )
        .unwrap();

    assert_eq!(report.split_edge, 0);
    assert_eq!(report.a_face, 0);
    assert_eq!(report.b_face, 2);
    assert_eq!(report.a_split, 0);
    assert_eq!(report.b_split, 0);
    assert_eq!(cube.split_edges.len(), 1);
    assert_eq!(cube.face_split_count(), 2);
    assert_eq!(cube.split_edges[report.split_edge].curve, edge_curve);
    assert_eq!(cube.faces[0].split_curves[report.a_split].pcurve, a_pcurve);
    assert_eq!(cube.faces[2].split_curves[report.b_split].pcurve, b_pcurve);
    assert_eq!(cube.topology_counts(), before_counts);
    assert_eq!(cube.euler_characteristic(), before_euler);
    cube.validate().unwrap();
}

#[test]
fn invalid_split_curve_is_rejected_without_mutating_the_solid() {
    let mut cube = Solid::cube(2.0).unwrap();
    let original = cube.clone();
    let err = cube
        .split_faces_with_curves(
            0,
            1,
            EdgeCurve3D::Unresolved,
            TrimCurve2D::LineSegment {
                start: Vec2::new(0.0, 0.0),
                end: Vec2::new(1.0, 0.0),
            },
            TrimCurve2D::LineSegment {
                start: Vec2::new(0.0, 0.0),
                end: Vec2::new(0.0, 1.0),
            },
            1.0e-9,
        )
        .unwrap_err();

    assert_eq!(err, TopologyError::InvalidSplitCurve(0));
    assert_eq!(cube, original);
}

#[test]
fn trim_ready_nurbs_intersection_can_be_installed_as_face_split() {
    let a = parabolic_x_surface();
    let b = complementary_y_surface();
    let curve = intersect_nurbs_surfaces(&a, &b, 24, 24, 1.0e-7)
        .into_iter()
        .max_by_key(|curve| curve.points.len())
        .expect("NURBS/NURBS SSI should produce a trim-ready curve");
    let mut solid = Solid::cube(2.0).unwrap();
    let before_counts = solid.topology_counts();
    solid
        .set_face_surface(0, FaceSurface::Nurbs(Box::new(a)))
        .unwrap();
    solid
        .set_face_surface(1, FaceSurface::Nurbs(Box::new(b)))
        .unwrap();

    let report = curve.split_faces(&mut solid, 0, 1, 1.0e-6).unwrap();

    assert_eq!(solid.split_edges.len(), 1);
    assert_eq!(solid.face_split_count(), 2);
    assert_eq!(
        solid.faces[0].split_curves[report.a_split].pcurve,
        curve.a_pcurve
    );
    assert_eq!(
        solid.faces[1].split_curves[report.b_split].pcurve,
        curve.b_pcurve
    );
    assert_eq!(solid.topology_counts(), before_counts);
    solid.validate().unwrap();
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
