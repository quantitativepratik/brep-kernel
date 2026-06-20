use brep_kernel::intersection::{intersect_nurbs_surfaces, TrimReadyIntersectionCurve};
use brep_kernel::math::{Point3, Vec2};
use brep_kernel::nurbs::NurbsSurface;
use brep_kernel::topology::{
    EdgeCurve3D, FaceSurface, Solid, TopologyError, Trim, TrimCurve2D, TrimLoop, TrimLoopKind,
    TrimReadyFaceConversionKind,
};

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

#[test]
fn closed_trim_ready_curve_installs_inner_trim_loops_on_nurbs_faces() {
    let mut solid = Solid::cube(2.0).unwrap();
    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, -1.0, -1.0)],
        [Point3::new(-1.0, 1.0, -1.0), Point3::new(1.0, 1.0, -1.0)],
    ]);
    for face in [0, 1] {
        solid
            .set_face_surface(face, FaceSurface::Nurbs(Box::new(surface.clone())))
            .unwrap();
        solid
            .set_face_trim_loops(face, vec![unit_square_loop(TrimLoopKind::Outer)])
            .unwrap();
    }
    let pcurve = closed_square_pcurve();
    let edge_curve = EdgeCurve3D::Polyline {
        points: vec![
            Point3::new(-0.5, -0.5, -1.0),
            Point3::new(-0.5, 0.5, -1.0),
            Point3::new(0.5, 0.5, -1.0),
            Point3::new(0.5, -0.5, -1.0),
            Point3::new(-0.5, -0.5, -1.0),
        ],
    };

    let report = solid
        .install_trim_ready_face_curve(
            0,
            1,
            edge_curve.clone(),
            pcurve.clone(),
            pcurve.clone(),
            1.0e-7,
        )
        .unwrap();

    assert_eq!(report.kind, TrimReadyFaceConversionKind::ClosedInnerLoops);
    assert_eq!(report.a_loop, Some(1));
    assert_eq!(report.b_loop, Some(1));
    assert_eq!(solid.faces[0].trim_loops[1].kind, TrimLoopKind::Inner);
    assert_eq!(solid.faces[1].trim_loops[1].kind, TrimLoopKind::Inner);
    solid.validate_trim_loop_nesting(0, 1.0e-7).unwrap();
    solid.validate_trim_loop_nesting(1, 1.0e-7).unwrap();

    let repeat = solid
        .install_trim_ready_face_curve(0, 1, edge_curve, pcurve.clone(), pcurve, 1.0e-7)
        .unwrap();
    assert!(repeat.merged_existing);
    assert_eq!(solid.faces[0].trim_loops.len(), 2);
    assert_eq!(solid.faces[1].trim_loops.len(), 2);
}

#[test]
fn open_trim_ready_curve_closes_boundary_gaps_and_merges_duplicate_split() {
    let mut solid = Solid::cube(2.0).unwrap();
    for face in [0, 2] {
        solid
            .set_face_trim_loops(face, vec![unit_square_loop(TrimLoopKind::Outer)])
            .unwrap();
    }
    let edge_curve =
        EdgeCurve3D::line_segment(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0));
    let a_pcurve = TrimCurve2D::LineSegment {
        start: Vec2::new(5.0e-7, 0.5),
        end: Vec2::new(1.0 - 5.0e-7, 0.5),
    };
    let b_pcurve = TrimCurve2D::LineSegment {
        start: Vec2::new(0.5, 5.0e-7),
        end: Vec2::new(0.5, 1.0 - 5.0e-7),
    };

    let trim_ready_curve = TrimReadyIntersectionCurve {
        points: Vec::new(),
        edge_curve: edge_curve.clone(),
        a_pcurve: a_pcurve.clone(),
        b_pcurve: b_pcurve.clone(),
        nurbs_fit: None,
        max_residual: 0.0,
    };
    let report = trim_ready_curve
        .install_as_trimmed_faces(&mut solid, 0, 2, 1.0e-6)
        .unwrap();
    assert_eq!(report.kind, TrimReadyFaceConversionKind::OpenSplit);
    assert_eq!(report.snapped_pcurve_endpoints, 4);
    let split = report.split.unwrap();
    let (start, end) = solid.faces[0].split_curves[split.a_split]
        .pcurve
        .endpoints()
        .unwrap();
    assert!(vec2_close(start, Vec2::new(0.0, 0.5), 1.0e-12));
    assert!(vec2_close(end, Vec2::new(1.0, 0.5), 1.0e-12));

    let repeat = solid
        .install_trim_ready_face_curve(0, 2, edge_curve, a_pcurve, b_pcurve, 1.0e-6)
        .unwrap();
    assert!(repeat.merged_existing);
    assert_eq!(solid.split_edges.len(), 1);
    assert_eq!(solid.face_split_count(), 2);
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

fn unit_square_loop(kind: TrimLoopKind) -> TrimLoop {
    TrimLoop::new(
        kind,
        vec![
            line(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            line(Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)),
            line(Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)),
            line(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
        ],
    )
}

fn closed_square_pcurve() -> TrimCurve2D {
    TrimCurve2D::Polyline {
        points: vec![
            Vec2::new(0.25, 0.25),
            Vec2::new(0.25, 0.75),
            Vec2::new(0.75, 0.75),
            Vec2::new(0.75, 0.25),
            Vec2::new(0.25, 0.25),
        ],
    }
}

fn line(start: Vec2, end: Vec2) -> Trim {
    Trim::curve(TrimCurve2D::LineSegment { start, end }, 1.0e-9)
}

fn vec2_close(a: Vec2, b: Vec2, tolerance: f64) -> bool {
    let delta = a - b;
    delta.dot(delta).sqrt() <= tolerance
}
