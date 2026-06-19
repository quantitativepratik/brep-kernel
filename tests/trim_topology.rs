use brep_kernel::math::{Point3, Vec2};
use brep_kernel::nurbs::{NurbsCurve, NurbsSurface};
use brep_kernel::topology::{
    EdgeCurve3D, FaceSurface, Solid, TopologyError, Trim, TrimCurve2D, TrimLoop, TrimLoopKind,
    TrimLoopOrientation,
};

#[test]
fn triangle_faces_have_projected_outer_trim_loops() {
    let cube = Solid::cube(2.0).unwrap();
    cube.validate_trim_topology().unwrap();

    let (outer, inner) = cube.trim_loop_counts();
    assert_eq!(outer, cube.faces.len());
    assert_eq!(inner, 0);

    for face in &cube.faces {
        assert!(matches!(face.surface, FaceSurface::Plane(_)));
        assert_eq!(face.trim_loops.len(), 1);
        assert_eq!(face.trim_loops[0].kind, TrimLoopKind::Outer);
        assert_eq!(face.trim_loops[0].trims.len(), 3);
        assert!(face.trim_loops[0]
            .trims
            .iter()
            .all(|trim| trim.curve.endpoints().is_some()));
    }
}

#[test]
fn analytic_inner_trim_loop_can_be_attached_to_a_face() {
    let mut cube = Solid::cube(2.0).unwrap();
    let mut loops = cube.faces[0].trim_loops.clone();
    loops.push(square_loop(
        TrimLoopKind::Inner,
        [
            Vec2::new(-0.25, -0.25),
            Vec2::new(0.25, -0.25),
            Vec2::new(0.25, 0.25),
            Vec2::new(-0.25, 0.25),
        ],
    ));

    cube.set_face_trim_loops(0, loops).unwrap();
    cube.validate_trim_topology().unwrap();

    let (_, inner) = cube.trim_loop_counts();
    assert_eq!(inner, 1);
    assert_eq!(cube.faces[0].trim_loops[1].kind, TrimLoopKind::Inner);
}

#[test]
fn open_trim_loop_is_rejected_and_old_loops_are_restored() {
    let mut cube = Solid::cube(2.0).unwrap();
    let original = cube.faces[0].trim_loops.clone();
    let mut loops = original.clone();
    loops.push(TrimLoop::new(
        TrimLoopKind::Inner,
        vec![
            line(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            line(Vec2::new(2.0, 0.0), Vec2::new(0.0, 0.0)),
        ],
    ));

    assert_eq!(
        cube.set_face_trim_loops(0, loops),
        Err(TopologyError::OpenTrimLoop(0, 1))
    );
    assert_eq!(cube.faces[0].trim_loops, original);
}

#[test]
fn face_can_carry_nurbs_surface_and_parametric_trim_loop() {
    let mut cube = Solid::cube(2.0).unwrap();
    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, 0.0), Point3::new(1.0, -1.0, 0.0)],
        [Point3::new(-1.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.4)],
    ]);

    cube.set_face_surface(0, FaceSurface::Nurbs(Box::new(surface)))
        .unwrap();
    assert!(matches!(&cube.faces[0].surface, FaceSurface::Nurbs(_)));

    cube.set_face_trim_loops(
        0,
        vec![square_loop(
            TrimLoopKind::Outer,
            [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ],
        )],
    )
    .unwrap();
    cube.validate_trim_topology().unwrap();
}

#[test]
fn trim_loop_analysis_reports_orientation_and_nesting() {
    let mut cube = Solid::cube(2.0).unwrap();
    cube.set_face_trim_loops(
        0,
        vec![
            square_loop(
                TrimLoopKind::Outer,
                [
                    Vec2::new(-1.0, -1.0),
                    Vec2::new(1.0, -1.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(-1.0, 1.0),
                ],
            ),
            square_loop(
                TrimLoopKind::Inner,
                [
                    Vec2::new(-0.25, -0.25),
                    Vec2::new(-0.25, 0.25),
                    Vec2::new(0.25, 0.25),
                    Vec2::new(0.25, -0.25),
                ],
            ),
        ],
    )
    .unwrap();

    let analysis = cube.analyze_trim_loop_nesting(0, 1.0e-9).unwrap();
    assert_eq!(
        analysis.loops[0].orientation,
        TrimLoopOrientation::CounterClockwise
    );
    assert_eq!(analysis.loops[0].parent, None);
    assert_eq!(analysis.loops[0].depth, 0);
    assert_eq!(
        analysis.loops[1].orientation,
        TrimLoopOrientation::Clockwise
    );
    assert_eq!(analysis.loops[1].parent, Some(0));
    assert_eq!(analysis.loops[1].depth, 1);
    cube.validate_trim_loop_nesting(0, 1.0e-9).unwrap();
}

#[test]
fn trim_loop_nesting_rejects_inner_loop_outside_outer_loop() {
    let mut cube = Solid::cube(2.0).unwrap();
    cube.set_face_trim_loops(
        0,
        vec![
            square_loop(
                TrimLoopKind::Outer,
                [
                    Vec2::new(-1.0, -1.0),
                    Vec2::new(1.0, -1.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(-1.0, 1.0),
                ],
            ),
            square_loop(
                TrimLoopKind::Inner,
                [
                    Vec2::new(2.0, 2.0),
                    Vec2::new(2.5, 2.0),
                    Vec2::new(2.5, 2.5),
                    Vec2::new(2.0, 2.5),
                ],
            ),
        ],
    )
    .unwrap();

    assert_eq!(
        cube.validate_trim_loop_nesting(0, 1.0e-9),
        Err(TopologyError::InvalidTrimLoopNesting(0, 1))
    );
}

#[test]
fn pcurves_can_be_generated_on_nurbs_support_surface() {
    let mut cube = Solid::cube(2.0).unwrap();
    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, -1.0, -1.0)],
        [Point3::new(-1.0, 1.0, -1.0), Point3::new(1.0, 1.0, -1.0)],
    ]);

    cube.set_face_surface(0, FaceSurface::Nurbs(Box::new(surface)))
        .unwrap();
    cube.generate_face_pcurves(0, 8, 1.0e-7).unwrap();

    for trim in &cube.faces[0].trim_loops[0].trims {
        assert!(matches!(trim.curve, TrimCurve2D::Nurbs(_)));
        assert!(trim.curve.endpoints().is_some());
    }
    cube.validate_trim_topology().unwrap();
}

#[test]
fn pcurves_project_curved_edges_on_nurbs_support_surface() {
    let mut cube = Solid::cube(2.0).unwrap();
    let surface = curved_quadratic_surface(0.32, 0.18);
    cube.set_face_surface(0, FaceSurface::Nurbs(Box::new(surface)))
        .unwrap();

    let trim_halfedges: Vec<_> = cube.faces[0].trim_loops[0]
        .trims
        .iter()
        .map(|trim| trim.halfedge.unwrap())
        .collect();
    for halfedge in trim_halfedges {
        let start = cube.vertices[cube.halfedges[halfedge].origin].point;
        let end = cube.vertices[cube.halfedges[cube.halfedges[halfedge].next].origin].point;
        let start_uv = face_zero_uv(start);
        let end_uv = face_zero_uv(end);
        let curve = boundary_curve_on_curved_surface(start_uv, end_uv, 0.32, 0.18);
        let edge = cube.halfedges[halfedge].edge;
        cube.set_edge_curve(edge, curve, 1.0e-9).unwrap();
    }

    cube.generate_face_pcurves(0, 17, 1.0e-8).unwrap();

    for trim in &cube.faces[0].trim_loops[0].trims {
        let halfedge = trim.halfedge.unwrap();
        let start = cube.vertices[cube.halfedges[halfedge].origin].point;
        let end = cube.vertices[cube.halfedges[cube.halfedges[halfedge].next].origin].point;
        let expected_mid = (face_zero_uv(start) + face_zero_uv(end)) * 0.5;
        let uv_samples = trim.curve.sample_points(5).unwrap();

        assert!(matches!(trim.curve, TrimCurve2D::Nurbs(_)));
        assert!(vec2_close(uv_samples[2], expected_mid, 1.0e-6));
    }
    cube.validate_trim_topology().unwrap();
}

#[test]
fn pcurve_generation_rejects_edges_that_leave_nurbs_surface() {
    let mut cube = Solid::cube(2.0).unwrap();
    cube.set_face_surface(
        0,
        FaceSurface::Nurbs(Box::new(curved_quadratic_surface(0.32, 0.18))),
    )
    .unwrap();

    let halfedge = cube.faces[0].trim_loops[0].trims[0].halfedge.unwrap();
    let edge = cube.halfedges[halfedge].edge;
    let start = cube.vertices[cube.halfedges[halfedge].origin].point;
    let end = cube.vertices[cube.halfedges[cube.halfedges[halfedge].next].origin].point;
    let original_trim = cube.faces[0].trim_loops.clone();
    cube.set_edge_curve(
        edge,
        EdgeCurve3D::Polyline {
            points: vec![start, Point3::new(0.0, 0.0, 0.2), end],
        },
        1.0e-6,
    )
    .unwrap();

    assert_eq!(
        cube.generate_face_pcurves(0, 9, 1.0e-8),
        Err(TopologyError::PcurveProjectionFailed(0, halfedge))
    );
    assert_eq!(cube.faces[0].trim_loops, original_trim);
}

fn square_loop(kind: TrimLoopKind, points: [Vec2; 4]) -> TrimLoop {
    TrimLoop::new(
        kind,
        vec![
            line(points[0], points[1]),
            line(points[1], points[2]),
            line(points[2], points[3]),
            line(points[3], points[0]),
        ],
    )
}

fn line(start: Vec2, end: Vec2) -> Trim {
    Trim::curve(TrimCurve2D::LineSegment { start, end }, 1.0e-9)
}

fn curved_quadratic_surface(u_bulge: f64, v_bulge: f64) -> NurbsSurface {
    let coords = [-1.0, 0.0, 1.0];
    let mut points = Vec::with_capacity(9);
    for j in 0..3 {
        for i in 0..3 {
            let z = -1.0
                + if i == 1 { u_bulge * 0.5 } else { 0.0 }
                + if j == 1 { v_bulge * 0.5 } else { 0.0 };
            points.push(Point3::new(coords[i], coords[j], z));
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

fn face_zero_uv(point: Point3) -> Vec2 {
    if vec3_close(point, Point3::new(-1.0, -1.0, -1.0), 1.0e-12) {
        Vec2::new(0.0, 0.0)
    } else if vec3_close(point, Point3::new(1.0, -1.0, -1.0), 1.0e-12) {
        Vec2::new(1.0, 0.0)
    } else if vec3_close(point, Point3::new(1.0, 1.0, -1.0), 1.0e-12) {
        Vec2::new(1.0, 1.0)
    } else {
        panic!("unexpected face-zero vertex: {point:?}");
    }
}

fn boundary_curve_on_curved_surface(
    start_uv: Vec2,
    end_uv: Vec2,
    u_bulge: f64,
    v_bulge: f64,
) -> EdgeCurve3D {
    let middle_uv = (start_uv + end_uv) * 0.5;
    let u_term = if middle_uv.x > 0.0 && middle_uv.x < 1.0 {
        u_bulge * 0.5
    } else {
        0.0
    };
    let v_term = if middle_uv.y > 0.0 && middle_uv.y < 1.0 {
        v_bulge * 0.5
    } else {
        0.0
    };
    let middle_z = -1.0 + u_term + v_term;
    let control_points = vec![
        uv_to_xy_point(start_uv, -1.0),
        uv_to_xy_point(middle_uv, middle_z),
        uv_to_xy_point(end_uv, -1.0),
    ];
    EdgeCurve3D::Nurbs(Box::new(
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            control_points,
            vec![1.0; 3],
        )
        .unwrap(),
    ))
}

fn uv_to_xy_point(uv: Vec2, z: f64) -> Point3 {
    Point3::new(-1.0 + 2.0 * uv.x, -1.0 + 2.0 * uv.y, z)
}

fn vec2_close(a: Vec2, b: Vec2, tolerance: f64) -> bool {
    let delta = a - b;
    delta.dot(delta).sqrt() <= tolerance
}

fn vec3_close(a: Point3, b: Point3, tolerance: f64) -> bool {
    a.distance(b) <= tolerance
}
