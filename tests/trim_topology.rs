use brep_kernel::math::{Point3, Vec2};
use brep_kernel::nurbs::NurbsSurface;
use brep_kernel::topology::{
    FaceSurface, Solid, TopologyError, Trim, TrimCurve2D, TrimLoop, TrimLoopKind,
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
