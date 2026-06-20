use brep_kernel::boolean::{
    classify_boolean_regions, classify_split_faces, heal_classified_split_faces, BooleanError,
    BooleanOp, BooleanOperand, BooleanRegionAction, BooleanRegionSide, BooleanSplitStatus,
    HealedRegionSide, TrimDomainStatus,
};
use brep_kernel::geometry::Plane;
use brep_kernel::math::{Point3, Vec2, Vec3};
use brep_kernel::topology::{
    EdgeCurve3D, FaceSurface, Solid, Trim, TrimCurve2D, TrimLoop, TrimLoopKind,
};

#[test]
fn union_classification_assigns_keep_and_discard_sides() {
    let solid = planar_split_fixture(false);
    let operands = operands_for_fixture(&solid);

    let report = classify_split_faces(&solid, &operands, BooleanOp::Union, 1.0e-9).unwrap();

    assert_eq!(report.split_count, 1);
    assert_eq!(report.active_split_count, 1);
    let split = &report.classifications[0];
    assert_eq!(split.status, BooleanSplitStatus::Active);
    assert_eq!(split.a.face, 0);
    assert_eq!(split.a.trim_domain, TrimDomainStatus::Inside);
    assert_eq!(split.a.left_of_curve, BooleanRegionSide::InsideOther);
    assert_eq!(split.a.right_of_curve, BooleanRegionSide::OutsideOther);
    assert_eq!(split.a.left_action, BooleanRegionAction::Discard);
    assert_eq!(split.a.right_action, BooleanRegionAction::Keep);
    assert_eq!(split.b.face, 1);
    assert_eq!(split.b.left_action, BooleanRegionAction::Keep);
    assert_eq!(split.b.right_action, BooleanRegionAction::Discard);
}

#[test]
fn subtract_classification_reverses_kept_regions_from_right_operand() {
    let solid = planar_split_fixture(false);
    let operands = operands_for_fixture(&solid);

    let report = classify_split_faces(&solid, &operands, BooleanOp::Subtract, 1.0e-9).unwrap();

    let split = &report.classifications[0];
    assert_eq!(split.status, BooleanSplitStatus::Active);
    assert_eq!(split.a.operand, BooleanOperand::Left);
    assert_eq!(split.a.left_action, BooleanRegionAction::Discard);
    assert_eq!(split.a.right_action, BooleanRegionAction::Keep);
    assert_eq!(split.b.operand, BooleanOperand::Right);
    assert_eq!(split.b.left_action, BooleanRegionAction::Discard);
    assert_eq!(split.b.right_action, BooleanRegionAction::KeepReversed);
}

#[test]
fn split_outside_trim_domain_is_marked_ambiguous() {
    let solid = planar_split_fixture(true);
    let operands = operands_for_fixture(&solid);

    let report = classify_split_faces(&solid, &operands, BooleanOp::Intersect, 1.0e-9).unwrap();

    assert_eq!(report.active_split_count, 0);
    let split = &report.classifications[0];
    assert_eq!(split.status, BooleanSplitStatus::Ambiguous);
    assert_eq!(split.a.trim_domain, TrimDomainStatus::Outside);
}

#[test]
fn healing_promotes_boundary_split_sides_to_trim_regions_and_mesh() {
    let solid = planar_boundary_split_fixture();
    let operands = operands_for_fixture(&solid);
    let classification = classify_split_faces(&solid, &operands, BooleanOp::Union, 1.0e-9).unwrap();

    let healed = heal_classified_split_faces(&solid, &classification, 1.0e-9).unwrap();

    assert_eq!(healed.operation, BooleanOp::Union);
    assert_eq!(healed.regions.len(), 2);
    assert_eq!(healed.regions[0].source_face, 0);
    assert_eq!(healed.regions[0].side, HealedRegionSide::RightOfSplit);
    assert_eq!(healed.regions[0].action, BooleanRegionAction::Keep);
    assert_eq!(healed.regions[0].uv_loop.len(), 4);
    assert_eq!(healed.regions[1].source_face, 1);
    assert_eq!(healed.regions[1].side, HealedRegionSide::LeftOfSplit);
    assert_eq!(healed.regions[1].action, BooleanRegionAction::Keep);
    assert_eq!(healed.regions[1].uv_loop.len(), 4);
    assert_eq!(healed.mesh.triangles.len(), 4);
    assert!(!healed.mesh.vertices.is_empty());
    assert!(healed.solid.is_none());
    assert!(healed.solid_error.is_some());
    assert!(healed.sewing_report.is_some());
}

#[test]
fn healing_rejects_split_that_does_not_reach_face_boundary() {
    let solid = planar_split_fixture(false);
    let operands = operands_for_fixture(&solid);
    let classification = classify_split_faces(&solid, &operands, BooleanOp::Union, 1.0e-9).unwrap();

    assert_eq!(
        heal_classified_split_faces(&solid, &classification, 1.0e-9),
        Err(BooleanError::Unsupported)
    );
}

#[test]
fn region_classification_partitions_faces_with_multiple_split_curves() {
    let solid = planar_multi_split_fixture();
    let operands = operands_for_fixture(&solid);

    let report = classify_boolean_regions(&solid, &operands, BooleanOp::Union, 1.0e-9).unwrap();

    assert_eq!(report.split_classification.active_split_count, 2);
    assert_eq!(report.affected_face_count, 2);
    let face0_regions: Vec<_> = report
        .regions
        .iter()
        .filter(|region| region.face == 0)
        .collect();
    assert_eq!(face0_regions.len(), 3);
    assert!(face0_regions.iter().all(|region| !region.unsplit_face));
    assert!(face0_regions
        .iter()
        .all(|region| region.source_splits == vec![0, 1]));
    assert!(face0_regions
        .iter()
        .all(|region| region.action != BooleanRegionAction::Ambiguous));
    assert!(face0_regions
        .iter()
        .any(|region| region.action == BooleanRegionAction::Keep));
    assert!(face0_regions
        .iter()
        .any(|region| region.action == BooleanRegionAction::Discard));
}

#[test]
fn region_classification_includes_unsplit_faces_against_closed_other_operand() {
    let (solid, operands) = nested_box_operands();

    let report = classify_boolean_regions(&solid, &operands, BooleanOp::Intersect, 1.0e-8).unwrap();

    assert_eq!(report.split_count, 0);
    assert_eq!(report.affected_face_count, 0);
    assert_eq!(report.region_count, solid.faces.len());
    assert_eq!(report.unsplit_region_count, solid.faces.len());
    assert_eq!(report.split_region_count, 0);
    assert_eq!(report.ambiguous_region_count, 0);
    assert_eq!(
        report
            .regions
            .iter()
            .filter(|region| region.operand == BooleanOperand::Left
                && region.action == BooleanRegionAction::Keep)
            .count(),
        12
    );
    assert_eq!(
        report
            .regions
            .iter()
            .filter(|region| region.operand == BooleanOperand::Right
                && region.action == BooleanRegionAction::Discard)
            .count(),
        12
    );
}

fn planar_split_fixture(outside_first_face: bool) -> Solid {
    let a_pcurve = if outside_first_face {
        TrimCurve2D::LineSegment {
            start: Vec2::new(2.0, -0.5),
            end: Vec2::new(2.0, 0.5),
        }
    } else {
        TrimCurve2D::LineSegment {
            start: Vec2::new(0.0, -0.5),
            end: Vec2::new(0.0, 0.5),
        }
    };
    planar_fixture_with_pcurves(
        a_pcurve,
        TrimCurve2D::LineSegment {
            start: Vec2::new(0.0, -0.5),
            end: Vec2::new(0.0, 0.5),
        },
    )
}

fn planar_boundary_split_fixture() -> Solid {
    planar_fixture_with_pcurves(
        TrimCurve2D::LineSegment {
            start: Vec2::new(0.0, -1.0),
            end: Vec2::new(0.0, 1.0),
        },
        TrimCurve2D::LineSegment {
            start: Vec2::new(0.0, -1.0),
            end: Vec2::new(0.0, 1.0),
        },
    )
}

fn planar_fixture_with_pcurves(a_pcurve: TrimCurve2D, b_pcurve: TrimCurve2D) -> Solid {
    let mut solid = Solid::cube(2.0).unwrap();
    solid
        .set_face_surface(
            0,
            FaceSurface::Plane(Plane::new(Point3::ZERO, Vec3::new(0.0, 0.0, 1.0))),
        )
        .unwrap();
    solid
        .set_face_surface(
            1,
            FaceSurface::Plane(Plane::new(Point3::ZERO, Vec3::new(0.0, 1.0, 0.0))),
        )
        .unwrap();
    solid
        .set_face_trim_loops(0, vec![square_loop(TrimLoopKind::Outer)])
        .unwrap();
    solid
        .set_face_trim_loops(1, vec![square_loop(TrimLoopKind::Outer)])
        .unwrap();
    solid
        .split_faces_with_curves(
            0,
            1,
            EdgeCurve3D::line_segment(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0)),
            a_pcurve,
            b_pcurve,
            1.0e-9,
        )
        .unwrap();
    solid
}

fn planar_multi_split_fixture() -> Solid {
    let mut solid = planar_fixture_with_pcurves(
        TrimCurve2D::LineSegment {
            start: Vec2::new(-0.35, -1.0),
            end: Vec2::new(-0.35, 1.0),
        },
        TrimCurve2D::LineSegment {
            start: Vec2::new(-0.35, -1.0),
            end: Vec2::new(-0.35, 1.0),
        },
    );
    solid
        .split_faces_with_curves(
            0,
            1,
            EdgeCurve3D::line_segment(Point3::new(-0.5, 0.35, 0.0), Point3::new(0.5, 0.35, 0.0)),
            TrimCurve2D::LineSegment {
                start: Vec2::new(0.35, -1.0),
                end: Vec2::new(0.35, 1.0),
            },
            TrimCurve2D::LineSegment {
                start: Vec2::new(0.35, -1.0),
                end: Vec2::new(0.35, 1.0),
            },
            1.0e-9,
        )
        .unwrap();
    solid
}

fn nested_box_operands() -> (Solid, Vec<BooleanOperand>) {
    let mut points = Vec::new();
    let mut triangles = Vec::new();
    append_box_mesh(&mut points, &mut triangles, [1.0, 1.0, 1.0], Vec3::ZERO);
    append_box_mesh(&mut points, &mut triangles, [3.0, 3.0, 3.0], Vec3::ZERO);
    let solid = Solid::from_triangle_mesh(points, &triangles).unwrap();
    let mut operands = vec![BooleanOperand::Left; 12];
    operands.extend([BooleanOperand::Right; 12]);
    (solid, operands)
}

fn append_box_mesh(
    points: &mut Vec<Point3>,
    triangles: &mut Vec<[usize; 3]>,
    size: [f64; 3],
    offset: Vec3,
) {
    let base = points.len();
    let [sx, sy, sz] = size;
    let hx = sx * 0.5;
    let hy = sy * 0.5;
    let hz = sz * 0.5;
    points.extend([
        Point3::new(-hx, -hy, -hz) + offset,
        Point3::new(hx, -hy, -hz) + offset,
        Point3::new(hx, hy, -hz) + offset,
        Point3::new(-hx, hy, -hz) + offset,
        Point3::new(-hx, -hy, hz) + offset,
        Point3::new(hx, -hy, hz) + offset,
        Point3::new(hx, hy, hz) + offset,
        Point3::new(-hx, hy, hz) + offset,
    ]);
    triangles.extend(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
        .map(|tri| [tri[0] + base, tri[1] + base, tri[2] + base]),
    );
}

fn operands_for_fixture(solid: &Solid) -> Vec<BooleanOperand> {
    let mut operands = vec![BooleanOperand::Left; solid.faces.len()];
    operands[1] = BooleanOperand::Right;
    operands
}

fn square_loop(kind: TrimLoopKind) -> TrimLoop {
    let points = [
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ];
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
