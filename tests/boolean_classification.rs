use brep_kernel::boolean::{
    classify_split_faces, heal_classified_split_faces, BooleanError, BooleanOp, BooleanOperand,
    BooleanRegionAction, BooleanRegionSide, BooleanSplitStatus, HealedRegionSide, TrimDomainStatus,
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
