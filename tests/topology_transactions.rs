use brep_kernel::math::{Point3, Vec2};
use brep_kernel::topology::{
    CoedgeTolerance, EdgeCurve3D, EdgeTolerance, FaceTolerance, Solid, TopologyError,
    TopologyOperation, TrimCurve2D, VertexTolerance,
};

#[test]
fn tolerance_model_is_aligned_with_topology_arrays() {
    let mut cube = Solid::cube(2.0).unwrap();

    assert_eq!(cube.tolerance_model().vertices.len(), cube.vertices.len());
    assert_eq!(cube.tolerance_model().coedges.len(), cube.halfedges.len());
    assert_eq!(cube.tolerance_model().edges.len(), cube.edges.len());
    assert_eq!(cube.tolerance_model().faces.len(), cube.faces.len());
    cube.validate_tolerance_model().unwrap();

    cube.tolerances.faces.pop();
    assert_eq!(
        cube.validate_tolerance_model(),
        Err(TopologyError::InvalidToleranceModel)
    );
}

#[test]
fn transaction_commit_keeps_edits_and_rollback_log_entries() {
    let mut cube = Solid::cube(2.0).unwrap();
    let start_revision = cube.topology_revision();
    let vertex_id = cube.persistent_vertex_id(0).unwrap();
    let face_id = cube.persistent_face_id(0).unwrap();

    let report = {
        let mut transaction = cube.begin_topology_transaction();
        transaction
            .set_vertex_tolerance(0, VertexTolerance::new(1.0e-6))
            .unwrap();
        transaction
            .set_face_tolerance(0, FaceTolerance::new(2.0e-6, 3.0e-6, 4.0e-6))
            .unwrap();
        assert_eq!(transaction.rollback_entries().len(), 2);
        transaction.commit().unwrap()
    };

    assert_eq!(report.start_revision, start_revision);
    assert_eq!(report.end_revision, start_revision + 2);
    assert_eq!(report.entries.len(), 2);
    assert_eq!(
        report.entries[0].operation,
        TopologyOperation::SetVertexTolerance
    );
    assert_eq!(report.entries[0].modified, vec![vertex_id]);
    assert_eq!(
        report.entries[1].operation,
        TopologyOperation::SetFaceTolerance
    );
    assert_eq!(report.entries[1].modified, vec![face_id]);
    assert_eq!(cube.vertex_tolerance(0), Some(VertexTolerance::new(1.0e-6)));
    assert_eq!(
        cube.face_tolerance(0),
        Some(FaceTolerance::new(2.0e-6, 3.0e-6, 4.0e-6))
    );
    cube.validate().unwrap();
}

#[test]
fn transaction_rollback_restores_original_solid_and_reports_reverse_undo_log() {
    let original = Solid::cube(2.0).unwrap();
    let mut cube = original.clone();
    let start_revision = cube.topology_revision();

    let report = {
        let mut transaction = cube.begin_topology_transaction();
        transaction
            .set_coedge_tolerance(0, CoedgeTolerance::new(1.0e-6, 2.0e-6))
            .unwrap();
        transaction
            .set_edge_tolerance(0, EdgeTolerance::new(3.0e-6, 4.0e-6))
            .unwrap();
        transaction.rollback()
    };

    assert_eq!(report.restored_revision, start_revision);
    assert_eq!(report.entries.len(), 2);
    assert_eq!(
        report.entries[0].operation,
        TopologyOperation::SetEdgeTolerance
    );
    assert_eq!(
        report.entries[1].operation,
        TopologyOperation::SetCoedgeTolerance
    );
    assert_eq!(cube, original);
}

#[test]
fn dropped_transaction_rolls_back_without_explicit_call() {
    let original = Solid::cube(2.0).unwrap();
    let mut cube = original.clone();

    {
        let mut transaction = cube.begin_topology_transaction();
        transaction
            .set_vertex_tolerance(0, VertexTolerance::new(9.0e-6))
            .unwrap();
        assert_eq!(
            transaction.solid().vertex_tolerance(0),
            Some(VertexTolerance::new(9.0e-6))
        );
    }

    assert_eq!(cube, original);
}

#[test]
fn transaction_rollback_restores_created_split_edges() {
    let original = Solid::cube(2.0).unwrap();
    let mut cube = original.clone();

    let report = {
        let mut transaction = cube.begin_topology_transaction();
        transaction
            .split_faces_with_curves(
                0,
                1,
                EdgeCurve3D::line_segment(
                    Point3::new(-0.25, 0.0, -1.0),
                    Point3::new(0.25, 0.0, 1.0),
                ),
                TrimCurve2D::LineSegment {
                    start: Vec2::new(0.0, 0.0),
                    end: Vec2::new(0.5, 0.5),
                },
                TrimCurve2D::LineSegment {
                    start: Vec2::new(0.25, 0.0),
                    end: Vec2::new(0.75, 0.5),
                },
                1.0e-8,
            )
            .unwrap();
        assert_eq!(transaction.solid().split_edges.len(), 1);
        transaction.rollback()
    };

    assert_eq!(report.entries.len(), 1);
    assert_eq!(
        report.entries[0].operation,
        TopologyOperation::SplitFacesWithCurves
    );
    assert_eq!(report.entries[0].created.len(), 1);
    assert_eq!(cube, original);
}
