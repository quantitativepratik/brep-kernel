use brep_kernel::math::{Point3, Vec2};
use brep_kernel::topology::{
    EdgeCurve3D, PersistentTopologyKind, Solid, TopologyError, TopologyOperation, TrimCurve2D,
};
use std::collections::HashSet;

#[test]
fn constructed_solids_have_unique_persistent_topological_ids() {
    let cube = Solid::cube(2.0).unwrap();

    let ids = cube.topology_identity().all_ids();
    let unique: HashSet<_> = ids.iter().copied().collect();

    assert_eq!(unique.len(), ids.len());
    assert_eq!(
        ids.len(),
        cube.vertices.len()
            + cube.halfedges.len()
            + cube.edges.len()
            + cube.split_edges.len()
            + cube.faces.len()
            + cube.shells.len()
    );
    assert_eq!(cube.topology_revision(), 1);
    assert_eq!(cube.topology_history().len(), 1);
    assert_eq!(
        cube.topology_history()[0].operation,
        TopologyOperation::ConstructTriangleMesh
    );
    assert_eq!(cube.topology_history()[0].created.len(), ids.len());
    assert_eq!(
        cube.persistent_face_id(0).unwrap().kind,
        PersistentTopologyKind::Face
    );
    cube.validate_persistent_identity().unwrap();
}

#[test]
fn face_metadata_edit_preserves_persistent_id_and_records_history() {
    let mut cube = Solid::cube(2.0).unwrap();
    let face_id = cube.persistent_face_id(0).unwrap();
    let revision = cube.topology_revision();
    let surface = cube.faces[0].surface.clone();

    cube.set_face_surface(0, surface).unwrap();

    assert_eq!(cube.persistent_face_id(0), Some(face_id));
    assert_eq!(cube.topology_revision(), revision + 1);
    let event = cube.topology_history().last().unwrap();
    assert_eq!(event.operation, TopologyOperation::SetFaceSurface);
    assert_eq!(event.modified, vec![face_id]);
    assert_eq!(event.parents, vec![face_id]);
    cube.validate().unwrap();
}

#[test]
fn failed_edge_edit_does_not_advance_history_or_change_identity() {
    let mut cube = Solid::cube(2.0).unwrap();
    let edge_id = cube.persistent_edge_id(0).unwrap();
    let old_edge = cube.edges[0].clone();
    let revision = cube.topology_revision();

    assert_eq!(
        cube.set_edge_curve(0, EdgeCurve3D::Unresolved, -1.0),
        Err(TopologyError::InvalidEdgeCurve(0))
    );

    assert_eq!(cube.persistent_edge_id(0), Some(edge_id));
    assert_eq!(cube.edges[0], old_edge);
    assert_eq!(cube.topology_revision(), revision);
    cube.validate().unwrap();
}

#[test]
fn staged_split_creates_persistent_split_edge_with_face_ancestry() {
    let mut cube = Solid::cube(2.0).unwrap();
    let a_face_id = cube.persistent_face_id(0).unwrap();
    let b_face_id = cube.persistent_face_id(1).unwrap();
    let revision = cube.topology_revision();

    let split = cube
        .split_faces_with_curves(
            0,
            1,
            EdgeCurve3D::line_segment(Point3::new(-0.25, 0.0, -1.0), Point3::new(0.25, 0.0, 1.0)),
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

    let split_id = cube.persistent_split_edge_id(split.split_edge).unwrap();
    assert_eq!(split_id.kind, PersistentTopologyKind::SplitEdge);
    assert_eq!(cube.topology_revision(), revision + 1);
    let event = cube.topology_history().last().unwrap();
    assert_eq!(event.operation, TopologyOperation::SplitFacesWithCurves);
    assert_eq!(event.created, vec![split_id]);
    assert_eq!(event.modified, vec![a_face_id, b_face_id]);
    assert_eq!(event.parents, vec![a_face_id, b_face_id]);
    cube.validate().unwrap();
}
