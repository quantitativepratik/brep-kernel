use brep_kernel::boolean::{
    analyze_boolean_topology_merges, build_face_intersection_graph, BooleanEdgeMergeKind,
    BooleanFaceMergeKind, BooleanFacePairStatus,
};
use brep_kernel::math::{Point3, Vec3};
use brep_kernel::topology::Solid;

#[test]
fn disjoint_solids_still_emit_total_face_pair_matrix() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(5.0, 0.0, 0.0));

    let graph = build_face_intersection_graph(&left, &right, 1.0e-8).unwrap();

    assert_eq!(graph.left_face_count, left.faces.len());
    assert_eq!(graph.right_face_count, right.faces.len());
    assert_eq!(graph.face_pairs.len(), left.faces.len() * right.faces.len());
    assert_eq!(graph.active_pair_count, 0);
    assert_eq!(graph.curve_count, 0);
    assert!(graph.left_adjacency.iter().all(Vec::is_empty));
    assert!(graph.right_adjacency.iter().all(Vec::is_empty));
    assert!(graph
        .face_pairs
        .iter()
        .all(|pair| pair.status == BooleanFacePairStatus::Disjoint));
}

#[test]
fn overlapping_solids_emit_adjacency_and_trim_ready_segments() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(0.75, 0.0, 0.0));

    let graph = build_face_intersection_graph(&left, &right, 1.0e-8).unwrap();

    assert_eq!(graph.face_pairs.len(), left.faces.len() * right.faces.len());
    assert!(graph.active_pair_count > 0);
    assert!(graph.curve_count > 0);
    assert!(graph.left_adjacency.iter().any(|pairs| !pairs.is_empty()));
    assert!(graph.right_adjacency.iter().any(|pairs| !pairs.is_empty()));
    assert!(graph.face_pairs.iter().any(|pair| {
        pair.status == BooleanFacePairStatus::Intersecting && !pair.curves.is_empty()
    }));

    for pair in graph
        .face_pairs
        .iter()
        .filter(|pair| !pair.curves.is_empty())
    {
        for curve in &pair.curves {
            assert!(curve.points.len() >= 2);
            assert!(curve.left_pcurve.is_some());
            assert!(curve.right_pcurve.is_some());
            assert!(curve.max_residual <= 1.0e-6);
        }
    }
}

#[test]
fn touching_solids_record_coincident_face_pairs() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(2.0, 0.0, 0.0));

    let graph = build_face_intersection_graph(&left, &right, 1.0e-8).unwrap();

    assert!(graph.active_pair_count > 0);
    assert!(graph
        .face_pairs
        .iter()
        .any(|pair| pair.status == BooleanFacePairStatus::Coincident));
}

#[test]
fn merge_analysis_detects_coplanar_overlaps_and_shared_edges() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(2.0, 0.0, 0.0));

    let report = analyze_boolean_topology_merges(&left, &right, 1.0e-8).unwrap();

    assert!(report.merged_vertex_count >= 4);
    assert!(report.merged_edge_count > 0);
    assert!(report.merged_face_count > 0);
    assert!(report.coplanar_pair_count > 0);
    assert!(report.overlapping_pair_count > 0);
    assert!(report
        .edges
        .iter()
        .any(|edge| edge.kind == BooleanEdgeMergeKind::Coincident));
    assert!(report
        .faces
        .iter()
        .any(|face| face.kind == BooleanFaceMergeKind::CoplanarOverlap));
}

#[test]
fn merge_analysis_detects_nearly_coincident_topology() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(2.0 + 5.0e-8, 0.0, 0.0));

    let report = analyze_boolean_topology_merges(&left, &right, 1.0e-6).unwrap();

    assert!(report.merged_vertex_count >= 4);
    assert!(report.merged_edge_count > 0);
    assert!(report.merged_face_count > 0);
    assert!(report.nearly_coincident_pair_count > 0);
    assert!(report
        .edges
        .iter()
        .any(|edge| edge.kind == BooleanEdgeMergeKind::NearlyCoincident));
    assert!(report
        .faces
        .iter()
        .any(|face| face.kind == BooleanFaceMergeKind::NearlyCoincident));
}

#[test]
fn merge_analysis_detects_tangent_corner_contacts() {
    let left = translated_box([2.0, 2.0, 2.0], Vec3::ZERO);
    let right = translated_box([2.0, 2.0, 2.0], Vec3::new(2.0, 2.0, 2.0));

    let graph = build_face_intersection_graph(&left, &right, 1.0e-8).unwrap();
    let report = analyze_boolean_topology_merges(&left, &right, 1.0e-8).unwrap();

    assert!(graph
        .face_pairs
        .iter()
        .any(|pair| pair.status == BooleanFacePairStatus::Touching
            && !pair.contact_points.is_empty()));
    assert!(report.merged_vertex_count >= 1);
    assert!(report.tangent_pair_count > 0);
    assert!(report
        .faces
        .iter()
        .any(|face| face.kind == BooleanFaceMergeKind::TangentTouch));
}

fn translated_box(size: [f64; 3], offset: Vec3) -> Solid {
    let [sx, sy, sz] = size;
    let hx = sx * 0.5;
    let hy = sy * 0.5;
    let hz = sz * 0.5;
    let points = vec![
        Point3::new(-hx, -hy, -hz) + offset,
        Point3::new(hx, -hy, -hz) + offset,
        Point3::new(hx, hy, -hz) + offset,
        Point3::new(-hx, hy, -hz) + offset,
        Point3::new(-hx, -hy, hz) + offset,
        Point3::new(hx, -hy, hz) + offset,
        Point3::new(hx, hy, hz) + offset,
        Point3::new(-hx, hy, hz) + offset,
    ];
    let triangles = vec![
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
    ];
    Solid::from_triangle_mesh(points, &triangles).expect("box is a valid closed solid")
}
