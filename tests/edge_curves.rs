use brep_kernel::math::{Point3, Vec2};
use brep_kernel::topology::{EdgeCurve3D, Solid, TopologyError, TrimCurve2D};

#[test]
fn triangle_mesh_edges_have_model_space_line_curves() {
    let cube = Solid::cube(2.0).unwrap();
    cube.validate_edge_curves().unwrap();

    for edge_id in 0..cube.edges.len() {
        let (origin, destination) = cube.edge_points(edge_id).unwrap();
        let (curve_start, curve_end) = cube.edges[edge_id].curve.endpoints().unwrap();
        let forward =
            curve_start.distance(origin) < 1.0e-12 && curve_end.distance(destination) < 1.0e-12;
        let reverse =
            curve_start.distance(destination) < 1.0e-12 && curve_end.distance(origin) < 1.0e-12;
        assert!(forward || reverse);
    }
}

#[test]
fn edge_curve_can_be_replaced_with_matching_3d_curve() {
    let mut cube = Solid::cube(2.0).unwrap();
    let edge = 0;
    let (start, end) = cube.edge_points(edge).unwrap();
    let mid = (start + end) * 0.5 + Point3::new(0.0, 0.0, 0.05);
    let curve = EdgeCurve3D::Polyline {
        points: vec![start, mid, end],
    };

    cube.set_edge_curve(edge, curve.clone(), 1.0e-6).unwrap();

    assert_eq!(cube.edges[edge].curve, curve);
    cube.validate_edge_curves().unwrap();
}

#[test]
fn edge_curve_endpoint_mismatch_is_rejected_and_restored() {
    let mut cube = Solid::cube(2.0).unwrap();
    let edge = 0;
    let original = cube.edges[edge].curve.clone();
    let bad = EdgeCurve3D::line_segment(Point3::new(99.0, 0.0, 0.0), Point3::new(100.0, 0.0, 0.0));

    assert_eq!(
        cube.set_edge_curve(edge, bad, 1.0e-9),
        Err(TopologyError::EdgeCurveEndpointMismatch(edge))
    );
    assert_eq!(cube.edges[edge].curve, original);
}

#[test]
fn adjacent_faces_store_distinct_pcurves_for_the_same_edge() {
    let mut cube = Solid::cube(2.0).unwrap();
    let halfedge = cube.edges[0].halfedge;
    let twin = cube.halfedges[halfedge].twin.unwrap();
    let face_a = cube.halfedges[halfedge].face;
    let face_b = cube.halfedges[twin].face;

    let curve_a = curved_pcurve_from_existing(&cube, face_a, halfedge, Vec2::new(0.12, 0.03));
    let curve_b = curved_pcurve_from_existing(&cube, face_b, twin, Vec2::new(-0.07, 0.11));

    cube.set_trim_curve(face_a, halfedge, curve_a.clone(), 1.0e-6)
        .unwrap();
    cube.set_trim_curve(face_b, twin, curve_b.clone(), 1.0e-6)
        .unwrap();

    assert_eq!(
        cube.trim_curve_for_halfedge(face_a, halfedge),
        Some(&curve_a)
    );
    assert_eq!(cube.trim_curve_for_halfedge(face_b, twin), Some(&curve_b));
    assert_ne!(curve_a, curve_b);
    cube.validate_trim_topology().unwrap();
}

fn curved_pcurve_from_existing(
    solid: &Solid,
    face: usize,
    halfedge: usize,
    offset: Vec2,
) -> TrimCurve2D {
    let (start, end) = solid
        .trim_curve_for_halfedge(face, halfedge)
        .and_then(TrimCurve2D::endpoints)
        .expect("mesh faces have projected p-curves");
    TrimCurve2D::Polyline {
        points: vec![start, (start + end) * 0.5 + offset, end],
    }
}
