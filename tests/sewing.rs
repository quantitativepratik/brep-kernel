use brep_kernel::math::Point3;
use brep_kernel::topology::{Solid, TopologyError};

#[test]
fn sewing_closes_shell_with_near_duplicate_vertex() {
    let points = near_open_tetrahedron_points();
    let triangles = near_open_tetrahedron_triangles();

    assert!(matches!(
        Solid::from_triangle_mesh(points.clone(), &triangles),
        Err(TopologyError::BoundaryEdge(_, _))
    ));

    let (solid, report) = Solid::from_triangle_mesh_sewn(points, &triangles, 1.0e-6).unwrap();

    assert_eq!(report.input_vertices, 5);
    assert_eq!(report.output_vertices, 4);
    assert_eq!(report.input_triangles, 4);
    assert_eq!(report.output_triangles, 4);
    assert_eq!(report.merged_vertices, 1);
    assert_eq!(report.removed_degenerate_triangles, 0);
    assert_eq!(report.vertex_map[0], report.vertex_map[4]);
    assert_eq!(solid.topology_counts().vertices, 4);
    assert_eq!(solid.topology_counts().faces, 4);
    solid.validate().unwrap();
}

#[test]
fn sewing_reports_triangles_collapsed_by_vertex_merge() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0e-8, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let triangles = [[0, 1, 2], [0, 2, 3]];

    let sewn = Solid::sew_triangle_mesh(points, &triangles, 1.0e-6).unwrap();

    assert_eq!(sewn.vertices.len(), 3);
    assert_eq!(sewn.triangles, vec![[0, 1, 2]]);
    assert_eq!(sewn.report.merged_vertices, 1);
    assert_eq!(sewn.report.removed_degenerate_triangles, 1);
    assert_eq!(sewn.report.vertex_map[0], sewn.report.vertex_map[1]);
}

#[test]
fn mesh_construction_rejects_invalid_vertex_indices_without_panicking() {
    let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
    let triangles = [[0, 1, 2]];

    assert_eq!(
        Solid::from_triangle_mesh(points.clone(), &triangles),
        Err(TopologyError::InvalidVertex(2))
    );
    assert_eq!(
        Solid::sew_triangle_mesh(points, &triangles, 1.0e-6),
        Err(TopologyError::InvalidVertex(2))
    );
}

#[test]
fn sewing_rejects_invalid_tolerance() {
    assert_eq!(
        Solid::sew_triangle_mesh(Vec::new(), &[], f64::NAN),
        Err(TopologyError::InvalidSewingTolerance)
    );
    assert_eq!(
        Solid::from_triangle_mesh_sewn(Vec::new(), &[], -1.0),
        Err(TopologyError::InvalidSewingTolerance)
    );
}

fn near_open_tetrahedron_points() -> Vec<Point3> {
    vec![
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0e-8, -1.0e-8, 1.0 + 1.0e-8),
    ]
}

fn near_open_tetrahedron_triangles() -> Vec<[usize; 3]> {
    vec![[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 4]]
}
