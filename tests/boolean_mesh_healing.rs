use brep_kernel::boolean::heal_boolean_triangle_mesh;
use brep_kernel::math::Point3;
use brep_kernel::topology::TopologyError;

#[test]
fn boolean_healing_closes_gaps_and_preserves_manifold_validity() {
    let points = near_open_tetrahedron_points();
    let triangles = near_open_tetrahedron_triangles();

    let healed = heal_boolean_triangle_mesh(points, &triangles, 1.0e-6).unwrap();

    assert!(healed.report.manifold);
    assert_eq!(healed.report.merged_vertices, 1);
    assert_eq!(healed.report.output_vertices, 4);
    assert_eq!(healed.report.output_triangles, 4);
    let solid = healed.solid.expect("gap-closed tetrahedron validates");
    solid.validate().unwrap();
    assert_eq!(solid.euler_characteristic(), 2);
}

#[test]
fn boolean_healing_removes_slivers_before_manifold_validation() {
    let mut points = tetrahedron_points();
    points.extend([
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(110.0, 0.0, 0.0),
        Point3::new(10.0, 1.0e-5, 0.0),
    ]);
    let mut triangles = tetrahedron_triangles();
    triangles.push([4, 5, 6]);

    let healed = heal_boolean_triangle_mesh(points, &triangles, 1.0e-6).unwrap();

    assert_eq!(healed.report.input_triangles, 5);
    assert_eq!(healed.report.removed_sliver_triangles, 1);
    assert!(healed.report.manifold);
    assert_eq!(healed.report.output_vertices, 4);
    assert_eq!(healed.report.output_triangles, 4);
    healed.solid.unwrap().validate().unwrap();
}

#[test]
fn boolean_healing_reports_nonmanifold_inputs_without_panicking() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let triangles = [[0, 1, 2]];

    let healed = heal_boolean_triangle_mesh(points, &triangles, 1.0e-6).unwrap();

    assert!(!healed.report.manifold);
    assert!(healed.solid.is_none());
    assert!(matches!(
        healed.report.solid_error,
        Some(TopologyError::BoundaryEdge(_, _))
    ));
    assert_eq!(healed.report.output_triangles, 1);
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

fn tetrahedron_points() -> Vec<Point3> {
    vec![
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]
}

fn tetrahedron_triangles() -> Vec<[usize; 3]> {
    vec![[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
}
