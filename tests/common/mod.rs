use brep_kernel::math::Point3;
use brep_kernel::topology::Solid;

pub fn rectangular_box(size: [f64; 3]) -> Solid {
    let [sx, sy, sz] = size;
    let hx = sx * 0.5;
    let hy = sy * 0.5;
    let hz = sz * 0.5;
    let points = vec![
        Point3::new(-hx, -hy, -hz),
        Point3::new(hx, -hy, -hz),
        Point3::new(hx, hy, -hz),
        Point3::new(-hx, hy, -hz),
        Point3::new(-hx, -hy, hz),
        Point3::new(hx, -hy, hz),
        Point3::new(hx, hy, hz),
        Point3::new(-hx, hy, hz),
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
    Solid::from_triangle_mesh(points, &triangles).expect("rectangular box is a valid solid")
}

pub fn assert_closed_manifold(solid: &Solid) {
    let counts = solid.topology_counts();
    solid.validate().expect("solid validates");
    assert_eq!(counts.boundary_halfedges, 0);
    assert_eq!(counts.halfedges, counts.edges * 2);
    assert_eq!(counts.triangles, counts.faces);
    assert!(solid.volume().is_finite());
    assert!(solid.volume() >= 0.0);
    assert!(solid.surface_area().is_finite());
    assert!(solid.surface_area() > 0.0);
}
