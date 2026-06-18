use brep_kernel::math::{Point3, Vec3};
use brep_kernel::nurbs::{NurbsCurve, NurbsSurface};
use brep_kernel::tessellation::tessellate_nurbs_surface;

#[test]
fn rational_quarter_circle_hits_midpoint() {
    let w = 0.5_f64.sqrt();
    let curve = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        vec![1.0, w, 1.0],
    )
    .unwrap();
    let p = curve.evaluate(0.5);
    assert!((p.x - w).abs() < 1.0e-12, "{p:?}");
    assert!((p.y - w).abs() < 1.0e-12, "{p:?}");
}

#[test]
fn knot_insertion_preserves_curve_shape() {
    let curve = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ],
        vec![1.0; 4],
    )
    .unwrap();
    let refined = curve.insert_knot_once(0.75).unwrap();
    for u in [0.1, 0.4, 0.75, 1.3, 1.9] {
        assert!(curve.evaluate(u).distance(refined.evaluate(u)) < 1.0e-10);
    }
}

#[test]
fn bilinear_surface_has_stable_normal_and_tessellation() {
    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, 0.0), Point3::new(1.0, -1.0, 0.0)],
        [Point3::new(-1.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
    ]);
    let normal = surface.normal(0.5, 0.5);
    assert!(normal.distance(Vec3::new(0.0, 0.0, 1.0)) < 1.0e-12);
    let mesh = tessellate_nurbs_surface(&surface, 8, 4);
    assert_eq!(mesh.vertices.len(), 45);
    assert_eq!(mesh.indices.len(), 64);
}
