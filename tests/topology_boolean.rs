use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::topology::Solid;

#[test]
fn cube_is_closed_genus_zero() {
    let cube = Solid::cube(2.0).unwrap();
    assert_eq!(cube.boundary_halfedge_count(), 0);
    assert_eq!(cube.euler_characteristic(), 2);
    assert_eq!(cube.genus(), Some(0));
    assert!(cube.signed_volume() > 0.0);
}

#[test]
fn cube_minus_cylinder_is_closed_genus_one() {
    let report = subtract_cube_cylinder(2.0, 0.45, 64).unwrap();
    assert_eq!(report.solid.boundary_halfedge_count(), 0);
    assert_eq!(report.euler_characteristic, 0);
    assert_eq!(report.genus, Some(1));
    assert_eq!(report.triangle_count, 64 * 8);
    assert!(report.solid.signed_volume() > 0.0);
}
