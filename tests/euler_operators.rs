use brep_kernel::euler::{EulerBuilder, EulerError};
use brep_kernel::math::Point3;
use proptest::prelude::*;

#[test]
fn mvfs_mev_mef_preserve_euler_characteristic() {
    let mut builder = EulerBuilder::new();
    let (v0, face, shell) = builder.mvfs(Point3::new(0.0, 0.0, 0.0)).unwrap();
    assert_eq!(shell, 0);
    assert_eq!(builder.counts().euler_characteristic(), 2);
    assert!(builder.satisfies_euler_invariant());

    let v1 = builder.mev(face, v0, Point3::new(1.0, 0.0, 0.0)).unwrap();
    assert_eq!(builder.counts().vertices, 2);
    assert_eq!(builder.counts().edges, 1);
    assert_eq!(builder.counts().faces, 1);
    assert!(builder.satisfies_euler_invariant());

    let v2 = builder.mev(face, v1, Point3::new(0.0, 1.0, 0.0)).unwrap();
    assert_eq!(builder.counts().vertices, 3);
    assert_eq!(builder.counts().edges, 2);
    assert_eq!(builder.counts().faces, 1);
    assert!(builder.satisfies_euler_invariant());

    let closed_face = builder.mef(face, v2, v0).unwrap();
    assert_eq!(closed_face, 1);
    assert_eq!(builder.counts().vertices, 3);
    assert_eq!(builder.counts().edges, 3);
    assert_eq!(builder.counts().faces, 2);
    assert_eq!(builder.counts().euler_characteristic(), 2);
    assert!(builder.satisfies_euler_invariant());
}

#[test]
fn mef_splits_a_quad_face_into_two_triangular_loops() {
    let mut builder = EulerBuilder::new();
    let (v0, face, _) = builder.mvfs(Point3::new(-1.0, -1.0, 0.0)).unwrap();
    let v1 = builder.mev(face, v0, Point3::new(1.0, -1.0, 0.0)).unwrap();
    let v2 = builder.mev(face, v1, Point3::new(1.0, 1.0, 0.0)).unwrap();
    let v3 = builder.mev(face, v2, Point3::new(-1.0, 1.0, 0.0)).unwrap();

    let split_face = builder.mef(face, v0, v2).unwrap();
    assert_eq!(builder.face_loop(face).unwrap(), &[v2, v3, v0]);
    assert_eq!(builder.face_loop(split_face).unwrap(), &[v0, v1, v2]);
    assert_eq!(builder.counts().vertices, 4);
    assert_eq!(builder.counts().edges, 4);
    assert_eq!(builder.counts().faces, 2);
    assert!(builder.satisfies_euler_invariant());
}

#[test]
fn closed_triangle_sheet_converts_to_valid_halfedge_topology() {
    let mut builder = EulerBuilder::new();
    let (v0, face, _) = builder.mvfs(Point3::new(0.0, 0.0, 0.0)).unwrap();
    let v1 = builder.mev(face, v0, Point3::new(1.0, 0.0, 0.0)).unwrap();
    let v2 = builder.mev(face, v1, Point3::new(0.0, 1.0, 0.0)).unwrap();
    builder.mef(face, v2, v0).unwrap();

    let solid = builder.to_solid().unwrap();
    solid.validate().unwrap();
    assert_eq!(solid.boundary_halfedge_count(), 0);
    assert_eq!(solid.euler_characteristic(), 2);
    assert_eq!(solid.genus(), Some(0));
    assert!(solid.volume() >= 0.0);
}

#[test]
fn open_wire_cannot_convert_to_closed_solid() {
    let mut builder = EulerBuilder::new();
    let (v0, face, _) = builder.mvfs(Point3::new(0.0, 0.0, 0.0)).unwrap();
    builder.mev(face, v0, Point3::new(1.0, 0.0, 0.0)).unwrap();
    assert!(matches!(
        builder.to_solid(),
        Err(EulerError::OpenConstructionFace(0))
    ));
}

proptest! {
    #[test]
    fn random_triangle_sheet_preserves_operator_and_halfedge_invariants(
        ax in -10.0_f64..10.0,
        ay in -10.0_f64..10.0,
        bx in -10.0_f64..10.0,
        by in 0.1_f64..10.0,
    ) {
        let mut builder = EulerBuilder::new();
        let (v0, face, _) = builder.mvfs(Point3::new(0.0, 0.0, 0.0)).unwrap();
        let v1 = builder.mev(face, v0, Point3::new(ax, ay, 0.0)).unwrap();
        let v2 = builder.mev(face, v1, Point3::new(bx, by, 0.0)).unwrap();
        builder.mef(face, v2, v0).unwrap();

        prop_assert!(builder.satisfies_euler_invariant());
        prop_assert_eq!(builder.counts().euler_characteristic(), 2);

        let solid = builder.to_solid().unwrap();
        prop_assert_eq!(solid.boundary_halfedge_count(), 0);
        prop_assert_eq!(solid.euler_characteristic(), 2);
        prop_assert!(solid.volume() >= 0.0);
    }
}
