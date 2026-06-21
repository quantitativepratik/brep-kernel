use brep_kernel::api;
use brep_kernel::prelude::*;

#[test]
fn version_metadata_matches_package_and_abi_policy() {
    let version = api::version();

    assert_eq!(version.crate_name, env!("CARGO_PKG_NAME"));
    assert_eq!(version.crate_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(version.crate_version_major, env!("CARGO_PKG_VERSION_MAJOR"));
    assert_eq!(version.crate_version_minor, env!("CARGO_PKG_VERSION_MINOR"));
    assert_eq!(version.crate_version_patch, env!("CARGO_PKG_VERSION_PATCH"));
    assert_eq!(version.api_revision, api::API_REVISION);
    assert_eq!(version.wasm_abi_revision, api::WASM_ABI_REVISION);
    assert_eq!(version.msrv, api::MINIMUM_SUPPORTED_RUST_VERSION);
}

#[test]
fn prelude_supports_common_application_workflow() -> KernelResult<()> {
    let cube = Solid::cube(2.0)?;
    let step = export_step_faceted_brep(&cube, "public api cube")?;
    let imported = import_step_faceted_brep(&step)?;

    assert_eq!(imported.topology_counts(), cube.topology_counts());
    assert_eq!(imported.stable_mesh_hash(), cube.stable_mesh_hash());

    let feature_tree = parse_feature_prompt("60x24x10mm bracket with two M4 holes")?;
    let bracket = feature_tree.execute()?;
    bracket.validate()?;

    let surface = NurbsSurface::bilinear([
        [Point3::new(-1.0, -1.0, 0.0), Point3::new(-1.0, 1.0, 0.2)],
        [Point3::new(1.0, -1.0, 0.3), Point3::new(1.0, 1.0, 0.0)],
    ]);
    let tessellation = tessellate_nurbs_surface(&surface, 8, 8);
    assert_eq!(tessellation.indices.len(), 8 * 8 * 2);

    let boolean = subtract_cube_cylinder(4.0, 0.75, 32)?;
    assert_eq!(boolean.genus, Some(1));

    Ok(())
}
