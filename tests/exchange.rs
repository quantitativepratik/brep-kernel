use brep_kernel::errors::{KernelError, KernelErrorKind, KernelSubsystem};
use brep_kernel::exchange::{
    export_iges_faceted_brep, export_step_faceted_brep, import_iges_faceted_brep,
    import_step_faceted_brep,
};
use brep_kernel::topology::{Solid, TopologyError};

#[test]
fn step_faceted_brep_round_trips_closed_solid() {
    let cube = Solid::cube(2.0).unwrap();

    let step = export_step_faceted_brep(&cube, "cube").unwrap();
    assert!(step.contains("FACETED_BREP"));
    assert!(step.contains("CARTESIAN_POINT"));
    assert!(step.contains("CLOSED_SHELL"));

    let imported = import_step_faceted_brep(&step).unwrap();
    assert_eq!(imported.topology_counts(), cube.topology_counts());
    assert_eq!(imported.stable_mesh_hash(), cube.stable_mesh_hash());
    assert_eq!(imported.volume(), cube.volume());
    imported.validate().unwrap();
}

#[test]
fn iges_faceted_subset_round_trips_closed_solid() {
    let cube = Solid::cube(2.0).unwrap();

    let iges = export_iges_faceted_brep(&cube, "cube").unwrap();
    assert!(iges.contains("BREP_KERNEL_IGES_FACETED_SUBSET"));
    assert!(iges.contains("116,"));
    assert!(iges.contains("106,"));

    let imported = import_iges_faceted_brep(&iges).unwrap();
    assert_eq!(imported.topology_counts(), cube.topology_counts());
    assert_eq!(imported.stable_mesh_hash(), cube.stable_mesh_hash());
    assert_eq!(imported.surface_area(), cube.surface_area());
    imported.validate().unwrap();
}

#[test]
fn exchange_parse_errors_are_structured() {
    let error = import_step_faceted_brep(
        "ISO-10303-21;\nDATA;\n#1=CARTESIAN_POINT('',(0,0));\nENDSEC;\nEND-ISO-10303-21;",
    )
    .unwrap_err();

    assert_eq!(error.primary.subsystem, KernelSubsystem::Exchange);
    assert_eq!(error.primary.kind, KernelErrorKind::Parse);
    assert_eq!(error.primary.code, "exchange.parse");
    assert_eq!(
        error.primary.operation.as_deref(),
        Some("import_step_faceted_brep")
    );
}

#[test]
fn subsystem_errors_convert_to_kernel_errors() {
    let error = KernelError::from(TopologyError::InvalidVertex(7));

    assert_eq!(error.primary.subsystem, KernelSubsystem::Topology);
    assert_eq!(error.primary.kind, KernelErrorKind::Topology);
    assert_eq!(error.primary.code, "topology.error");
    assert!(error.primary.source.unwrap().contains("InvalidVertex(7)"));
}
