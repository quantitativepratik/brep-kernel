use brep_kernel::errors::KernelError;
use brep_kernel::exchange::{
    export_iges_faceted_brep, export_step_faceted_brep, import_iges_faceted_brep,
    import_step_faceted_brep,
};
use brep_kernel::topology::Solid;

fn main() -> Result<(), KernelError> {
    let cube = Solid::cube(2.0)?;
    let step = export_step_faceted_brep(&cube, "example cube")?;
    let iges = export_iges_faceted_brep(&cube, "example cube")?;
    let from_step = import_step_faceted_brep(&step)?;
    let from_iges = import_iges_faceted_brep(&iges)?;

    println!("cube topology: {:?}", cube.topology_counts());
    println!("cube volume: {:.6}", cube.volume());
    println!("cube area: {:.6}", cube.surface_area());
    println!("original hash: {:016x}", cube.stable_mesh_hash());
    println!("STEP hash:     {:016x}", from_step.stable_mesh_hash());
    println!("IGES hash:     {:016x}", from_iges.stable_mesh_hash());
    println!("STEP bytes: {}", step.len());
    println!("IGES bytes: {}", iges.len());

    Ok(())
}
