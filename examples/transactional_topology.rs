use brep_kernel::errors::KernelError;
use brep_kernel::topology::{FaceTolerance, Solid, VertexTolerance};

fn main() -> Result<(), KernelError> {
    let mut cube = Solid::cube(2.0)?;
    let start_revision = cube.topology_revision();

    let report = {
        let mut transaction = cube.begin_topology_transaction();
        transaction.set_vertex_tolerance(0, VertexTolerance::new(1.0e-6))?;
        transaction.set_face_tolerance(0, FaceTolerance::new(2.0e-6, 3.0e-6, 4.0e-6))?;
        transaction.commit()?
    };

    println!("start revision: {}", start_revision);
    println!("end revision: {}", report.end_revision);
    println!("rollback entries recorded: {}", report.entries.len());
    println!("vertex[0] tolerance: {:?}", cube.vertex_tolerance(0));
    println!("face[0] tolerance: {:?}", cube.face_tolerance(0));

    let rollback_report = {
        let mut transaction = cube.begin_topology_transaction();
        transaction.set_vertex_tolerance(0, VertexTolerance::new(9.0e-6))?;
        transaction.rollback()
    };

    println!(
        "rollback restored revision: {} with {} undo entries",
        rollback_report.restored_revision,
        rollback_report.entries.len()
    );

    Ok(())
}
