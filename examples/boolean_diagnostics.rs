use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::errors::KernelError;

fn main() {
    let error = KernelError::from(
        subtract_cube_cylinder(10.0, 6.0, 32)
            .expect_err("radius larger than the cube half-size should fail"),
    )
    .with_operation("subtract_cube_cylinder");

    println!("code: {}", error.primary.code);
    println!("kind: {:?}", error.primary.kind);
    println!("message: {}", error.primary.message);
    if let Some(operation) = &error.primary.operation {
        println!("operation: {operation}");
    }
    for note in &error.primary.notes {
        println!("note: {note}");
    }
}
