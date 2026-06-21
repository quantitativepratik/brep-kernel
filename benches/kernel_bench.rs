use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::exchange::{export_step_faceted_brep, import_step_faceted_brep};
use brep_kernel::features::parse_feature_prompt;
use brep_kernel::nurbs::NurbsSurface;
use brep_kernel::tessellation::tessellate_nurbs_surface;
use brep_kernel::topology::Solid;
use brep_kernel::Point3;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    measure("cube validate", 2_000, || {
        let cube = Solid::cube(2.0).unwrap();
        cube.validate().unwrap();
        cube.stable_mesh_hash()
    });

    measure("cube minus cylinder", 200, || {
        let report = subtract_cube_cylinder(10.0, 2.0, 64).unwrap();
        report.solid.stable_mesh_hash() ^ report.triangle_count as u64
    });

    measure("STEP faceted roundtrip", 300, || {
        let cube = Solid::cube(2.0).unwrap();
        let step = export_step_faceted_brep(&cube, "bench cube").unwrap();
        let imported = import_step_faceted_brep(&step).unwrap();
        imported.stable_mesh_hash() ^ step.len() as u64
    });

    measure("feature prompt execute", 200, || {
        let tree = parse_feature_prompt("60x24x10mm bracket with two M4 holes").unwrap();
        let solid = tree.execute().unwrap();
        solid.stable_mesh_hash()
    });

    measure("NURBS tessellation 64x64", 100, || {
        let surface = NurbsSurface::bilinear([
            [Point3::new(-1.0, -1.0, 0.0), Point3::new(-1.0, 1.0, 0.2)],
            [Point3::new(1.0, -1.0, 0.4), Point3::new(1.0, 1.0, 0.0)],
        ]);
        let mesh = tessellate_nurbs_surface(&surface, 64, 64);
        mesh.vertices.len() as u64 ^ ((mesh.indices.len() as u64) << 32)
    });
}

fn measure(label: &str, iterations: usize, mut workload: impl FnMut() -> u64) {
    let start = Instant::now();
    let mut guard = 0u64;
    for index in 0..iterations {
        let value = black_box(workload());
        guard = guard.rotate_left(7).wrapping_mul(0x9e37_79b1_85eb_ca87)
            ^ value.wrapping_add(index as u64);
    }
    let elapsed = start.elapsed();
    let average_ms = elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    println!(
        "{label:<28} iterations={iterations:<6} total={:>8.3}ms avg={average_ms:>8.4}ms guard={guard:016x}",
        elapsed.as_secs_f64() * 1_000.0
    );
}
