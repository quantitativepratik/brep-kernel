//! Raw WASM exports for the browser viewer.
//!
//! The exports avoid `wasm-bindgen` so the crate can build without external
//! dependencies:
//!
//! - `brep_demo_mesh(segments)` fills an internal `f32` buffer with
//!   interleaved position/normal data for cube-minus-cylinder.
//! - `brep_buffer_ptr()` returns a pointer to that buffer.
//! - `brep_buffer_len()` returns the number of `f32` values.

use crate::boolean::subtract_cube_cylinder;
use crate::math::Vec3;
use std::sync::{Mutex, OnceLock};

static BUFFER: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();

/// ABI version for the browser.
#[no_mangle]
pub extern "C" fn brep_version() -> u32 {
    1
}

/// Generate the cube-minus-cylinder demo mesh.
#[no_mangle]
pub extern "C" fn brep_demo_mesh(segments: u32) -> usize {
    let Ok(report) = subtract_cube_cylinder(2.0, 0.45, segments as usize) else {
        return 0;
    };
    let mut out = buffer().lock().expect("buffer mutex poisoned");
    out.clear();
    let triangles = report.solid.triangles();
    for tri in triangles {
        let a = report.solid.vertices[tri[0]].point;
        let b = report.solid.vertices[tri[1]].point;
        let c = report.solid.vertices[tri[2]].point;
        let normal = (b - a).cross(c - a).normalized();
        push_vertex(&mut out, a, normal);
        push_vertex(&mut out, b, normal);
        push_vertex(&mut out, c, normal);
    }
    out.len()
}

/// Pointer to the generated mesh buffer.
#[no_mangle]
pub extern "C" fn brep_buffer_ptr() -> *const f32 {
    let out = buffer().lock().expect("buffer mutex poisoned");
    out.as_ptr()
}

/// Length of the generated mesh buffer in `f32` values.
#[no_mangle]
pub extern "C" fn brep_buffer_len() -> usize {
    let out = buffer().lock().expect("buffer mutex poisoned");
    out.len()
}

fn buffer() -> &'static Mutex<Vec<f32>> {
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_vertex(out: &mut Vec<f32>, point: Vec3, normal: Vec3) {
    out.extend_from_slice(&[
        point.x as f32,
        point.y as f32,
        point.z as f32,
        normal.x as f32,
        normal.y as f32,
        normal.z as f32,
    ]);
}
