//! CPU tessellation helpers mirrored by the WebGPU shader.

use crate::math::{Point3, Vec3};
use crate::nurbs::NurbsSurface;

/// Tessellated vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TessVertex {
    /// Position.
    pub position: Point3,
    /// Unit normal.
    pub normal: Vec3,
    /// U parameter.
    pub u: f64,
    /// V parameter.
    pub v: f64,
}

/// Triangle mesh generated from a surface.
#[derive(Clone, Debug, PartialEq)]
pub struct Tessellation {
    /// Vertices.
    pub vertices: Vec<TessVertex>,
    /// Triangle indices.
    pub indices: Vec<[u32; 3]>,
}

/// Tessellate a NURBS surface on a regular parameter grid.
pub fn tessellate_nurbs_surface(
    surface: &NurbsSurface,
    u_steps: usize,
    v_steps: usize,
) -> Tessellation {
    let u_steps = u_steps.max(1);
    let v_steps = v_steps.max(1);
    let ((u0, u1), (v0, v1)) = surface.domain();
    let mut vertices = Vec::with_capacity((u_steps + 1) * (v_steps + 1));
    for j in 0..=v_steps {
        for i in 0..=u_steps {
            let u = lerp(u0, u1, i as f64 / u_steps as f64);
            let v = lerp(v0, v1, j as f64 / v_steps as f64);
            vertices.push(TessVertex {
                position: surface.evaluate(u, v),
                normal: surface.normal(u, v),
                u,
                v,
            });
        }
    }

    let mut indices = Vec::with_capacity(u_steps * v_steps * 2);
    let row = u_steps + 1;
    for j in 0..v_steps {
        for i in 0..u_steps {
            let a = (j * row + i) as u32;
            let b = a + 1;
            let c = ((j + 1) * row + i) as u32;
            let d = c + 1;
            indices.push([a, b, d]);
            indices.push([a, d, c]);
        }
    }
    Tessellation { vertices, indices }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}
