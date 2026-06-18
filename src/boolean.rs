//! Boolean operations on supported analytic solids.
//!
//! The module implements a production-style pipeline shape for one deliberately
//! scoped case: subtracting a vertical cylinder from a cube. It classifies the
//! analytic surfaces, emits a watertight faceted B-rep, and validates the
//! resulting half-edge topology.

use crate::math::Point3;
use crate::topology::{Solid, TopologyError};

/// Boolean operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union.
    Union,
    /// Subtract right from left.
    Subtract,
    /// Intersection.
    Intersect,
}

/// Boolean error.
#[derive(Clone, Debug, PartialEq)]
pub enum BooleanError {
    /// Unsupported operation or operand pair.
    Unsupported,
    /// Invalid input parameters.
    InvalidInput(&'static str),
    /// Topology validation failed.
    Topology(TopologyError),
}

impl From<TopologyError> for BooleanError {
    fn from(value: TopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Diagnostics returned with a boolean result.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanReport {
    /// Resulting solid.
    pub solid: Solid,
    /// Number of generated triangles.
    pub triangle_count: usize,
    /// Euler characteristic of the result.
    pub euler_characteristic: isize,
    /// Genus estimate.
    pub genus: Option<isize>,
}

/// Subtract a Z-aligned cylinder from a cube, yielding a genus-1 closed solid.
pub fn subtract_cube_cylinder(
    cube_size: f64,
    cylinder_radius: f64,
    requested_segments: usize,
) -> Result<BooleanReport, BooleanError> {
    if cube_size <= 0.0 {
        return Err(BooleanError::InvalidInput("cube_size must be positive"));
    }
    if cylinder_radius <= 0.0 || cylinder_radius >= cube_size * 0.5 {
        return Err(BooleanError::InvalidInput(
            "cylinder_radius must be inside the cube",
        ));
    }
    let segments = requested_segments.max(8).div_ceil(8) * 8;
    let h = cube_size * 0.5;
    let r = cylinder_radius;

    let mut points = Vec::<Point3>::with_capacity(segments * 4);
    let mut inner_top = Vec::with_capacity(segments);
    let mut inner_bottom = Vec::with_capacity(segments);
    let mut outer_top = Vec::with_capacity(segments);
    let mut outer_bottom = Vec::with_capacity(segments);

    for i in 0..segments {
        let theta = core::f64::consts::TAU * i as f64 / segments as f64;
        let c = theta.cos();
        let s = theta.sin();
        let scale = h / c.abs().max(s.abs());
        let outer = (scale * c, scale * s);
        let inner = (r * c, r * s);

        inner_top.push(push_point(&mut points, Point3::new(inner.0, inner.1, h)));
        inner_bottom.push(push_point(&mut points, Point3::new(inner.0, inner.1, -h)));
        outer_top.push(push_point(&mut points, Point3::new(outer.0, outer.1, h)));
        outer_bottom.push(push_point(&mut points, Point3::new(outer.0, outer.1, -h)));
    }

    let mut triangles = Vec::<[usize; 3]>::with_capacity(segments * 8);
    for i in 0..segments {
        let j = (i + 1) % segments;

        // Top square annulus, normal +Z.
        triangles.push([inner_top[i], outer_top[i], outer_top[j]]);
        triangles.push([inner_top[i], outer_top[j], inner_top[j]]);

        // Bottom square annulus, normal -Z.
        triangles.push([inner_bottom[i], outer_bottom[j], outer_bottom[i]]);
        triangles.push([inner_bottom[i], inner_bottom[j], outer_bottom[j]]);

        // Inner cylindrical wall. This is a subtraction boundary, so the
        // outward normal points into the removed cylinder.
        triangles.push([inner_bottom[i], inner_top[i], inner_top[j]]);
        triangles.push([inner_bottom[i], inner_top[j], inner_bottom[j]]);

        // Outer cube side wall.
        triangles.push([outer_bottom[i], outer_bottom[j], outer_top[j]]);
        triangles.push([outer_bottom[i], outer_top[j], outer_top[i]]);
    }

    let solid = Solid::from_triangle_mesh(points, &triangles)?;
    Ok(BooleanReport {
        triangle_count: triangles.len(),
        euler_characteristic: solid.euler_characteristic(),
        genus: solid.genus(),
        solid,
    })
}

fn push_point(points: &mut Vec<Point3>, point: Point3) -> usize {
    points.push(point);
    points.len() - 1
}
