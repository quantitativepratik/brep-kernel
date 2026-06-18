//! Analytic geometry primitives.

use crate::math::{Point3, Vec3};

/// Infinite 3D line represented by origin plus unit direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    /// Point on the line.
    pub origin: Point3,
    /// Unit direction.
    pub direction: Vec3,
}

impl Line {
    /// Construct a line. The direction is normalized.
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }

    /// Evaluate at parameter `t`.
    pub fn point_at(self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

/// Plane represented by origin plus unit normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Point on the plane.
    pub origin: Point3,
    /// Unit normal.
    pub normal: Vec3,
}

impl Plane {
    /// Construct a plane. The normal is normalized.
    pub fn new(origin: Point3, normal: Vec3) -> Self {
        Self {
            origin,
            normal: normal.normalized(),
        }
    }

    /// Build a plane through three non-collinear points.
    pub fn from_points(a: Point3, b: Point3, c: Point3) -> Option<Self> {
        let n = (b - a).cross(c - a);
        if n.norm() <= f64::EPSILON {
            None
        } else {
            Some(Self::new(a, n))
        }
    }

    /// Signed distance to the plane.
    pub fn signed_distance(self, point: Point3) -> f64 {
        self.normal.dot(point - self.origin)
    }

    /// Project a point onto the plane.
    pub fn project(self, point: Point3) -> Point3 {
        point - self.normal * self.signed_distance(point)
    }

    /// Plane constant `normal . x = constant`.
    pub fn constant(self) -> f64 {
        self.normal.dot(self.origin)
    }
}

/// Circle in a 3D plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    /// Center point.
    pub center: Point3,
    /// Unit normal of the circle plane.
    pub normal: Vec3,
    /// Radius.
    pub radius: f64,
}

impl Circle {
    /// Construct a circle.
    pub fn new(center: Point3, normal: Vec3, radius: f64) -> Self {
        Self {
            center,
            normal: normal.normalized(),
            radius,
        }
    }

    /// Sample a point. The local frame is deterministic for a given normal.
    pub fn point_at(self, theta: f64) -> Point3 {
        let helper = if self.normal.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = self.normal.cross(helper).normalized();
        let v = self.normal.cross(u).normalized();
        self.center + u * (theta.cos() * self.radius) + v * (theta.sin() * self.radius)
    }
}

/// Axis-aligned box, centered at `center`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisAlignedBox {
    /// Box center.
    pub center: Point3,
    /// Full side lengths.
    pub size: Vec3,
}

impl AxisAlignedBox {
    /// Construct a cube centered at the origin.
    pub fn cube(size: f64) -> Self {
        Self {
            center: Point3::ZERO,
            size: Vec3::new(size, size, size),
        }
    }

    /// Half extent along each axis.
    pub fn half_extents(self) -> Vec3 {
        self.size * 0.5
    }
}

/// Vertical cylinder used by the boolean demo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Center of the cylinder axis.
    pub center: Point3,
    /// Radius.
    pub radius: f64,
    /// Full height.
    pub height: f64,
}

impl Cylinder {
    /// Construct a cylinder aligned to the Z axis.
    pub fn z(center: Point3, radius: f64, height: f64) -> Self {
        Self {
            center,
            radius,
            height,
        }
    }
}
