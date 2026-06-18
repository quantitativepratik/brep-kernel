//! Small vector types used by the kernel.

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// 2D vector or point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Vec2 {
    /// Construct a new 2D vector.
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Dot product.
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// 2D cross product magnitude.
    pub fn cross(self, rhs: Self) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }
}

/// 3D vector or point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

/// 3D point alias.
pub type Point3 = Vec3;

impl Vec3 {
    /// Zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Construct a new 3D vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Dot product.
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Squared Euclidean length.
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Return a normalized vector. Returns zero for a near-zero input.
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n <= f64::EPSILON {
            Self::ZERO
        } else {
            self / n
        }
    }

    /// Distance between two points.
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).norm()
    }

    /// Linear interpolation.
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self * (1.0 - t) + rhs * t
    }

    /// Convert to a float triplet.
    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// Homogeneous point used for rational NURBS arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    /// X coordinate multiplied by weight.
    pub x: f64,
    /// Y coordinate multiplied by weight.
    pub y: f64,
    /// Z coordinate multiplied by weight.
    pub z: f64,
    /// Rational weight.
    pub w: f64,
}

impl Vec4 {
    /// Construct a homogeneous point.
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Convert a Euclidean point plus weight to homogeneous form.
    pub fn from_point_weight(point: Point3, weight: f64) -> Self {
        Self::new(point.x * weight, point.y * weight, point.z * weight, weight)
    }

    /// Convert back to Euclidean coordinates.
    pub fn to_point(self) -> Point3 {
        Point3::new(self.x / self.w, self.y / self.w, self.z / self.w)
    }

    /// XYZ part.
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Add for Vec4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Div<f64> for Vec4 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

/// Approximate equality helper for tests and geometric tolerances.
pub fn nearly_equal(a: f64, b: f64, abs_tol: f64) -> bool {
    (a - b).abs() <= abs_tol
}
