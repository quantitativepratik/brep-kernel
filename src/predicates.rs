//! Interval-filtered geometric predicates.
//!
//! The predicates here are deliberately conservative: when directed rounding
//! cannot certify a sign, they return [`RobustSign::Uncertain`] instead of
//! silently trusting a near-zero floating-point determinant.

use crate::math::{Vec2, Vec3};

/// Sign classification for robust predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustSign {
    /// Certified negative.
    Negative,
    /// Certified zero.
    Zero,
    /// Certified positive.
    Positive,
    /// The interval straddles zero.
    Uncertain,
}

impl RobustSign {
    /// True if the sign is known.
    pub fn is_certified(self) -> bool {
        !matches!(self, Self::Uncertain)
    }
}

/// Closed interval with outward-rounded arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

impl Interval {
    /// Exact point interval.
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// Interval sign classification.
    pub fn sign(self) -> RobustSign {
        if self.lo > 0.0 {
            RobustSign::Positive
        } else if self.hi < 0.0 {
            RobustSign::Negative
        } else if self.lo == 0.0 && self.hi == 0.0 {
            RobustSign::Zero
        } else {
            RobustSign::Uncertain
        }
    }

    /// Interval addition with outward rounding.
    pub fn add_i(self, rhs: Self) -> Self {
        Self {
            lo: next_down(self.lo + rhs.lo),
            hi: next_up(self.hi + rhs.hi),
        }
    }

    /// Interval subtraction with outward rounding.
    pub fn sub_i(self, rhs: Self) -> Self {
        Self {
            lo: next_down(self.lo - rhs.hi),
            hi: next_up(self.hi - rhs.lo),
        }
    }

    /// Interval multiplication with outward rounding.
    pub fn mul_i(self, rhs: Self) -> Self {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        let mut lo = products[0];
        let mut hi = products[0];
        for value in products.iter().copied().skip(1) {
            lo = lo.min(value);
            hi = hi.max(value);
        }
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }
}

/// Certified orientation of three 2D points.
pub fn orient2d(a: Vec2, b: Vec2, c: Vec2) -> RobustSign {
    let ax = Interval::point(a.x).sub_i(Interval::point(c.x));
    let ay = Interval::point(a.y).sub_i(Interval::point(c.y));
    let bx = Interval::point(b.x).sub_i(Interval::point(c.x));
    let by = Interval::point(b.y).sub_i(Interval::point(c.y));
    ax.mul_i(by).sub_i(ay.mul_i(bx)).sign()
}

/// Certified orientation of four 3D points.
pub fn orient3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> RobustSign {
    let adx = Interval::point(a.x).sub_i(Interval::point(d.x));
    let ady = Interval::point(a.y).sub_i(Interval::point(d.y));
    let adz = Interval::point(a.z).sub_i(Interval::point(d.z));
    let bdx = Interval::point(b.x).sub_i(Interval::point(d.x));
    let bdy = Interval::point(b.y).sub_i(Interval::point(d.y));
    let bdz = Interval::point(b.z).sub_i(Interval::point(d.z));
    let cdx = Interval::point(c.x).sub_i(Interval::point(d.x));
    let cdy = Interval::point(c.y).sub_i(Interval::point(d.y));
    let cdz = Interval::point(c.z).sub_i(Interval::point(d.z));

    let term1 = adx.mul_i(bdy.mul_i(cdz).sub_i(bdz.mul_i(cdy)));
    let term2 = ady.mul_i(bdx.mul_i(cdz).sub_i(bdz.mul_i(cdx)));
    let term3 = adz.mul_i(bdx.mul_i(cdy).sub_i(bdy.mul_i(cdx)));
    term1.sub_i(term2).add_i(term3).sign()
}

/// Certified incircle test for 2D points.
pub fn incircle2d(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> RobustSign {
    let adx = Interval::point(a.x).sub_i(Interval::point(d.x));
    let ady = Interval::point(a.y).sub_i(Interval::point(d.y));
    let bdx = Interval::point(b.x).sub_i(Interval::point(d.x));
    let bdy = Interval::point(b.y).sub_i(Interval::point(d.y));
    let cdx = Interval::point(c.x).sub_i(Interval::point(d.x));
    let cdy = Interval::point(c.y).sub_i(Interval::point(d.y));

    let abdet = adx.mul_i(bdy).sub_i(bdx.mul_i(ady));
    let bcdet = bdx.mul_i(cdy).sub_i(cdx.mul_i(bdy));
    let cadet = cdx.mul_i(ady).sub_i(adx.mul_i(cdy));

    let alift = adx.mul_i(adx).add_i(ady.mul_i(ady));
    let blift = bdx.mul_i(bdx).add_i(bdy.mul_i(bdy));
    let clift = cdx.mul_i(cdx).add_i(cdy.mul_i(cdy));

    alift
        .mul_i(bcdet)
        .add_i(blift.mul_i(cadet))
        .add_i(clift.mul_i(abdet))
        .sign()
}

/// Floating determinant for diagnostics only.
pub fn orient2d_fast(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}
