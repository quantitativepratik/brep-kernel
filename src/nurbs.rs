//! NURBS curves and tensor-product surfaces.

use crate::math::{Point3, Vec3, Vec4};

/// Knot vector with polynomial degree.
#[derive(Clone, Debug, PartialEq)]
pub struct KnotVector {
    /// Polynomial degree.
    pub degree: usize,
    /// Nondecreasing knot values.
    pub knots: Vec<f64>,
}

impl KnotVector {
    /// Construct and validate a knot vector.
    pub fn new(degree: usize, knots: Vec<f64>) -> Result<Self, NurbsError> {
        if knots.len() < degree + 2 {
            return Err(NurbsError::InvalidKnotVector);
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return Err(NurbsError::InvalidKnotVector);
        }
        Ok(Self { degree, knots })
    }

    /// Parametric domain.
    pub fn domain(&self) -> (f64, f64) {
        let n = self.knots.len() - self.degree - 2;
        (self.knots[self.degree], self.knots[n + 1])
    }

    /// Find the active span for `u`.
    pub fn find_span(&self, control_point_count: usize, u: f64) -> usize {
        let n = control_point_count - 1;
        if u >= self.knots[n + 1] {
            return n;
        }
        if u <= self.knots[self.degree] {
            return self.degree;
        }
        let mut low = self.degree;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while u < self.knots[mid] || u >= self.knots[mid + 1] {
            if u < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    /// Multiplicity of a knot value under a tolerance.
    pub fn multiplicity(&self, u: f64, tol: f64) -> usize {
        self.knots
            .iter()
            .filter(|knot| (**knot - u).abs() <= tol)
            .count()
    }
}

/// NURBS error.
#[derive(Clone, Debug, PartialEq)]
pub enum NurbsError {
    /// Knot vector is malformed.
    InvalidKnotVector,
    /// Counts do not satisfy NURBS invariants.
    InvalidControlNet,
    /// Requested operation is outside the mathematical domain.
    InvalidParameter,
}

/// Rational B-spline curve.
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsCurve {
    /// Knot vector.
    pub knots: KnotVector,
    /// Control points.
    pub control_points: Vec<Point3>,
    /// Rational weights.
    pub weights: Vec<f64>,
}

impl NurbsCurve {
    /// Construct a curve.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<Point3>,
        weights: Vec<f64>,
    ) -> Result<Self, NurbsError> {
        if control_points.len() != weights.len() || control_points.len() < degree + 1 {
            return Err(NurbsError::InvalidControlNet);
        }
        let knot_vector = KnotVector::new(degree, knots)?;
        if knot_vector.knots.len() != control_points.len() + degree + 1 {
            return Err(NurbsError::InvalidKnotVector);
        }
        Ok(Self {
            knots: knot_vector,
            control_points,
            weights,
        })
    }

    /// Interpolate points with a non-rational open B-spline curve.
    ///
    /// This uses chord-length parameters and the standard averaging knot
    /// vector for global interpolation. The returned curve has unit weights,
    /// degree `degree.min(points.len() - 1)`, and passes through every input
    /// point up to the linear solve tolerance.
    pub fn interpolate(
        points: &[Point3],
        degree: usize,
        tolerance: f64,
    ) -> Result<Self, NurbsError> {
        if points.len() < 2
            || degree == 0
            || !tolerance.is_finite()
            || tolerance < 0.0
            || points.iter().any(|point| !finite_point(*point))
        {
            return Err(NurbsError::InvalidParameter);
        }
        let degree = degree.min(points.len() - 1);
        let parameters = chord_length_parameters(points, tolerance)?;
        let knots = interpolation_knots(&parameters, degree);
        let matrix = interpolation_matrix(&parameters, degree, &knots);
        let control_points = solve_point_system(matrix, points, tolerance)?;
        let weights = vec![1.0; control_points.len()];
        Self::new(degree, knots, control_points, weights)
    }

    /// Parametric domain.
    pub fn domain(&self) -> (f64, f64) {
        self.knots.domain()
    }

    /// Evaluate the curve by rational de Boor/basis evaluation.
    pub fn evaluate(&self, u: f64) -> Point3 {
        self.evaluate_homogeneous(u).to_point()
    }

    /// First derivative with respect to `u`.
    pub fn derivative(&self, u: f64) -> Vec3 {
        let p = self.knots.degree;
        let span = self.knots.find_span(self.control_points.len(), u);
        let ders = ders_basis_funs(span, u, p, 1, &self.knots.knots);
        let mut c0 = Vec4::default();
        let mut c1 = Vec4::default();
        for (j, (&b0, &b1)) in ders[0].iter().zip(ders[1].iter()).enumerate().take(p + 1) {
            let idx = span - p + j;
            let pw = Vec4::from_point_weight(self.control_points[idx], self.weights[idx]);
            c0 = c0 + pw * b0;
            c1 = c1 + pw * b1;
        }
        (c1.xyz() * c0.w - c0.xyz() * c1.w) / (c0.w * c0.w)
    }

    /// Insert one knot and return a geometrically equivalent curve.
    pub fn insert_knot_once(&self, u: f64) -> Result<Self, NurbsError> {
        let p = self.knots.degree;
        let n = self.control_points.len() - 1;
        let k = self.knots.find_span(self.control_points.len(), u);
        let s = self.knots.multiplicity(u, 1.0e-12);
        if s >= p {
            return Err(NurbsError::InvalidParameter);
        }

        let old: Vec<Vec4> = self
            .control_points
            .iter()
            .zip(self.weights.iter())
            .map(|(point, weight)| Vec4::from_point_weight(*point, *weight))
            .collect();
        let mut new_pw = vec![Vec4::default(); n + 2];

        let left_count = k - p + 1;
        new_pw[..left_count].copy_from_slice(&old[..left_count]);
        let right_start = k - s;
        let right_len = n - right_start + 1;
        new_pw[right_start + 1..right_start + 1 + right_len]
            .copy_from_slice(&old[right_start..right_start + right_len]);
        for i in (k - p + 1)..=(k - s) {
            let denom = self.knots.knots[i + p] - self.knots.knots[i];
            let alpha = if denom.abs() <= f64::EPSILON {
                0.0
            } else {
                (u - self.knots.knots[i]) / denom
            };
            new_pw[i] = old[i - 1] * (1.0 - alpha) + old[i] * alpha;
        }

        let mut new_knots = Vec::with_capacity(self.knots.knots.len() + 1);
        new_knots.extend_from_slice(&self.knots.knots[..=k]);
        new_knots.push(u);
        new_knots.extend_from_slice(&self.knots.knots[k + 1..]);

        let mut control_points = Vec::with_capacity(new_pw.len());
        let mut weights = Vec::with_capacity(new_pw.len());
        for pw in new_pw {
            weights.push(pw.w);
            control_points.push(pw.to_point());
        }
        Self::new(p, new_knots, control_points, weights)
    }

    fn evaluate_homogeneous(&self, u: f64) -> Vec4 {
        let p = self.knots.degree;
        let span = self.knots.find_span(self.control_points.len(), u);
        let basis = basis_funs(span, u, p, &self.knots.knots);
        let mut c = Vec4::default();
        for (j, basis_value) in basis.iter().copied().enumerate() {
            let idx = span - p + j;
            let pw = Vec4::from_point_weight(self.control_points[idx], self.weights[idx]);
            c = c + pw * basis_value;
        }
        c
    }
}

fn chord_length_parameters(points: &[Point3], tolerance: f64) -> Result<Vec<f64>, NurbsError> {
    let mut parameters = Vec::with_capacity(points.len());
    parameters.push(0.0);
    let mut total = 0.0;
    for edge in points.windows(2) {
        let length = edge[0].distance(edge[1]);
        if length <= tolerance {
            return Err(NurbsError::InvalidParameter);
        }
        total += length;
        parameters.push(total);
    }
    if total <= tolerance {
        return Err(NurbsError::InvalidParameter);
    }
    for parameter in &mut parameters {
        *parameter /= total;
    }
    if let Some(last) = parameters.last_mut() {
        *last = 1.0;
    }
    Ok(parameters)
}

fn interpolation_knots(parameters: &[f64], degree: usize) -> Vec<f64> {
    let point_count = parameters.len();
    let n = point_count - 1;
    let mut knots = vec![0.0; point_count + degree + 1];
    for knot in knots.iter_mut().take(point_count + degree + 1).skip(n + 1) {
        *knot = 1.0;
    }
    if n > degree {
        for j in 1..=(n - degree) {
            let sum: f64 = parameters.iter().skip(j).take(degree).sum();
            knots[j + degree] = sum / degree as f64;
        }
    }
    knots
}

fn interpolation_matrix(parameters: &[f64], degree: usize, knots: &[f64]) -> Vec<Vec<f64>> {
    let point_count = parameters.len();
    let knot_vector = KnotVector {
        degree,
        knots: knots.to_vec(),
    };
    let mut matrix = vec![vec![0.0; point_count]; point_count];
    for (row, parameter) in parameters.iter().copied().enumerate() {
        let span = knot_vector.find_span(point_count, parameter);
        let basis = basis_funs(span, parameter, degree, knots);
        for (offset, value) in basis.into_iter().enumerate() {
            matrix[row][span - degree + offset] = value;
        }
    }
    matrix
}

fn solve_point_system(
    matrix: Vec<Vec<f64>>,
    points: &[Point3],
    tolerance: f64,
) -> Result<Vec<Point3>, NurbsError> {
    let x = solve_scalar_system(
        matrix.clone(),
        points.iter().map(|point| point.x).collect(),
        tolerance,
    )?;
    let y = solve_scalar_system(
        matrix.clone(),
        points.iter().map(|point| point.y).collect(),
        tolerance,
    )?;
    let z = solve_scalar_system(
        matrix,
        points.iter().map(|point| point.z).collect(),
        tolerance,
    )?;
    Ok((0..points.len())
        .map(|index| Point3::new(x[index], y[index], z[index]))
        .collect())
}

fn solve_scalar_system(
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<f64>,
    tolerance: f64,
) -> Result<Vec<f64>, NurbsError> {
    let n = rhs.len();
    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col][pivot_col].abs();
        for (row, values) in matrix.iter().enumerate().take(n).skip(pivot_col + 1) {
            let candidate = values[pivot_col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= tolerance.max(1.0e-14) {
            return Err(NurbsError::InvalidParameter);
        }
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }
        let pivot = matrix[pivot_col][pivot_col];
        for value in matrix[pivot_col].iter_mut().skip(pivot_col) {
            *value /= pivot;
        }
        rhs[pivot_col] /= pivot;
        let pivot_values = matrix[pivot_col].clone();
        for row in 0..n {
            if row == pivot_col {
                continue;
            }
            let factor = matrix[row][pivot_col];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for (col, pivot_value) in pivot_values.iter().enumerate().skip(pivot_col) {
                matrix[row][col] -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }
    Ok(rhs)
}

fn finite_point(point: Point3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

/// Rational tensor-product B-spline surface.
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsSurface {
    /// Knot vector in U.
    pub u_knots: KnotVector,
    /// Knot vector in V.
    pub v_knots: KnotVector,
    /// Number of control points in U.
    pub u_count: usize,
    /// Number of control points in V.
    pub v_count: usize,
    /// Row-major control points, indexed `v * u_count + u`.
    pub control_points: Vec<Point3>,
    /// Row-major weights.
    pub weights: Vec<f64>,
}

impl NurbsSurface {
    /// Construct a NURBS surface.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        u_count: usize,
        v_count: usize,
        control_points: Vec<Point3>,
        weights: Vec<f64>,
    ) -> Result<Self, NurbsError> {
        if u_count * v_count != control_points.len()
            || control_points.len() != weights.len()
            || u_count < u_degree + 1
            || v_count < v_degree + 1
        {
            return Err(NurbsError::InvalidControlNet);
        }
        let u_knots = KnotVector::new(u_degree, u_knots)?;
        let v_knots = KnotVector::new(v_degree, v_knots)?;
        if u_knots.knots.len() != u_count + u_degree + 1
            || v_knots.knots.len() != v_count + v_degree + 1
        {
            return Err(NurbsError::InvalidKnotVector);
        }
        Ok(Self {
            u_knots,
            v_knots,
            u_count,
            v_count,
            control_points,
            weights,
        })
    }

    /// Bilinear planar NURBS patch.
    pub fn bilinear(points: [[Point3; 2]; 2]) -> Self {
        Self::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            2,
            2,
            vec![points[0][0], points[0][1], points[1][0], points[1][1]],
            vec![1.0; 4],
        )
        .expect("valid bilinear patch")
    }

    /// U/V domains.
    pub fn domain(&self) -> ((f64, f64), (f64, f64)) {
        (self.u_knots.domain(), self.v_knots.domain())
    }

    /// Evaluate surface position.
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.evaluate_homogeneous(u, v, 0, 0).0.to_point()
    }

    /// Surface partial derivatives `(du, dv)`.
    pub fn partials(&self, u: f64, v: f64) -> (Vec3, Vec3) {
        let (s, su, sv) = self.evaluate_homogeneous(u, v, 1, 1);
        let du = (su.xyz() * s.w - s.xyz() * su.w) / (s.w * s.w);
        let dv = (sv.xyz() * s.w - s.xyz() * sv.w) / (s.w * s.w);
        (du, dv)
    }

    /// Unit normal from first partial derivatives.
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let (du, dv) = self.partials(u, v);
        du.cross(dv).normalized()
    }

    fn control(&self, u: usize, v: usize) -> Vec4 {
        let idx = v * self.u_count + u;
        Vec4::from_point_weight(self.control_points[idx], self.weights[idx])
    }

    fn evaluate_homogeneous(
        &self,
        u: f64,
        v: f64,
        du_order: usize,
        dv_order: usize,
    ) -> (Vec4, Vec4, Vec4) {
        let p = self.u_knots.degree;
        let q = self.v_knots.degree;
        let uspan = self.u_knots.find_span(self.u_count, u);
        let vspan = self.v_knots.find_span(self.v_count, v);
        let nu = ders_basis_funs(uspan, u, p, du_order.min(1), &self.u_knots.knots);
        let nv = ders_basis_funs(vspan, v, q, dv_order.min(1), &self.v_knots.knots);

        let mut s = Vec4::default();
        let mut su = Vec4::default();
        let mut sv = Vec4::default();
        let zero_u = vec![0.0; p + 1];
        let zero_v = vec![0.0; q + 1];
        let nu1 = if du_order > 0 { &nu[1] } else { &zero_u };
        let nv1 = if dv_order > 0 { &nv[1] } else { &zero_v };

        for (l, (&nv0_l, &nv1_l)) in nv[0].iter().zip(nv1.iter()).enumerate().take(q + 1) {
            for (k, (&nu0_k, &nu1_k)) in nu[0].iter().zip(nu1.iter()).enumerate().take(p + 1) {
                let cp = self.control(uspan - p + k, vspan - q + l);
                let b = nu0_k * nv0_l;
                s = s + cp * b;
                if du_order > 0 {
                    su = su + cp * (nu1_k * nv0_l);
                }
                if dv_order > 0 {
                    sv = sv + cp * (nu0_k * nv1_l);
                }
            }
        }
        (s, su, sv)
    }
}

/// Basis functions for a span.
pub fn basis_funs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut basis = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    basis[0] = 1.0;
    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom.abs() <= f64::EPSILON {
                0.0
            } else {
                basis[r] / denom
            };
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
    basis
}

/// Basis functions and derivatives up to `derivative_order`.
pub fn ders_basis_funs(
    span: usize,
    u: f64,
    degree: usize,
    derivative_order: usize,
    knots: &[f64],
) -> Vec<Vec<f64>> {
    let n = derivative_order.min(degree);
    let mut ders = vec![vec![0.0; degree + 1]; derivative_order + 1];
    let mut ndu = vec![vec![0.0; degree + 1]; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];

    ndu[0][0] = 1.0;
    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = if ndu[j][r].abs() <= f64::EPSILON {
                0.0
            } else {
                ndu[r][j - 1] / ndu[j][r]
            };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    for (j, value) in ders[0].iter_mut().enumerate().take(degree + 1) {
        *value = ndu[j][degree];
    }

    let mut a = vec![vec![0.0; degree + 1]; 2];
    for r in 0..=degree {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;
        for k in 1..=n {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = degree as isize - k as isize;
            if r >= k {
                let denom = ndu[(pk + 1) as usize][rk as usize];
                a[s2][0] = if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    a[s1][0] / denom
                };
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if r as isize - 1 <= pk {
                k - 1
            } else {
                degree - r
            };
            for j in j1..=j2 {
                let denom = ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                a[s2][j] = if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    (a[s1][j] - a[s1][j - 1]) / denom
                };
                d += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }
            if r as isize <= pk {
                let denom = ndu[(pk + 1) as usize][r];
                a[s2][k] = if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    -a[s1][k - 1] / denom
                };
                d += a[s2][k] * ndu[r][pk as usize];
            }
            ders[k][r] = d;
            core::mem::swap(&mut s1, &mut s2);
        }
    }

    let mut factor = degree as f64;
    for row in ders.iter_mut().take(n + 1).skip(1) {
        for value in row.iter_mut().take(degree + 1) {
            *value *= factor;
        }
        factor *= (degree - 1) as f64;
    }
    ders
}
