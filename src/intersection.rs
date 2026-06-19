//! Curve/surface and surface/surface intersection routines.

use crate::geometry::{Line, Plane};
use crate::math::{Point3, Vec2, Vec3};
use crate::nurbs::{NurbsCurve, NurbsSurface};
use crate::predicates::{Interval, RobustSign};
use crate::tessellation::tessellate_nurbs_surface;
use crate::topology::{EdgeCurve3D, FaceId, Solid, SplitFacesReport, TopologyError, TrimCurve2D};

/// Classification for a line-plane intersection.
#[derive(Clone, Debug, PartialEq)]
pub enum LinePlaneIntersection {
    /// No intersection.
    Empty,
    /// One point with line parameter.
    Point {
        /// Line parameter.
        t: f64,
        /// Intersection point.
        point: Point3,
        /// Plane residual.
        residual: f64,
    },
    /// The line lies in the plane.
    Coincident,
    /// The denominator could not be certified.
    Uncertain,
}

/// Classification for a plane-plane intersection.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanePlaneIntersection {
    /// Parallel disjoint planes.
    Empty,
    /// One intersection line.
    Line(Line),
    /// Coincident planes.
    Coincident,
}

/// Type of curve/surface hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitKind {
    /// The curve crosses the surface.
    Crossing,
    /// The curve is tangent or nearly tangent to the surface.
    Tangent,
}

/// Curve/surface hit.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveSurfaceHit {
    /// Curve parameter.
    pub u: f64,
    /// Intersection point.
    pub point: Point3,
    /// Signed residual.
    pub residual: f64,
    /// Crossing/tangent classification.
    pub kind: HitKind,
}

/// Polyline approximation to a surface/surface intersection curve.
#[derive(Clone, Debug, PartialEq)]
pub struct IntersectionPolyline {
    /// Points on the intersection curve.
    pub points: Vec<Point3>,
    /// Maximum absolute residual sampled or refined while marching.
    pub max_residual: f64,
}

/// One refined sample on a surface/surface intersection curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSurfaceIntersectionPoint {
    /// Refined model-space point on the intersection.
    pub point: Point3,
    /// Parameter on the first surface.
    pub a_uv: Vec2,
    /// Parameter on the second surface.
    pub b_uv: Vec2,
    /// Distance between the two refined surface evaluations.
    pub residual: f64,
}

/// Trim-ready approximation of a NURBS/NURBS surface intersection curve.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimReadyIntersectionCurve {
    /// Refined samples in curve order.
    pub points: Vec<SurfaceSurfaceIntersectionPoint>,
    /// Model-space curve suitable for a topological `Edge`.
    pub edge_curve: EdgeCurve3D,
    /// P-curve on the first surface.
    pub a_pcurve: TrimCurve2D,
    /// P-curve on the second surface.
    pub b_pcurve: TrimCurve2D,
    /// Maximum refined residual across samples.
    pub max_residual: f64,
}

impl TrimReadyIntersectionCurve {
    /// Convert to the legacy point-only polyline form.
    pub fn to_polyline(&self) -> IntersectionPolyline {
        IntersectionPolyline {
            points: self.points.iter().map(|sample| sample.point).collect(),
            max_residual: self.max_residual,
        }
    }

    /// Install this trim-ready SSI curve as a staged split between two faces.
    pub fn split_faces(
        &self,
        solid: &mut Solid,
        a_face: FaceId,
        b_face: FaceId,
        tolerance: f64,
    ) -> Result<SplitFacesReport, TopologyError> {
        solid.split_faces_with_curves(
            a_face,
            b_face,
            self.edge_curve.clone(),
            self.a_pcurve.clone(),
            self.b_pcurve.clone(),
            tolerance,
        )
    }
}

/// Intersect an infinite line with an infinite plane.
pub fn intersect_line_plane(line: Line, plane: Plane, tol: f64) -> LinePlaneIntersection {
    let denom = plane.normal.dot(line.direction);
    let numerator = -plane.signed_distance(line.origin);
    let denom_interval = Interval::point(plane.normal.x)
        .mul_i(Interval::point(line.direction.x))
        .add_i(Interval::point(plane.normal.y).mul_i(Interval::point(line.direction.y)))
        .add_i(Interval::point(plane.normal.z).mul_i(Interval::point(line.direction.z)));

    if denom_interval.sign() == RobustSign::Uncertain && denom.abs() <= tol {
        if numerator.abs() <= tol {
            return LinePlaneIntersection::Coincident;
        }
        return LinePlaneIntersection::Empty;
    }
    if denom.abs() <= tol {
        return if numerator.abs() <= tol {
            LinePlaneIntersection::Coincident
        } else {
            LinePlaneIntersection::Empty
        };
    }
    let t = numerator / denom;
    let point = line.point_at(t);
    LinePlaneIntersection::Point {
        t,
        point,
        residual: plane.signed_distance(point),
    }
}

/// Intersect two planes.
pub fn intersect_plane_plane(a: Plane, b: Plane, tol: f64) -> PlanePlaneIntersection {
    let direction = a.normal.cross(b.normal);
    let denom = direction.norm_squared();
    if denom <= tol * tol {
        return if a.signed_distance(b.origin).abs() <= tol {
            PlanePlaneIntersection::Coincident
        } else {
            PlanePlaneIntersection::Empty
        };
    }
    let c1 = a.constant();
    let c2 = b.constant();
    let point = (b.normal * c1 - a.normal * c2).cross(direction) / denom;
    PlanePlaneIntersection::Line(Line::new(point, direction))
}

/// Intersect a NURBS curve with a plane using bracketing and bisection.
pub fn intersect_curve_plane(
    curve: &NurbsCurve,
    plane: Plane,
    samples: usize,
    tol: f64,
) -> Vec<CurveSurfaceHit> {
    let (u0, u1) = curve.domain();
    let samples = samples.max(4);
    let mut params = Vec::with_capacity(samples + 1);
    let mut values = Vec::with_capacity(samples + 1);
    for i in 0..=samples {
        let t = i as f64 / samples as f64;
        let u = u0 * (1.0 - t) + u1 * t;
        params.push(u);
        values.push(plane.signed_distance(curve.evaluate(u)));
    }

    let mut hits = Vec::new();
    for i in 0..samples {
        let a = params[i];
        let b = params[i + 1];
        let fa = values[i];
        let fb = values[i + 1];
        if fa.abs() <= tol {
            push_curve_hit(&mut hits, curve, plane, a, tol);
        }
        if fa * fb < 0.0 {
            let root = bisect_curve_plane(curve, plane, a, b, tol);
            push_curve_hit(&mut hits, curve, plane, root, tol);
        } else if fa.abs().min(fb.abs()) > tol && local_minimum(values.as_slice(), i, tol * 10.0) {
            let mid = (a + b) * 0.5;
            if plane.signed_distance(curve.evaluate(mid)).abs() <= tol * 10.0 {
                push_curve_hit(&mut hits, curve, plane, mid, tol * 10.0);
            }
        }
    }
    if values[samples].abs() <= tol {
        push_curve_hit(&mut hits, curve, plane, params[samples], tol);
    }
    hits.sort_by(|a, b| a.u.total_cmp(&b.u));
    hits.dedup_by(|a, b| (a.u - b.u).abs() <= tol);
    hits
}

/// March a plane/NURBS-surface intersection into short polylines.
pub fn intersect_plane_nurbs_surface(
    plane: Plane,
    surface: &NurbsSurface,
    u_steps: usize,
    v_steps: usize,
    tol: f64,
) -> Vec<IntersectionPolyline> {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let u_steps = u_steps.max(2);
    let v_steps = v_steps.max(2);
    let mut values = vec![0.0; (u_steps + 1) * (v_steps + 1)];
    let idx = |i: usize, j: usize| j * (u_steps + 1) + i;
    for j in 0..=v_steps {
        for i in 0..=u_steps {
            let u = lerp(u0, u1, i as f64 / u_steps as f64);
            let v = lerp(v0, v1, j as f64 / v_steps as f64);
            values[idx(i, j)] = plane.signed_distance(surface.evaluate(u, v));
        }
    }

    let mut polylines = Vec::new();
    for j in 0..v_steps {
        for i in 0..u_steps {
            let corners = [
                (i, j, values[idx(i, j)]),
                (i + 1, j, values[idx(i + 1, j)]),
                (i + 1, j + 1, values[idx(i + 1, j + 1)]),
                (i, j + 1, values[idx(i, j + 1)]),
            ];
            let mut crossings = Vec::<(f64, f64, f64)>::new();
            for e in 0..4 {
                let (ia, ja, fa) = corners[e];
                let (ib, jb, fb) = corners[(e + 1) % 4];
                if fa.abs() <= tol {
                    crossings.push((ia as f64, ja as f64, fa.abs()));
                }
                if fa * fb < 0.0 {
                    let t = fa / (fa - fb);
                    crossings.push((
                        lerp(ia as f64, ib as f64, t),
                        lerp(ja as f64, jb as f64, t),
                        0.0,
                    ));
                }
            }
            crossings.dedup_by(|a, b| (a.0 - b.0).abs() <= 1.0e-9 && (a.1 - b.1).abs() <= 1.0e-9);
            if crossings.len() >= 2 {
                let mut points = Vec::new();
                let mut max_residual: f64 = 0.0;
                for (gi, gj, residual) in crossings.into_iter().take(2) {
                    let u = lerp(u0, u1, gi / u_steps as f64);
                    let v = lerp(v0, v1, gj / v_steps as f64);
                    let point = surface.evaluate(u, v);
                    max_residual =
                        max_residual.max(residual.max(plane.signed_distance(point).abs()));
                    points.push(point);
                }
                polylines.push(IntersectionPolyline {
                    points,
                    max_residual,
                });
            }
        }
    }
    polylines
}

/// Intersect two NURBS surfaces by tessellated discovery plus NURBS residual refinement.
///
/// This is a practical first NURBS/NURBS SSI stage: both surfaces are sampled into
/// triangle grids, candidate triangle-triangle intersection segments are found,
/// segment endpoints are refined against the original parametric surfaces, and
/// the stitched result carries a 3D edge curve plus one p-curve on each surface.
/// Coplanar overlap regions are treated as degenerate and are not emitted.
pub fn intersect_nurbs_surfaces(
    a: &NurbsSurface,
    b: &NurbsSurface,
    u_steps: usize,
    v_steps: usize,
    tol: f64,
) -> Vec<TrimReadyIntersectionCurve> {
    let tol = tol.max(1.0e-12);
    let stitch_tol = (tol * 64.0).max(1.0e-7);
    let a_triangles = tessellated_surface_triangles(a, u_steps.max(2), v_steps.max(2));
    let b_triangles = tessellated_surface_triangles(b, u_steps.max(2), v_steps.max(2));
    let mut raw_segments = Vec::new();

    for a_triangle in &a_triangles {
        for b_triangle in &b_triangles {
            let Some(pair) = triangle_pair_intersection(a_triangle, b_triangle, tol) else {
                continue;
            };
            let p0 = refine_surface_pair(a, b, pair[0], tol);
            let p1 = refine_surface_pair(a, b, pair[1], tol);
            push_unique_segment(
                &mut raw_segments,
                RawIntersectionSegment {
                    a: p0,
                    b: p1,
                    max_residual: p0.residual.max(p1.residual),
                },
                stitch_tol,
            );
        }
    }

    stitch_segments(raw_segments, stitch_tol)
}

#[derive(Clone, Copy, Debug)]
struct SurfaceSample {
    point: Point3,
    u: f64,
    v: f64,
}

#[derive(Clone, Copy, Debug)]
struct SurfaceTriangle {
    vertices: [SurfaceSample; 3],
    normal: Vec3,
    normal_len: f64,
    min: Point3,
    max: Point3,
}

#[derive(Clone, Copy, Debug)]
struct SurfaceCut {
    a: SurfaceSample,
    b: SurfaceSample,
}

#[derive(Clone, Copy, Debug)]
struct SurfacePairPoint {
    a: SurfaceSample,
    b: SurfaceSample,
}

#[derive(Clone, Copy, Debug)]
struct RawIntersectionSegment {
    a: SurfaceSurfaceIntersectionPoint,
    b: SurfaceSurfaceIntersectionPoint,
    max_residual: f64,
}

#[derive(Clone, Copy, Debug)]
struct GraphEdge {
    a: usize,
    b: usize,
    max_residual: f64,
}

fn tessellated_surface_triangles(
    surface: &NurbsSurface,
    u_steps: usize,
    v_steps: usize,
) -> Vec<SurfaceTriangle> {
    let mesh = tessellate_nurbs_surface(surface, u_steps, v_steps);
    let mut triangles = Vec::with_capacity(mesh.indices.len());
    for indices in mesh.indices {
        let vertices = indices.map(|index| {
            let vertex = mesh.vertices[index as usize];
            SurfaceSample {
                point: vertex.position,
                u: vertex.u,
                v: vertex.v,
            }
        });
        let normal =
            (vertices[1].point - vertices[0].point).cross(vertices[2].point - vertices[0].point);
        let normal_len = normal.norm();
        if normal_len <= f64::EPSILON {
            continue;
        }
        let (min, max) = triangle_bounds(vertices);
        triangles.push(SurfaceTriangle {
            vertices,
            normal,
            normal_len,
            min,
            max,
        });
    }
    triangles
}

fn triangle_pair_intersection(
    a: &SurfaceTriangle,
    b: &SurfaceTriangle,
    tol: f64,
) -> Option<[SurfacePairPoint; 2]> {
    if !bounds_overlap(a.min, a.max, b.min, b.max, tol) {
        return None;
    }

    let axis = a.normal.cross(b.normal);
    let axis_len = axis.norm();
    let parallel_tol = tol * a.normal_len * b.normal_len;
    if axis_len <= parallel_tol {
        return None;
    }

    let b_to_a = distances_to_plane(b, a);
    let a_plane_eps = tol * a.normal_len;
    if all_on_one_side(&b_to_a, a_plane_eps) {
        return None;
    }

    let a_to_b = distances_to_plane(a, b);
    let b_plane_eps = tol * b.normal_len;
    if all_on_one_side(&a_to_b, b_plane_eps) {
        return None;
    }

    let a_cut = cut_triangle_by_plane(a, &a_to_b, b_plane_eps, tol)?;
    let b_cut = cut_triangle_by_plane(b, &b_to_a, a_plane_eps, tol)?;
    overlap_surface_cuts(a_cut, b_cut, axis / axis_len, tol)
}

fn distances_to_plane(triangle: &SurfaceTriangle, plane_triangle: &SurfaceTriangle) -> [f64; 3] {
    triangle.vertices.map(|vertex| {
        plane_triangle
            .normal
            .dot(vertex.point - plane_triangle.vertices[0].point)
    })
}

fn all_on_one_side(distances: &[f64; 3], eps: f64) -> bool {
    distances.iter().all(|d| *d > eps) || distances.iter().all(|d| *d < -eps)
}

fn cut_triangle_by_plane(
    triangle: &SurfaceTriangle,
    distances: &[f64; 3],
    plane_eps: f64,
    tol: f64,
) -> Option<SurfaceCut> {
    let mut samples = Vec::with_capacity(4);
    for (vertex, distance) in triangle.vertices.iter().zip(distances) {
        if distance.abs() <= plane_eps {
            push_unique_sample(&mut samples, *vertex, tol);
        }
    }
    for edge in 0..3 {
        let next = (edge + 1) % 3;
        let a_distance = distances[edge];
        let b_distance = distances[next];
        if (a_distance > plane_eps && b_distance < -plane_eps)
            || (a_distance < -plane_eps && b_distance > plane_eps)
        {
            let t = a_distance / (a_distance - b_distance);
            push_unique_sample(
                &mut samples,
                interpolate_sample(triangle.vertices[edge], triangle.vertices[next], t),
                tol,
            );
        } else if a_distance.abs() <= plane_eps && b_distance.abs() <= plane_eps {
            push_unique_sample(&mut samples, triangle.vertices[edge], tol);
            push_unique_sample(&mut samples, triangle.vertices[next], tol);
        }
    }

    furthest_sample_pair(&samples, tol).map(|(a, b)| SurfaceCut { a, b })
}

fn overlap_surface_cuts(
    a: SurfaceCut,
    b: SurfaceCut,
    axis: Vec3,
    tol: f64,
) -> Option<[SurfacePairPoint; 2]> {
    let a0 = a.a.point.dot(axis);
    let a1 = a.b.point.dot(axis);
    let b0 = b.a.point.dot(axis);
    let b1 = b.b.point.dot(axis);
    let a_min = a0.min(a1);
    let a_max = a0.max(a1);
    let b_min = b0.min(b1);
    let b_max = b0.max(b1);
    let low = a_min.max(b_min);
    let high = a_max.min(b_max);
    if high - low <= tol {
        return None;
    }

    let p0 = SurfacePairPoint {
        a: sample_on_cut_at(a, axis, low, tol),
        b: sample_on_cut_at(b, axis, low, tol),
    };
    let p1 = SurfacePairPoint {
        a: sample_on_cut_at(a, axis, high, tol),
        b: sample_on_cut_at(b, axis, high, tol),
    };
    if p0
        .a
        .point
        .distance(p0.b.point)
        .max(p1.a.point.distance(p1.b.point))
        > tol * 128.0
    {
        return None;
    }
    Some([p0, p1])
}

fn refine_surface_pair(
    a: &NurbsSurface,
    b: &NurbsSurface,
    guess: SurfacePairPoint,
    tol: f64,
) -> SurfaceSurfaceIntersectionPoint {
    let ((au0, au1), (av0, av1)) = a.domain();
    let ((bu0, bu1), (bv0, bv1)) = b.domain();
    let mut au = guess.a.u.clamp(au0, au1);
    let mut av = guess.a.v.clamp(av0, av1);
    let mut bu = guess.b.u.clamp(bu0, bu1);
    let mut bv = guess.b.v.clamp(bv0, bv1);

    for _ in 0..20 {
        let pa = a.evaluate(au, av);
        let pb = b.evaluate(bu, bv);
        let diff = pa - pb;
        let residual = diff.norm();
        if residual <= tol {
            break;
        }

        let (a_du, a_dv) = a.partials(au, av);
        let (b_du, b_dv) = b.partials(bu, bv);
        let columns = [a_du, a_dv, -b_du, -b_dv];
        let Some(lambda) = solve_gram_3x3(columns, diff, tol) else {
            break;
        };
        let step = [
            -columns[0].dot(lambda),
            -columns[1].dot(lambda),
            -columns[2].dot(lambda),
            -columns[3].dot(lambda),
        ];
        if step.iter().any(|value| !value.is_finite()) {
            break;
        }

        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..8 {
            let next_au = (au + step[0] * scale).clamp(au0, au1);
            let next_av = (av + step[1] * scale).clamp(av0, av1);
            let next_bu = (bu + step[2] * scale).clamp(bu0, bu1);
            let next_bv = (bv + step[3] * scale).clamp(bv0, bv1);
            let next_residual =
                (a.evaluate(next_au, next_av) - b.evaluate(next_bu, next_bv)).norm();
            if next_residual < residual {
                au = next_au;
                av = next_av;
                bu = next_bu;
                bv = next_bv;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    let pa = a.evaluate(au, av);
    let pb = b.evaluate(bu, bv);
    let residual = pa.distance(pb);
    SurfaceSurfaceIntersectionPoint {
        point: (pa + pb) * 0.5,
        a_uv: Vec2::new(au, av),
        b_uv: Vec2::new(bu, bv),
        residual,
    }
}

fn solve_gram_3x3(columns: [Vec3; 4], rhs: Vec3, tol: f64) -> Option<Vec3> {
    let mut matrix = [[0.0; 3]; 3];
    for column in columns {
        matrix[0][0] += column.x * column.x;
        matrix[0][1] += column.x * column.y;
        matrix[0][2] += column.x * column.z;
        matrix[1][0] += column.y * column.x;
        matrix[1][1] += column.y * column.y;
        matrix[1][2] += column.y * column.z;
        matrix[2][0] += column.z * column.x;
        matrix[2][1] += column.z * column.y;
        matrix[2][2] += column.z * column.z;
    }
    solve_3x3(matrix, rhs, tol)
}

fn solve_3x3(matrix: [[f64; 3]; 3], rhs: Vec3, tol: f64) -> Option<Vec3> {
    let det = determinant_3x3(matrix);
    if det.abs() <= tol * tol {
        return None;
    }
    let rhs_column = [rhs.x, rhs.y, rhs.z];
    let mut mx = matrix;
    let mut my = matrix;
    let mut mz = matrix;
    for row in 0..3 {
        mx[row][0] = rhs_column[row];
        my[row][1] = rhs_column[row];
        mz[row][2] = rhs_column[row];
    }
    Some(Vec3::new(
        determinant_3x3(mx) / det,
        determinant_3x3(my) / det,
        determinant_3x3(mz) / det,
    ))
}

fn determinant_3x3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn stitch_segments(
    segments: Vec<RawIntersectionSegment>,
    stitch_tol: f64,
) -> Vec<TrimReadyIntersectionCurve> {
    let mut vertices = Vec::<SurfaceSurfaceIntersectionPoint>::new();
    let mut edges = Vec::<GraphEdge>::new();
    for segment in segments {
        let a = graph_vertex(&mut vertices, segment.a, stitch_tol);
        let b = graph_vertex(&mut vertices, segment.b, stitch_tol);
        if a == b {
            continue;
        }
        if edges
            .iter()
            .any(|edge| (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a))
        {
            continue;
        }
        edges.push(GraphEdge {
            a,
            b,
            max_residual: segment.max_residual,
        });
    }

    let mut adjacency = vec![Vec::<usize>::new(); vertices.len()];
    for (index, edge) in edges.iter().enumerate() {
        adjacency[edge.a].push(index);
        adjacency[edge.b].push(index);
    }

    let mut visited = vec![false; edges.len()];
    let mut curves = Vec::new();
    for start in 0..vertices.len() {
        if adjacency[start].len() == 2 {
            continue;
        }
        while let Some(edge_index) = next_unvisited_edge(start, &adjacency, &visited) {
            let curve = walk_intersection_curve(
                start,
                edge_index,
                &vertices,
                &edges,
                &adjacency,
                &mut visited,
            );
            if curve.points.len() >= 2 {
                curves.push(curve);
            }
        }
    }

    for edge_index in 0..edges.len() {
        if visited[edge_index] {
            continue;
        }
        let start = edges[edge_index].a;
        let curve = walk_intersection_curve(
            start,
            edge_index,
            &vertices,
            &edges,
            &adjacency,
            &mut visited,
        );
        if curve.points.len() >= 2 {
            curves.push(curve);
        }
    }

    curves.sort_by(|a, b| {
        b.points
            .len()
            .cmp(&a.points.len())
            .then_with(|| compare_points(a.points[0].point, b.points[0].point))
    });
    curves
}

fn walk_intersection_curve(
    start: usize,
    first_edge: usize,
    vertices: &[SurfaceSurfaceIntersectionPoint],
    edges: &[GraphEdge],
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
) -> TrimReadyIntersectionCurve {
    let mut points = vec![vertices[start]];
    let mut current = start;
    let mut next_edge = Some(first_edge);
    let mut max_residual: f64 = 0.0;
    while let Some(edge_index) = next_edge {
        if visited[edge_index] {
            break;
        }
        visited[edge_index] = true;
        let edge = edges[edge_index];
        max_residual = max_residual.max(edge.max_residual);
        current = if edge.a == current { edge.b } else { edge.a };
        points.push(vertices[current]);
        next_edge = next_unvisited_edge(current, adjacency, visited);
        if current == start {
            break;
        }
    }
    trim_ready_curve(points, max_residual)
}

fn next_unvisited_edge(vertex: usize, adjacency: &[Vec<usize>], visited: &[bool]) -> Option<usize> {
    adjacency[vertex]
        .iter()
        .copied()
        .find(|edge_index| !visited[*edge_index])
}

fn graph_vertex(
    vertices: &mut Vec<SurfaceSurfaceIntersectionPoint>,
    point: SurfaceSurfaceIntersectionPoint,
    tol: f64,
) -> usize {
    if let Some(index) = vertices
        .iter()
        .position(|existing| existing.point.distance(point.point) <= tol)
    {
        if point.residual < vertices[index].residual {
            vertices[index] = point;
        }
        index
    } else {
        vertices.push(point);
        vertices.len() - 1
    }
}

fn push_unique_segment(
    segments: &mut Vec<RawIntersectionSegment>,
    segment: RawIntersectionSegment,
    tol: f64,
) {
    if segment.a.point.distance(segment.b.point) <= tol {
        return;
    }
    if segments.iter().any(|existing| {
        (existing.a.point.distance(segment.a.point) <= tol
            && existing.b.point.distance(segment.b.point) <= tol)
            || (existing.a.point.distance(segment.b.point) <= tol
                && existing.b.point.distance(segment.a.point) <= tol)
    }) {
        return;
    }
    segments.push(segment);
}

fn trim_ready_curve(
    points: Vec<SurfaceSurfaceIntersectionPoint>,
    edge_residual: f64,
) -> TrimReadyIntersectionCurve {
    let max_residual = points
        .iter()
        .fold(edge_residual, |acc, point| acc.max(point.residual));
    TrimReadyIntersectionCurve {
        edge_curve: edge_curve_from_samples(&points),
        a_pcurve: pcurve_from_samples(&points, SurfaceSide::A),
        b_pcurve: pcurve_from_samples(&points, SurfaceSide::B),
        points,
        max_residual,
    }
}

fn edge_curve_from_samples(points: &[SurfaceSurfaceIntersectionPoint]) -> EdgeCurve3D {
    if points.len() == 2 {
        EdgeCurve3D::line_segment(points[0].point, points[1].point)
    } else {
        EdgeCurve3D::Polyline {
            points: points.iter().map(|sample| sample.point).collect(),
        }
    }
}

#[derive(Clone, Copy)]
enum SurfaceSide {
    A,
    B,
}

fn pcurve_from_samples(
    points: &[SurfaceSurfaceIntersectionPoint],
    side: SurfaceSide,
) -> TrimCurve2D {
    let uv = |point: &SurfaceSurfaceIntersectionPoint| match side {
        SurfaceSide::A => point.a_uv,
        SurfaceSide::B => point.b_uv,
    };
    if points.len() == 2 {
        TrimCurve2D::LineSegment {
            start: uv(&points[0]),
            end: uv(&points[1]),
        }
    } else {
        TrimCurve2D::Polyline {
            points: points.iter().map(uv).collect(),
        }
    }
}

fn push_unique_sample(samples: &mut Vec<SurfaceSample>, sample: SurfaceSample, tol: f64) {
    if samples
        .iter()
        .all(|existing| existing.point.distance(sample.point) > tol)
    {
        samples.push(sample);
    }
}

fn furthest_sample_pair(
    samples: &[SurfaceSample],
    tol: f64,
) -> Option<(SurfaceSample, SurfaceSample)> {
    let mut best = None;
    let mut best_distance = tol;
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let distance = samples[i].point.distance(samples[j].point);
            if distance > best_distance {
                best_distance = distance;
                best = Some((samples[i], samples[j]));
            }
        }
    }
    best
}

fn sample_on_cut_at(cut: SurfaceCut, axis: Vec3, target: f64, tol: f64) -> SurfaceSample {
    let a = cut.a.point.dot(axis);
    let b = cut.b.point.dot(axis);
    let denom = b - a;
    if denom.abs() <= tol {
        cut.a
    } else {
        interpolate_sample(cut.a, cut.b, (target - a) / denom)
    }
}

fn interpolate_sample(a: SurfaceSample, b: SurfaceSample, t: f64) -> SurfaceSample {
    SurfaceSample {
        point: a.point.lerp(b.point, t),
        u: lerp(a.u, b.u, t),
        v: lerp(a.v, b.v, t),
    }
}

fn triangle_bounds(vertices: [SurfaceSample; 3]) -> (Point3, Point3) {
    let mut min = vertices[0].point;
    let mut max = vertices[0].point;
    for vertex in vertices.iter().skip(1) {
        min = min_point(min, vertex.point);
        max = max_point(max, vertex.point);
    }
    (min, max)
}

fn bounds_overlap(a_min: Point3, a_max: Point3, b_min: Point3, b_max: Point3, tol: f64) -> bool {
    a_min.x <= b_max.x + tol
        && a_max.x + tol >= b_min.x
        && a_min.y <= b_max.y + tol
        && a_max.y + tol >= b_min.y
        && a_min.z <= b_max.z + tol
        && a_max.z + tol >= b_min.z
}

fn min_point(a: Point3, b: Point3) -> Point3 {
    Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

fn max_point(a: Point3, b: Point3) -> Point3 {
    Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

fn compare_points(a: Point3, b: Point3) -> core::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn bisect_curve_plane(curve: &NurbsCurve, plane: Plane, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let mut fa = plane.signed_distance(curve.evaluate(a));
    for _ in 0..80 {
        let mid = (a + b) * 0.5;
        let fm = plane.signed_distance(curve.evaluate(mid));
        if fm.abs() <= tol || (b - a).abs() <= tol {
            return mid;
        }
        if fa * fm <= 0.0 {
            b = mid;
        } else {
            a = mid;
            fa = fm;
        }
    }
    (a + b) * 0.5
}

fn push_curve_hit(
    hits: &mut Vec<CurveSurfaceHit>,
    curve: &NurbsCurve,
    plane: Plane,
    u: f64,
    tol: f64,
) {
    let point = curve.evaluate(u);
    let residual = plane.signed_distance(point);
    let speed = plane.normal.dot(curve.derivative(u)).abs();
    let kind = if speed <= tol.sqrt() {
        HitKind::Tangent
    } else {
        HitKind::Crossing
    };
    hits.push(CurveSurfaceHit {
        u,
        point,
        residual,
        kind,
    });
}

fn local_minimum(values: &[f64], i: usize, tol: f64) -> bool {
    if i == 0 || i + 2 >= values.len() {
        return false;
    }
    let y0 = values[i - 1].abs();
    let y1 = values[i].abs();
    let y2 = values[i + 1].abs();
    y1 <= y0 && y1 <= y2 && y1 <= tol
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

#[allow(dead_code)]
fn _line_residual(line: Line, point: Point3) -> f64 {
    let v: Vec3 = point - line.origin;
    v.cross(line.direction).norm()
}
