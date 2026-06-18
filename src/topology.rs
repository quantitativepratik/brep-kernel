//! Half-edge topology for closed manifold B-reps.

use crate::geometry::{Cylinder, Plane};
use crate::math::{Point3, Vec2, Vec3};
use crate::nurbs::NurbsSurface;
use std::collections::HashMap;

/// Vertex identifier.
pub type VertexId = usize;
/// Half-edge identifier.
pub type HalfEdgeId = usize;
/// Edge identifier.
pub type EdgeId = usize;
/// Face identifier.
pub type FaceId = usize;

/// A topological vertex with geometric position.
#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    /// Position in model space.
    pub point: Point3,
    /// One outgoing half-edge.
    pub halfedge: Option<HalfEdgeId>,
}

/// Directed edge around a face.
#[derive(Clone, Debug, PartialEq)]
pub struct HalfEdge {
    /// Origin vertex.
    pub origin: VertexId,
    /// Opposite half-edge, if the mesh is closed.
    pub twin: Option<HalfEdgeId>,
    /// Next half-edge in the face loop.
    pub next: HalfEdgeId,
    /// Previous half-edge in the face loop.
    pub prev: HalfEdgeId,
    /// Incident face.
    pub face: FaceId,
    /// Undirected edge.
    pub edge: EdgeId,
}

/// Undirected topological edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// One of the two half-edges.
    pub halfedge: HalfEdgeId,
}

/// Analytic support surface for a face.
#[derive(Clone, Debug, PartialEq)]
pub enum FaceSurface {
    /// Planar face.
    Plane(Plane),
    /// Z-aligned cylindrical face.
    Cylinder(Cylinder),
    /// NURBS support surface.
    Nurbs(Box<NurbsSurface>),
    /// Faceted or not yet analytically classified.
    Faceted,
}

impl FaceSurface {
    /// Project a model-space point into the surface parameter domain when a direct projection exists.
    ///
    /// Planes use a deterministic orthonormal frame. Cylinders use `(angle, height)`
    /// around the Z-aligned cylinder. NURBS inverse evaluation is deliberately not
    /// hidden here, so this returns `None` for NURBS surfaces.
    pub fn project_point(&self, point: Point3) -> Option<Vec2> {
        match self {
            Self::Plane(plane) => {
                let (u_axis, v_axis) = plane_frame(*plane);
                let d = point - plane.origin;
                Some(Vec2::new(d.dot(u_axis), d.dot(v_axis)))
            }
            Self::Cylinder(cylinder) => {
                let d = point - cylinder.center;
                Some(Vec2::new(d.y.atan2(d.x), d.z))
            }
            Self::Nurbs(_) | Self::Faceted => None,
        }
    }
}

/// Backward-compatible name for the face support surface tag.
pub type FaceGeometry = FaceSurface;

/// Role of a trim loop on a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimLoopKind {
    /// Boundary of the visible face region.
    Outer,
    /// Hole or island-excluding loop inside the outer boundary.
    Inner,
}

/// Two-dimensional curve in a face's parameter domain.
#[derive(Clone, Debug, PartialEq)]
pub enum TrimCurve2D {
    /// Straight p-curve segment from `start` to `end`.
    LineSegment {
        /// Start point in `(u, v)`.
        start: Vec2,
        /// End point in `(u, v)`.
        end: Vec2,
    },
    /// Circular p-curve arc.
    CircularArc {
        /// Arc center in `(u, v)`.
        center: Vec2,
        /// Arc radius.
        radius: f64,
        /// Start angle in radians.
        start_angle: f64,
        /// End angle in radians.
        end_angle: f64,
    },
    /// Piecewise-linear p-curve.
    Polyline {
        /// Ordered polyline points.
        points: Vec<Vec2>,
    },
    /// Topological trim exists, but the p-curve has not been fitted or projected yet.
    Unresolved,
}

impl TrimCurve2D {
    /// Return start and end points if the p-curve has explicit endpoints.
    pub fn endpoints(&self) -> Option<(Vec2, Vec2)> {
        match self {
            Self::LineSegment { start, end } => Some((*start, *end)),
            Self::CircularArc {
                center,
                radius,
                start_angle,
                end_angle,
            } => Some((
                *center + Vec2::new(start_angle.cos() * *radius, start_angle.sin() * *radius),
                *center + Vec2::new(end_angle.cos() * *radius, end_angle.sin() * *radius),
            )),
            Self::Polyline { points } => Some((*points.first()?, *points.last()?)),
            Self::Unresolved => None,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::LineSegment { start, end } => finite_vec2(*start) && finite_vec2(*end),
            Self::CircularArc {
                center,
                radius,
                start_angle,
                end_angle,
            } => {
                finite_vec2(*center)
                    && radius.is_finite()
                    && *radius > 0.0
                    && start_angle.is_finite()
                    && end_angle.is_finite()
            }
            Self::Polyline { points } => {
                points.len() >= 2 && points.iter().all(|point| finite_vec2(*point))
            }
            Self::Unresolved => true,
        }
    }
}

/// One oriented trim edge on a face boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct Trim {
    /// Optional topological half-edge represented by this trim.
    pub halfedge: Option<HalfEdgeId>,
    /// P-curve in the face parameter domain.
    pub curve: TrimCurve2D,
    /// Geometric tolerance for endpoint matching and later classification.
    pub tolerance: f64,
}

impl Trim {
    /// Construct an unresolved trim attached to a half-edge.
    pub fn unresolved(halfedge: HalfEdgeId) -> Self {
        Self {
            halfedge: Some(halfedge),
            curve: TrimCurve2D::Unresolved,
            tolerance: DEFAULT_TRIM_TOLERANCE,
        }
    }

    /// Construct an analytic trim curve that is not tied to an existing half-edge.
    pub fn curve(curve: TrimCurve2D, tolerance: f64) -> Self {
        Self {
            halfedge: None,
            curve,
            tolerance,
        }
    }

    /// Construct a line-segment p-curve attached to a half-edge.
    pub fn line_segment(halfedge: HalfEdgeId, start: Vec2, end: Vec2, tolerance: f64) -> Self {
        Self {
            halfedge: Some(halfedge),
            curve: TrimCurve2D::LineSegment { start, end },
            tolerance,
        }
    }
}

/// Ordered trim loop bounding a face region.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimLoop {
    /// Outer or inner loop role.
    pub kind: TrimLoopKind,
    /// Ordered trim edges.
    pub trims: Vec<Trim>,
}

impl TrimLoop {
    /// Construct a trim loop from explicit trims.
    pub fn new(kind: TrimLoopKind, trims: Vec<Trim>) -> Self {
        Self { kind, trims }
    }

    /// Construct a topological loop from ordered half-edges.
    pub fn from_halfedges(kind: TrimLoopKind, halfedges: Vec<HalfEdgeId>) -> Self {
        Self {
            kind,
            trims: halfedges.into_iter().map(Trim::unresolved).collect(),
        }
    }
}

/// Topological face.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// One half-edge on the outer loop.
    pub halfedge: HalfEdgeId,
    /// Analytic support surface.
    pub surface: FaceSurface,
    /// Ordered outer and inner trim loops on the face.
    pub trim_loops: Vec<TrimLoop>,
}

/// Connected boundary shell.
#[derive(Clone, Debug, PartialEq)]
pub struct Shell {
    /// Faces in the shell.
    pub faces: Vec<FaceId>,
}

/// Solid represented by closed half-edge shells.
#[derive(Clone, Debug, PartialEq)]
pub struct Solid {
    /// Vertices.
    pub vertices: Vec<Vertex>,
    /// Half-edges.
    pub halfedges: Vec<HalfEdge>,
    /// Undirected edges.
    pub edges: Vec<Edge>,
    /// Faces.
    pub faces: Vec<Face>,
    /// Shells.
    pub shells: Vec<Shell>,
}

/// Stable topology counts used by golden reference models and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyCounts {
    /// Number of vertices.
    pub vertices: usize,
    /// Number of undirected edges.
    pub edges: usize,
    /// Number of half-edges.
    pub halfedges: usize,
    /// Number of faces.
    pub faces: usize,
    /// Number of shells.
    pub shells: usize,
    /// Number of triangular face loops.
    pub triangles: usize,
    /// Number of half-edges without a twin.
    pub boundary_halfedges: usize,
}

/// Topology construction or validation error.
#[derive(Clone, Debug, PartialEq)]
pub enum TopologyError {
    /// A face does not contain three distinct vertices.
    DegenerateTriangle(usize),
    /// The same directed edge appears more than once.
    DuplicateDirectedEdge(VertexId, VertexId),
    /// More or fewer than two faces use an undirected edge.
    NonManifoldEdge(VertexId, VertexId),
    /// The mesh has a boundary edge.
    BoundaryEdge(VertexId, VertexId),
    /// Half-edge links are inconsistent.
    BrokenHalfEdge(HalfEdgeId),
    /// A face id does not exist.
    InvalidFace(FaceId),
    /// A face is missing exactly one outer trim loop.
    MissingOuterTrimLoop(FaceId),
    /// A trim loop is malformed.
    InvalidTrimLoop(FaceId, usize),
    /// A trim references a missing or wrong-face half-edge.
    InvalidTrimHalfEdge(FaceId, HalfEdgeId),
    /// A trim p-curve is malformed.
    InvalidTrimCurve(FaceId, usize, usize),
    /// Consecutive p-curves do not close within tolerance.
    OpenTrimLoop(FaceId, usize),
}

impl Solid {
    /// Construct a closed half-edge solid from an indexed triangle mesh.
    pub fn from_triangle_mesh(
        points: Vec<Point3>,
        triangles: &[[usize; 3]],
    ) -> Result<Self, TopologyError> {
        let mut vertices: Vec<Vertex> = points
            .into_iter()
            .map(|point| Vertex {
                point,
                halfedge: None,
            })
            .collect();
        let mut halfedges = Vec::<HalfEdge>::new();
        let mut faces = Vec::<Face>::new();
        let mut directed = HashMap::<(usize, usize), HalfEdgeId>::new();

        for (face_id, tri) in triangles.iter().copied().enumerate() {
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
                return Err(TopologyError::DegenerateTriangle(face_id));
            }
            let base = halfedges.len();
            for i in 0..3 {
                let origin = tri[i];
                let dest = tri[(i + 1) % 3];
                if directed.insert((origin, dest), base + i).is_some() {
                    return Err(TopologyError::DuplicateDirectedEdge(origin, dest));
                }
                if vertices[origin].halfedge.is_none() {
                    vertices[origin].halfedge = Some(base + i);
                }
                halfedges.push(HalfEdge {
                    origin,
                    twin: None,
                    next: base + (i + 1) % 3,
                    prev: base + (i + 2) % 3,
                    face: face_id,
                    edge: usize::MAX,
                });
            }
            faces.push(Face {
                halfedge: base,
                surface: FaceSurface::Faceted,
                trim_loops: vec![TrimLoop::from_halfedges(
                    TrimLoopKind::Outer,
                    vec![base, base + 1, base + 2],
                )],
            });
        }

        for (&(origin, dest), &halfedge) in &directed {
            let Some(&twin) = directed.get(&(dest, origin)) else {
                return Err(TopologyError::BoundaryEdge(origin, dest));
            };
            halfedges[halfedge].twin = Some(twin);
        }

        let mut edges = Vec::<Edge>::new();
        for halfedge in 0..halfedges.len() {
            if halfedges[halfedge].edge != usize::MAX {
                continue;
            }
            let twin = halfedges[halfedge].twin.ok_or_else(|| {
                let dest = halfedges[halfedges[halfedge].next].origin;
                TopologyError::BoundaryEdge(halfedges[halfedge].origin, dest)
            })?;
            if halfedges[twin].edge != usize::MAX {
                return Err(TopologyError::NonManifoldEdge(
                    halfedges[halfedge].origin,
                    halfedges[halfedges[halfedge].next].origin,
                ));
            }
            let edge_id = edges.len();
            halfedges[halfedge].edge = edge_id;
            halfedges[twin].edge = edge_id;
            edges.push(Edge { halfedge });
        }

        let mut solid = Self {
            vertices,
            halfedges,
            edges,
            faces,
            shells: vec![Shell {
                faces: (0..triangles.len()).collect(),
            }],
        };
        solid.classify_planar_faces();
        solid.validate()?;
        Ok(solid)
    }

    /// Unit cube centered at the origin.
    pub fn cube(size: f64) -> Result<Self, TopologyError> {
        let h = size * 0.5;
        let v = vec![
            Point3::new(-h, -h, -h),
            Point3::new(h, -h, -h),
            Point3::new(h, h, -h),
            Point3::new(-h, h, -h),
            Point3::new(-h, -h, h),
            Point3::new(h, -h, h),
            Point3::new(h, h, h),
            Point3::new(-h, h, h),
        ];
        let t = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        Self::from_triangle_mesh(v, &t)
    }

    /// Validate internal half-edge adjacency.
    pub fn validate(&self) -> Result<(), TopologyError> {
        for (id, he) in self.halfedges.iter().enumerate() {
            if self.halfedges[he.next].prev != id || self.halfedges[he.prev].next != id {
                return Err(TopologyError::BrokenHalfEdge(id));
            }
            let Some(twin) = he.twin else {
                let dest = self.halfedges[he.next].origin;
                return Err(TopologyError::BoundaryEdge(he.origin, dest));
            };
            if self.halfedges[twin].twin != Some(id) {
                return Err(TopologyError::BrokenHalfEdge(id));
            }
        }
        self.validate_trim_topology()?;
        Ok(())
    }

    /// Validate per-face support surfaces and trim-loop topology.
    pub fn validate_trim_topology(&self) -> Result<(), TopologyError> {
        for (face_id, face) in self.faces.iter().enumerate() {
            if face.halfedge >= self.halfedges.len()
                || self.halfedges[face.halfedge].face != face_id
            {
                return Err(TopologyError::BrokenHalfEdge(face.halfedge));
            }
            let outer_count = face
                .trim_loops
                .iter()
                .filter(|trim_loop| trim_loop.kind == TrimLoopKind::Outer)
                .count();
            if outer_count != 1 {
                return Err(TopologyError::MissingOuterTrimLoop(face_id));
            }
            for (loop_index, trim_loop) in face.trim_loops.iter().enumerate() {
                self.validate_trim_loop(face_id, loop_index, trim_loop)?;
            }
        }
        Ok(())
    }

    /// Replace a face's analytic support surface and rebuild direct p-curves when possible.
    pub fn set_face_surface(
        &mut self,
        face: FaceId,
        surface: FaceSurface,
    ) -> Result<(), TopologyError> {
        if face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(face));
        }
        self.faces[face].surface = surface;
        self.rebuild_face_trim_curves(face);
        self.validate_trim_topology()
    }

    /// Replace a face's trim loops.
    pub fn set_face_trim_loops(
        &mut self,
        face: FaceId,
        trim_loops: Vec<TrimLoop>,
    ) -> Result<(), TopologyError> {
        if face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(face));
        }
        let old = core::mem::replace(&mut self.faces[face].trim_loops, trim_loops);
        if let Err(error) = self.validate_trim_topology() {
            self.faces[face].trim_loops = old;
            return Err(error);
        }
        Ok(())
    }

    /// Count trim loops by `(outer, inner)` role.
    pub fn trim_loop_counts(&self) -> (usize, usize) {
        let mut outer = 0;
        let mut inner = 0;
        for trim_loop in self.faces.iter().flat_map(|face| &face.trim_loops) {
            match trim_loop.kind {
                TrimLoopKind::Outer => outer += 1,
                TrimLoopKind::Inner => inner += 1,
            }
        }
        (outer, inner)
    }

    /// Euler characteristic `V - E + F`.
    pub fn euler_characteristic(&self) -> isize {
        self.vertices.len() as isize - self.edges.len() as isize + self.faces.len() as isize
    }

    /// Snapshot stable topology counts.
    pub fn topology_counts(&self) -> TopologyCounts {
        TopologyCounts {
            vertices: self.vertices.len(),
            edges: self.edges.len(),
            halfedges: self.halfedges.len(),
            faces: self.faces.len(),
            shells: self.shells.len(),
            triangles: self.faces.len(),
            boundary_halfedges: self.boundary_halfedge_count(),
        }
    }

    /// Genus estimate for a single closed orientable shell.
    pub fn genus(&self) -> Option<isize> {
        if self.shells.len() != 1 {
            return None;
        }
        let chi = self.euler_characteristic();
        if (2 - chi) % 2 == 0 {
            Some((2 - chi) / 2)
        } else {
            None
        }
    }

    /// Return the vertex ids around a triangular face.
    pub fn face_vertices(&self, face: FaceId) -> [VertexId; 3] {
        let a = self.faces[face].halfedge;
        let b = self.halfedges[a].next;
        let c = self.halfedges[b].next;
        [
            self.halfedges[a].origin,
            self.halfedges[b].origin,
            self.halfedges[c].origin,
        ]
    }

    /// Return all indexed triangles in face order.
    pub fn triangles(&self) -> Vec<[usize; 3]> {
        (0..self.faces.len())
            .map(|face| self.face_vertices(face))
            .collect()
    }

    /// Signed volume enclosed by the oriented boundary.
    pub fn signed_volume(&self) -> f64 {
        let mut volume = 0.0;
        for tri in self.triangles() {
            let a = self.vertices[tri[0]].point;
            let b = self.vertices[tri[1]].point;
            let c = self.vertices[tri[2]].point;
            volume += a.dot(b.cross(c)) / 6.0;
        }
        volume
    }

    /// Absolute enclosed volume.
    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Total triangle surface area.
    pub fn surface_area(&self) -> f64 {
        let mut area = 0.0;
        for tri in self.triangles() {
            let a = self.vertices[tri[0]].point;
            let b = self.vertices[tri[1]].point;
            let c = self.vertices[tri[2]].point;
            area += (b - a).cross(c - a).norm() * 0.5;
        }
        area
    }

    /// Stable FNV-1a hash over quantized vertex coordinates and face indices.
    ///
    /// Coordinates are rounded to a `1e-9` grid before hashing. The hash pins
    /// the emitted mesh, including vertex and face ordering, which is exactly
    /// what golden reference tests need to catch accidental output changes.
    pub fn stable_mesh_hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        hash = hash_u64(hash, self.vertices.len() as u64);
        for vertex in &self.vertices {
            hash = hash_i64(hash, quantized_coord(vertex.point.x));
            hash = hash_i64(hash, quantized_coord(vertex.point.y));
            hash = hash_i64(hash, quantized_coord(vertex.point.z));
        }
        let triangles = self.triangles();
        hash = hash_u64(hash, triangles.len() as u64);
        for tri in triangles {
            hash = hash_u64(hash, tri[0] as u64);
            hash = hash_u64(hash, tri[1] as u64);
            hash = hash_u64(hash, tri[2] as u64);
        }
        hash
    }

    /// Count half-edges that lack a twin.
    pub fn boundary_halfedge_count(&self) -> usize {
        self.halfedges.iter().filter(|he| he.twin.is_none()).count()
    }

    fn validate_trim_loop(
        &self,
        face_id: FaceId,
        loop_index: usize,
        trim_loop: &TrimLoop,
    ) -> Result<(), TopologyError> {
        if trim_loop.trims.is_empty() {
            return Err(TopologyError::InvalidTrimLoop(face_id, loop_index));
        }
        for (trim_index, trim) in trim_loop.trims.iter().enumerate() {
            if !trim.tolerance.is_finite() || trim.tolerance < 0.0 {
                return Err(TopologyError::InvalidTrimCurve(
                    face_id, loop_index, trim_index,
                ));
            }
            if !trim.curve.is_valid() {
                return Err(TopologyError::InvalidTrimCurve(
                    face_id, loop_index, trim_index,
                ));
            }
            if let Some(halfedge) = trim.halfedge {
                if halfedge >= self.halfedges.len() || self.halfedges[halfedge].face != face_id {
                    return Err(TopologyError::InvalidTrimHalfEdge(face_id, halfedge));
                }
                let next_trim = &trim_loop.trims[(trim_index + 1) % trim_loop.trims.len()];
                if let Some(next_halfedge) = next_trim.halfedge {
                    if self.halfedges[halfedge].next != next_halfedge {
                        return Err(TopologyError::InvalidTrimLoop(face_id, loop_index));
                    }
                }
            }
        }

        for trim_index in 0..trim_loop.trims.len() {
            let trim = &trim_loop.trims[trim_index];
            let next_trim = &trim_loop.trims[(trim_index + 1) % trim_loop.trims.len()];
            let Some((_, end)) = trim.curve.endpoints() else {
                continue;
            };
            let Some((next_start, _)) = next_trim.curve.endpoints() else {
                continue;
            };
            let tolerance = trim
                .tolerance
                .max(next_trim.tolerance)
                .max(DEFAULT_TRIM_TOLERANCE);
            if vec2_distance(end, next_start) > tolerance {
                return Err(TopologyError::OpenTrimLoop(face_id, loop_index));
            }
        }
        Ok(())
    }

    fn classify_planar_faces(&mut self) {
        for face_id in 0..self.faces.len() {
            let tri = self.face_vertices(face_id);
            let a = self.vertices[tri[0]].point;
            let b = self.vertices[tri[1]].point;
            let c = self.vertices[tri[2]].point;
            if let Some(plane) = Plane::from_points(a, b, c) {
                self.faces[face_id].surface = FaceSurface::Plane(plane);
                self.rebuild_face_trim_curves(face_id);
            }
        }
    }

    fn rebuild_face_trim_curves(&mut self, face_id: FaceId) {
        let surface = self.faces[face_id].surface.clone();
        for trim_loop in &mut self.faces[face_id].trim_loops {
            for trim in &mut trim_loop.trims {
                let Some(halfedge) = trim.halfedge else {
                    continue;
                };
                let start = self.vertices[self.halfedges[halfedge].origin].point;
                let end = self.vertices[self.halfedges[self.halfedges[halfedge].next].origin].point;
                trim.curve = if let (Some(start), Some(end)) =
                    (surface.project_point(start), surface.project_point(end))
                {
                    TrimCurve2D::LineSegment { start, end }
                } else {
                    TrimCurve2D::Unresolved
                };
                trim.tolerance = DEFAULT_TRIM_TOLERANCE;
            }
        }
    }
}

/// Compute the unit normal of an indexed triangle.
pub fn triangle_normal(points: &[Point3], tri: [usize; 3]) -> Vec3 {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    (b - a).cross(c - a).normalized()
}

const DEFAULT_TRIM_TOLERANCE: f64 = 1.0e-9;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const HASH_GRID: f64 = 1_000_000_000.0;

fn plane_frame(plane: Plane) -> (Vec3, Vec3) {
    let helper = if plane.normal.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = plane.normal.cross(helper).normalized();
    let v = plane.normal.cross(u).normalized();
    (u, v)
}

fn finite_vec2(value: Vec2) -> bool {
    value.x.is_finite() && value.y.is_finite()
}

fn vec2_distance(a: Vec2, b: Vec2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_i64(hash: u64, value: i64) -> u64 {
    hash_u64(hash, value as u64)
}

fn quantized_coord(value: f64) -> i64 {
    let rounded = (value * HASH_GRID).round();
    if rounded == 0.0 {
        0
    } else {
        rounded as i64
    }
}
