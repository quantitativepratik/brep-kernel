//! Half-edge topology for closed manifold B-reps.

use crate::geometry::{Circle, Cylinder, Plane};
use crate::math::{Point3, Vec2, Vec3};
use crate::nurbs::{NurbsCurve, NurbsSurface};
use crate::predicates::{orient2d, Interval, RobustSign};
use std::collections::HashMap;

/// Vertex identifier.
pub type VertexId = usize;
/// Half-edge identifier.
pub type HalfEdgeId = usize;
/// Edge identifier.
pub type EdgeId = usize;
/// Face identifier.
pub type FaceId = usize;
/// Split-edge identifier for staged face-splitting wires.
pub type SplitEdgeId = usize;

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
    /// Model-space curve supporting this edge.
    pub curve: EdgeCurve3D,
    /// Geometric tolerance for endpoint checks and later edge classification.
    pub tolerance: f64,
}

/// Shared model-space curve produced by a face split.
///
/// Split edges are deliberately staged outside the closed shell half-edge graph.
/// They carry the 3D intersection/split curve and are referenced by face-local
/// p-curves until a later healing step rewrites the affected trim loops.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitEdge {
    /// Model-space curve supporting the split.
    pub curve: EdgeCurve3D,
    /// Start point in model space.
    pub start: Point3,
    /// End point in model space.
    pub end: Point3,
    /// Geometric tolerance for endpoint matching and later classification.
    pub tolerance: f64,
}

/// Trim-loop winding in a face parameter domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimLoopOrientation {
    /// Positive signed area in UV space.
    CounterClockwise,
    /// Negative signed area in UV space.
    Clockwise,
    /// Certified zero area.
    Degenerate,
    /// The interval area test could not certify a sign.
    Uncertain,
}

/// Model-space curve carried by a topological edge.
#[derive(Clone, Debug, PartialEq)]
pub enum EdgeCurve3D {
    /// Straight segment from `start` to `end`.
    LineSegment {
        /// Start point in model space.
        start: Point3,
        /// End point in model space.
        end: Point3,
    },
    /// Circular arc in a 3D plane.
    CircularArc {
        /// Arc center.
        center: Point3,
        /// Unit normal for the circle plane.
        normal: Vec3,
        /// Arc radius.
        radius: f64,
        /// Start angle in radians in the circle's deterministic frame.
        start_angle: f64,
        /// End angle in radians in the circle's deterministic frame.
        end_angle: f64,
    },
    /// NURBS curve.
    Nurbs(Box<NurbsCurve>),
    /// Piecewise-linear model-space curve.
    Polyline {
        /// Ordered points.
        points: Vec<Point3>,
    },
    /// Topological edge exists, but its model-space curve has not been fitted yet.
    Unresolved,
}

impl EdgeCurve3D {
    /// Construct a straight segment.
    pub fn line_segment(start: Point3, end: Point3) -> Self {
        Self::LineSegment { start, end }
    }

    /// Return start and end points if the model-space curve has explicit endpoints.
    pub fn endpoints(&self) -> Option<(Point3, Point3)> {
        match self {
            Self::LineSegment { start, end } => Some((*start, *end)),
            Self::CircularArc {
                center,
                normal,
                radius,
                start_angle,
                end_angle,
            } => {
                let circle = Circle::new(*center, *normal, *radius);
                Some((circle.point_at(*start_angle), circle.point_at(*end_angle)))
            }
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                Some((curve.evaluate(u0), curve.evaluate(u1)))
            }
            Self::Polyline { points } => Some((*points.first()?, *points.last()?)),
            Self::Unresolved => None,
        }
    }

    /// Sample model-space points along the curve.
    pub fn sample_points(&self, samples: usize) -> Option<Vec<Point3>> {
        let samples = samples.max(2);
        match self {
            Self::LineSegment { start, end } => Some(
                (0..samples)
                    .map(|index| start.lerp(*end, normalized_index(index, samples)))
                    .collect(),
            ),
            Self::CircularArc {
                center,
                normal,
                radius,
                start_angle,
                end_angle,
            } => {
                let circle = Circle::new(*center, *normal, *radius);
                Some(
                    (0..samples)
                        .map(|index| {
                            let t = normalized_index(index, samples);
                            circle.point_at(start_angle * (1.0 - t) + end_angle * t)
                        })
                        .collect(),
                )
            }
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                Some(
                    (0..samples)
                        .map(|index| {
                            let t = normalized_index(index, samples);
                            curve.evaluate(u0 * (1.0 - t) + u1 * t)
                        })
                        .collect(),
                )
            }
            Self::Polyline { points } => Some(resample_polyline3(points, samples)?),
            Self::Unresolved => None,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::LineSegment { start, end } => finite_point3(*start) && finite_point3(*end),
            Self::CircularArc {
                center,
                normal,
                radius,
                start_angle,
                end_angle,
            } => {
                finite_point3(*center)
                    && finite_point3(*normal)
                    && normal.norm() > f64::EPSILON
                    && radius.is_finite()
                    && *radius > 0.0
                    && start_angle.is_finite()
                    && end_angle.is_finite()
            }
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                u0.is_finite()
                    && u1.is_finite()
                    && finite_point3(curve.evaluate(u0))
                    && finite_point3(curve.evaluate(u1))
            }
            Self::Polyline { points } => {
                points.len() >= 2 && points.iter().all(|point| finite_point3(*point))
            }
            Self::Unresolved => true,
        }
    }
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
    /// Evaluate a point from the surface parameter domain when supported.
    pub fn evaluate(&self, uv: Vec2) -> Option<Point3> {
        match self {
            Self::Plane(plane) => {
                let (u_axis, v_axis) = plane_frame(*plane);
                Some(plane.origin + u_axis * uv.x + v_axis * uv.y)
            }
            Self::Cylinder(cylinder) => Some(
                cylinder.center
                    + Vec3::new(
                        cylinder.radius * uv.x.cos(),
                        cylinder.radius * uv.x.sin(),
                        uv.y,
                    ),
            ),
            Self::Nurbs(surface) => Some(surface.evaluate(uv.x, uv.y)),
            Self::Faceted => None,
        }
    }

    /// Evaluate first partial derivatives `(du, dv)` when supported.
    pub fn partials(&self, uv: Vec2) -> Option<(Vec3, Vec3)> {
        match self {
            Self::Plane(plane) => Some(plane_frame(*plane)),
            Self::Cylinder(cylinder) => Some((
                Vec3::new(
                    -cylinder.radius * uv.x.sin(),
                    cylinder.radius * uv.x.cos(),
                    0.0,
                ),
                Vec3::new(0.0, 0.0, 1.0),
            )),
            Self::Nurbs(surface) => Some(surface.partials(uv.x, uv.y)),
            Self::Faceted => None,
        }
    }

    /// Evaluate an outward-oriented support normal when supported.
    pub fn normal_at(&self, uv: Vec2) -> Option<Vec3> {
        match self {
            Self::Plane(plane) => Some(plane.normal),
            Self::Cylinder(_) | Self::Nurbs(_) => {
                let (du, dv) = self.partials(uv)?;
                let normal = du.cross(dv).normalized();
                if normal.norm() <= f64::EPSILON {
                    None
                } else {
                    Some(normal)
                }
            }
            Self::Faceted => None,
        }
    }

    /// Project a model-space point into the surface parameter domain when a direct projection exists.
    ///
    /// Planes use a deterministic orthonormal frame. Cylinders use `(angle, height)`
    /// around the Z-aligned cylinder. NURBS surfaces use a small Newton inverse
    /// solve and return `None` when the closest point is outside tolerance.
    pub fn project_point(&self, point: Point3) -> Option<Vec2> {
        self.project_point_with_tolerance(point, DEFAULT_TRIM_TOLERANCE)
    }

    /// Project a model-space point into the surface parameter domain with a tolerance.
    pub fn project_point_with_tolerance(&self, point: Point3, tolerance: f64) -> Option<Vec2> {
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
            Self::Nurbs(surface) => inverse_project_nurbs(surface, point, tolerance),
            Self::Faceted => None,
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
    /// NURBS p-curve. The curve's `(x, y)` coordinates are face `(u, v)`.
    Nurbs(Box<NurbsCurve>),
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
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                let start = curve.evaluate(u0);
                let end = curve.evaluate(u1);
                Some((Vec2::new(start.x, start.y), Vec2::new(end.x, end.y)))
            }
            Self::Unresolved => None,
        }
    }

    /// Sample parameter-space points along the p-curve.
    pub fn sample_points(&self, samples: usize) -> Option<Vec<Vec2>> {
        let samples = samples.max(2);
        match self {
            Self::LineSegment { start, end } => Some(
                (0..samples)
                    .map(|index| {
                        *start * (1.0 - normalized_index(index, samples))
                            + *end * normalized_index(index, samples)
                    })
                    .collect(),
            ),
            Self::CircularArc {
                center,
                radius,
                start_angle,
                end_angle,
            } => Some(
                (0..samples)
                    .map(|index| {
                        let t = normalized_index(index, samples);
                        let angle = start_angle * (1.0 - t) + end_angle * t;
                        *center + Vec2::new(angle.cos() * *radius, angle.sin() * *radius)
                    })
                    .collect(),
            ),
            Self::Polyline { points } => Some(resample_polyline2(points, samples)?),
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                Some(
                    (0..samples)
                        .map(|index| {
                            let t = normalized_index(index, samples);
                            let point = curve.evaluate(u0 * (1.0 - t) + u1 * t);
                            Vec2::new(point.x, point.y)
                        })
                        .collect(),
                )
            }
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
            Self::Nurbs(curve) => {
                let (u0, u1) = curve.domain();
                let start = curve.evaluate(u0);
                let end = curve.evaluate(u1);
                u0.is_finite()
                    && u1.is_finite()
                    && finite_vec2(Vec2::new(start.x, start.y))
                    && finite_vec2(Vec2::new(end.x, end.y))
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

/// One face-local use of a staged split edge.
///
/// The `pcurve` is the same model-space split curve expressed in this face's
/// parameter domain. A robust Boolean pipeline can use these records as the
/// input to region classification and trim-loop healing.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceSplit {
    /// Shared model-space split edge.
    pub split_edge: SplitEdgeId,
    /// Face parameter-space curve for this split.
    pub pcurve: TrimCurve2D,
    /// Geometric tolerance for endpoint matching and later classification.
    pub tolerance: f64,
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
    /// Staged split curves installed on this face before trim-loop healing.
    pub split_curves: Vec<FaceSplit>,
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
    /// Staged split edges that are not yet part of the closed shell graph.
    pub split_edges: Vec<SplitEdge>,
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

/// One trim-loop record in a face nesting analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimLoopNesting {
    /// Loop index on the face.
    pub loop_index: usize,
    /// Outer or inner loop role.
    pub kind: TrimLoopKind,
    /// Robustly classified loop orientation.
    pub orientation: TrimLoopOrientation,
    /// Floating signed area in UV space for diagnostics.
    pub signed_area: f64,
    /// Parent loop containing this loop, if any.
    pub parent: Option<usize>,
    /// Nesting depth, where top-level loops have depth zero.
    pub depth: usize,
}

/// Trim-loop orientation and nesting analysis for one face.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimLoopAnalysis {
    /// Face that was analyzed.
    pub face: FaceId,
    /// Per-loop nesting records.
    pub loops: Vec<TrimLoopNesting>,
}

/// Result of installing one split curve on two faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitFacesReport {
    /// Shared staged split edge.
    pub split_edge: SplitEdgeId,
    /// First face receiving the split.
    pub a_face: FaceId,
    /// Second face receiving the split.
    pub b_face: FaceId,
    /// Index into `a_face.split_curves`.
    pub a_split: usize,
    /// Index into `b_face.split_curves`.
    pub b_split: usize,
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
    /// An edge id does not exist.
    InvalidEdge(EdgeId),
    /// An edge's model-space curve is malformed.
    InvalidEdgeCurve(EdgeId),
    /// An edge curve's endpoints do not match its topological vertices.
    EdgeCurveEndpointMismatch(EdgeId),
    /// A face id does not exist.
    InvalidFace(FaceId),
    /// A face split has no usable span or cannot be represented by this operation.
    DegenerateFaceSplit(FaceId),
    /// A split-edge id does not exist.
    InvalidSplitEdge(SplitEdgeId),
    /// A split-edge model-space curve is malformed.
    InvalidSplitCurve(SplitEdgeId),
    /// A split-edge curve's endpoints do not match the stored split endpoints.
    SplitCurveEndpointMismatch(SplitEdgeId),
    /// A face-local split-curve use is malformed.
    InvalidFaceSplit(FaceId, usize),
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
    /// Trim loop nesting is inconsistent.
    InvalidTrimLoopNesting(FaceId, usize),
    /// A p-curve could not be generated for an edge on a face.
    PcurveProjectionFailed(FaceId, HalfEdgeId),
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
                split_curves: Vec::new(),
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
            let start = vertices[halfedges[halfedge].origin].point;
            let end = vertices[halfedges[halfedges[halfedge].next].origin].point;
            edges.push(Edge {
                halfedge,
                curve: EdgeCurve3D::line_segment(start, end),
                tolerance: DEFAULT_EDGE_TOLERANCE,
            });
        }

        let mut solid = Self {
            vertices,
            halfedges,
            edges,
            split_edges: Vec::new(),
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
        self.validate_edge_curves()?;
        self.validate_trim_topology()?;
        self.validate_face_splits()?;
        Ok(())
    }

    /// Validate model-space curves attached to topological edges.
    pub fn validate_edge_curves(&self) -> Result<(), TopologyError> {
        for (edge_id, edge) in self.edges.iter().enumerate() {
            if edge.halfedge >= self.halfedges.len()
                || self.halfedges[edge.halfedge].edge != edge_id
            {
                return Err(TopologyError::InvalidEdge(edge_id));
            }
            if !edge.tolerance.is_finite() || edge.tolerance < 0.0 || !edge.curve.is_valid() {
                return Err(TopologyError::InvalidEdgeCurve(edge_id));
            }
            let Some((curve_start, curve_end)) = edge.curve.endpoints() else {
                continue;
            };
            let Some((origin, destination)) = self.edge_points(edge_id) else {
                return Err(TopologyError::InvalidEdge(edge_id));
            };
            let tolerance = edge.tolerance.max(DEFAULT_EDGE_TOLERANCE);
            let forward = curve_start.distance(origin) <= tolerance
                && curve_end.distance(destination) <= tolerance;
            let reverse = curve_start.distance(destination) <= tolerance
                && curve_end.distance(origin) <= tolerance;
            if !forward && !reverse {
                return Err(TopologyError::EdgeCurveEndpointMismatch(edge_id));
            }
        }
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

    /// Validate staged split edges and their face-local p-curves.
    pub fn validate_face_splits(&self) -> Result<(), TopologyError> {
        for (split_edge_id, split_edge) in self.split_edges.iter().enumerate() {
            if !split_edge.tolerance.is_finite()
                || split_edge.tolerance < 0.0
                || !finite_point3(split_edge.start)
                || !finite_point3(split_edge.end)
                || !split_edge.curve.is_valid()
            {
                return Err(TopologyError::InvalidSplitCurve(split_edge_id));
            }
            let Some((curve_start, curve_end)) = split_edge.curve.endpoints() else {
                return Err(TopologyError::InvalidSplitCurve(split_edge_id));
            };
            let tolerance = split_edge.tolerance.max(DEFAULT_EDGE_TOLERANCE);
            if !edge_curve_has_span(&split_edge.curve, tolerance) {
                return Err(TopologyError::InvalidSplitCurve(split_edge_id));
            }
            let forward = curve_start.distance(split_edge.start) <= tolerance
                && curve_end.distance(split_edge.end) <= tolerance;
            let reverse = curve_start.distance(split_edge.end) <= tolerance
                && curve_end.distance(split_edge.start) <= tolerance;
            if !forward && !reverse {
                return Err(TopologyError::SplitCurveEndpointMismatch(split_edge_id));
            }
        }

        let mut uses = vec![Vec::<FaceId>::new(); self.split_edges.len()];
        for (face_id, face) in self.faces.iter().enumerate() {
            for (split_index, split) in face.split_curves.iter().enumerate() {
                if split.split_edge >= self.split_edges.len() {
                    return Err(TopologyError::InvalidSplitEdge(split.split_edge));
                }
                if !split.tolerance.is_finite()
                    || split.tolerance < 0.0
                    || !split.pcurve.is_valid()
                    || split.pcurve.endpoints().is_none()
                {
                    return Err(TopologyError::InvalidFaceSplit(face_id, split_index));
                }
                uses[split.split_edge].push(face_id);
            }
        }
        for (split_edge_id, faces) in uses.iter().enumerate() {
            if faces.len() != 2 || faces[0] == faces[1] {
                return Err(TopologyError::InvalidSplitEdge(split_edge_id));
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

    /// Replace an edge's model-space support curve.
    pub fn set_edge_curve(
        &mut self,
        edge: EdgeId,
        curve: EdgeCurve3D,
        tolerance: f64,
    ) -> Result<(), TopologyError> {
        if edge >= self.edges.len() {
            return Err(TopologyError::InvalidEdge(edge));
        }
        let old_curve = core::mem::replace(&mut self.edges[edge].curve, curve);
        let old_tolerance = core::mem::replace(&mut self.edges[edge].tolerance, tolerance);
        if let Err(error) = self.validate_edge_curves() {
            self.edges[edge].curve = old_curve;
            self.edges[edge].tolerance = old_tolerance;
            return Err(error);
        }
        Ok(())
    }

    /// Replace the 2D p-curve for one face-side use of a topological edge.
    pub fn set_trim_curve(
        &mut self,
        face: FaceId,
        halfedge: HalfEdgeId,
        curve: TrimCurve2D,
        tolerance: f64,
    ) -> Result<(), TopologyError> {
        if face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(face));
        }
        if halfedge >= self.halfedges.len() || self.halfedges[halfedge].face != face {
            return Err(TopologyError::InvalidTrimHalfEdge(face, halfedge));
        }
        let Some((loop_index, trim_index)) = self.find_trim(face, halfedge) else {
            return Err(TopologyError::InvalidTrimHalfEdge(face, halfedge));
        };

        let old_curve = core::mem::replace(
            &mut self.faces[face].trim_loops[loop_index].trims[trim_index].curve,
            curve,
        );
        let old_tolerance = core::mem::replace(
            &mut self.faces[face].trim_loops[loop_index].trims[trim_index].tolerance,
            tolerance,
        );
        if let Err(error) = self.validate_trim_topology() {
            self.faces[face].trim_loops[loop_index].trims[trim_index].curve = old_curve;
            self.faces[face].trim_loops[loop_index].trims[trim_index].tolerance = old_tolerance;
            return Err(error);
        }
        Ok(())
    }

    /// Generate p-curves for every half-edge trim on a face from its 3D edge curve.
    ///
    /// This is intended for analytic faces, especially NURBS support surfaces.
    /// Edge curves are sampled in model space, inverse-projected into the face
    /// parameter domain, and fitted with a non-rational NURBS p-curve when more
    /// than two samples are requested.
    pub fn generate_face_pcurves(
        &mut self,
        face: FaceId,
        samples: usize,
        tolerance: f64,
    ) -> Result<(), TopologyError> {
        if face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(face));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(TopologyError::InvalidFace(face));
        }
        let samples = samples.max(2);
        let old_loops = self.faces[face].trim_loops.clone();
        let trim_refs: Vec<(usize, usize, HalfEdgeId)> = self.faces[face]
            .trim_loops
            .iter()
            .enumerate()
            .flat_map(|(loop_index, trim_loop)| {
                trim_loop
                    .trims
                    .iter()
                    .enumerate()
                    .filter_map(move |(trim_index, trim)| {
                        trim.halfedge
                            .map(|halfedge| (loop_index, trim_index, halfedge))
                    })
            })
            .collect();

        for (loop_index, trim_index, halfedge) in trim_refs {
            let Some(pcurve) = self.pcurve_for_halfedge(face, halfedge, samples, tolerance) else {
                self.faces[face].trim_loops = old_loops;
                return Err(TopologyError::PcurveProjectionFailed(face, halfedge));
            };
            self.faces[face].trim_loops[loop_index].trims[trim_index].curve = pcurve;
            self.faces[face].trim_loops[loop_index].trims[trim_index].tolerance = tolerance;
        }

        if let Err(error) = self.validate_trim_topology() {
            self.faces[face].trim_loops = old_loops;
            return Err(error);
        }
        Ok(())
    }

    /// Install a trim-ready split curve on two faces.
    ///
    /// This operation is the topological staging point after SSI and before
    /// Boolean region classification. It records a shared 3D split edge plus
    /// one p-curve per face, while leaving the closed shell half-edge graph and
    /// Euler counts unchanged.
    pub fn split_faces_with_curves(
        &mut self,
        a_face: FaceId,
        b_face: FaceId,
        edge_curve: EdgeCurve3D,
        a_pcurve: TrimCurve2D,
        b_pcurve: TrimCurve2D,
        tolerance: f64,
    ) -> Result<SplitFacesReport, TopologyError> {
        if a_face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(a_face));
        }
        if b_face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(b_face));
        }
        if a_face == b_face {
            return Err(TopologyError::DegenerateFaceSplit(a_face));
        }
        if !tolerance.is_finite() || tolerance < 0.0 || !edge_curve.is_valid() {
            return Err(TopologyError::InvalidSplitCurve(self.split_edges.len()));
        }
        let Some((start, end)) = edge_curve.endpoints() else {
            return Err(TopologyError::InvalidSplitCurve(self.split_edges.len()));
        };
        if !edge_curve_has_span(&edge_curve, tolerance.max(DEFAULT_EDGE_TOLERANCE)) {
            return Err(TopologyError::DegenerateFaceSplit(a_face));
        }
        if !a_pcurve.is_valid() || a_pcurve.endpoints().is_none() {
            return Err(TopologyError::InvalidFaceSplit(
                a_face,
                self.faces[a_face].split_curves.len(),
            ));
        }
        if !b_pcurve.is_valid() || b_pcurve.endpoints().is_none() {
            return Err(TopologyError::InvalidFaceSplit(
                b_face,
                self.faces[b_face].split_curves.len(),
            ));
        }

        let split_edge = self.split_edges.len();
        let a_split = self.faces[a_face].split_curves.len();
        let b_split = self.faces[b_face].split_curves.len();
        self.split_edges.push(SplitEdge {
            curve: edge_curve,
            start,
            end,
            tolerance,
        });
        self.faces[a_face].split_curves.push(FaceSplit {
            split_edge,
            pcurve: a_pcurve,
            tolerance,
        });
        self.faces[b_face].split_curves.push(FaceSplit {
            split_edge,
            pcurve: b_pcurve,
            tolerance,
        });

        if let Err(error) = self.validate_face_splits() {
            self.faces[b_face].split_curves.pop();
            self.faces[a_face].split_curves.pop();
            self.split_edges.pop();
            return Err(error);
        }

        Ok(SplitFacesReport {
            split_edge,
            a_face,
            b_face,
            a_split,
            b_split,
        })
    }

    /// Return the model-space endpoint vertex ids for an edge.
    pub fn edge_vertices(&self, edge: EdgeId) -> Option<(VertexId, VertexId)> {
        let halfedge = self.edges.get(edge)?.halfedge;
        Some((
            self.halfedges.get(halfedge)?.origin,
            self.halfedges
                .get(self.halfedges.get(halfedge)?.next)?
                .origin,
        ))
    }

    /// Return the model-space endpoint points for an edge.
    pub fn edge_points(&self, edge: EdgeId) -> Option<(Point3, Point3)> {
        let (origin, destination) = self.edge_vertices(edge)?;
        Some((
            self.vertices.get(origin)?.point,
            self.vertices.get(destination)?.point,
        ))
    }

    /// Return the p-curve for a face-side half-edge trim.
    pub fn trim_curve_for_halfedge(
        &self,
        face: FaceId,
        halfedge: HalfEdgeId,
    ) -> Option<&TrimCurve2D> {
        let (loop_index, trim_index) = self.find_trim(face, halfedge)?;
        Some(
            &self
                .faces
                .get(face)?
                .trim_loops
                .get(loop_index)?
                .trims
                .get(trim_index)?
                .curve,
        )
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

    /// Analyze trim-loop orientation and nesting on a face.
    pub fn analyze_trim_loop_nesting(
        &self,
        face: FaceId,
        tolerance: f64,
    ) -> Result<TrimLoopAnalysis, TopologyError> {
        if face >= self.faces.len() {
            return Err(TopologyError::InvalidFace(face));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(TopologyError::InvalidFace(face));
        }
        let face_record = &self.faces[face];
        let mut sampled = Vec::<Vec<Vec2>>::with_capacity(face_record.trim_loops.len());
        let mut records = Vec::<TrimLoopNesting>::with_capacity(face_record.trim_loops.len());
        for (loop_index, trim_loop) in face_record.trim_loops.iter().enumerate() {
            let Some(points) = trim_loop_sample_points(trim_loop, 16, tolerance) else {
                return Err(TopologyError::InvalidTrimLoop(face, loop_index));
            };
            if points.len() < 3 {
                return Err(TopologyError::InvalidTrimLoop(face, loop_index));
            }
            let signed_area = signed_loop_area(&points);
            let orientation = robust_loop_orientation(&points);
            sampled.push(points);
            records.push(TrimLoopNesting {
                loop_index,
                kind: trim_loop.kind,
                orientation,
                signed_area,
                parent: None,
                depth: 0,
            });
        }

        for child in 0..sampled.len() {
            let probe = interior_probe(&sampled[child]);
            let mut parent = None;
            let mut parent_area = f64::INFINITY;
            for candidate in 0..sampled.len() {
                if candidate == child {
                    continue;
                }
                let area = records[candidate].signed_area.abs();
                if area <= records[child].signed_area.abs() {
                    continue;
                }
                if matches!(
                    classify_point_in_loop(probe, &sampled[candidate], tolerance),
                    LoopPointLocation::Inside | LoopPointLocation::Boundary
                ) && area < parent_area
                {
                    parent = Some(candidate);
                    parent_area = area;
                }
            }
            records[child].parent = parent;
        }

        for index in 0..records.len() {
            records[index].depth = nesting_depth(index, &records);
        }

        Ok(TrimLoopAnalysis {
            face,
            loops: records,
        })
    }

    /// Validate that inner trim loops are nested inside the outer loop.
    pub fn validate_trim_loop_nesting(
        &self,
        face: FaceId,
        tolerance: f64,
    ) -> Result<(), TopologyError> {
        let analysis = self.analyze_trim_loop_nesting(face, tolerance)?;
        for record in &analysis.loops {
            match record.kind {
                TrimLoopKind::Outer => {
                    if record.parent.is_some() || record.depth != 0 {
                        return Err(TopologyError::InvalidTrimLoopNesting(
                            face,
                            record.loop_index,
                        ));
                    }
                }
                TrimLoopKind::Inner => {
                    let Some(parent) = record.parent else {
                        return Err(TopologyError::InvalidTrimLoopNesting(
                            face,
                            record.loop_index,
                        ));
                    };
                    if analysis.loops[parent].kind != TrimLoopKind::Outer {
                        return Err(TopologyError::InvalidTrimLoopNesting(
                            face,
                            record.loop_index,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Count staged face-split uses across all faces.
    pub fn face_split_count(&self) -> usize {
        self.faces.iter().map(|face| face.split_curves.len()).sum()
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

    fn find_trim(&self, face: FaceId, halfedge: HalfEdgeId) -> Option<(usize, usize)> {
        let face = self.faces.get(face)?;
        for (loop_index, trim_loop) in face.trim_loops.iter().enumerate() {
            for (trim_index, trim) in trim_loop.trims.iter().enumerate() {
                if trim.halfedge == Some(halfedge) {
                    return Some((loop_index, trim_index));
                }
            }
        }
        None
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

    fn pcurve_for_halfedge(
        &self,
        face: FaceId,
        halfedge: HalfEdgeId,
        samples: usize,
        tolerance: f64,
    ) -> Option<TrimCurve2D> {
        if self.halfedges.get(halfedge)?.face != face {
            return None;
        }
        let edge = self.edges.get(self.halfedges[halfedge].edge)?;
        let mut model_points = edge.curve.sample_points(samples)?;
        let start = self.vertices[self.halfedges[halfedge].origin].point;
        let end = self.vertices[self.halfedges[self.halfedges[halfedge].next].origin].point;
        let forward = model_points.first()?.distance(start) + model_points.last()?.distance(end);
        let reverse = model_points.first()?.distance(end) + model_points.last()?.distance(start);
        if reverse < forward {
            model_points.reverse();
        }

        let surface = &self.faces[face].surface;
        let mut uv_points = Vec::with_capacity(model_points.len());
        for point in model_points {
            let uv = surface.project_point_with_tolerance(point, tolerance)?;
            uv_points.push(uv);
        }
        pcurve_from_uv_samples(&uv_points, tolerance)
    }
}

/// Compute the unit normal of an indexed triangle.
pub fn triangle_normal(points: &[Point3], tri: [usize; 3]) -> Vec3 {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    (b - a).cross(c - a).normalized()
}

const DEFAULT_EDGE_TOLERANCE: f64 = 1.0e-9;
const DEFAULT_TRIM_TOLERANCE: f64 = 1.0e-9;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const HASH_GRID: f64 = 1_000_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopPointLocation {
    Inside,
    Boundary,
    Outside,
}

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

fn finite_point3(value: Point3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn normalized_index(index: usize, samples: usize) -> f64 {
    if samples <= 1 {
        0.0
    } else {
        index as f64 / (samples - 1) as f64
    }
}

fn inverse_project_nurbs(surface: &NurbsSurface, point: Point3, tolerance: f64) -> Option<Vec2> {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let mut best_uv = Vec2::new(u0, v0);
    let mut best_distance = f64::INFINITY;
    for j in 0..=4 {
        for i in 0..=4 {
            let u = lerp_scalar(u0, u1, i as f64 / 4.0);
            let v = lerp_scalar(v0, v1, j as f64 / 4.0);
            let distance = surface.evaluate(u, v).distance(point);
            if distance < best_distance {
                best_distance = distance;
                best_uv = Vec2::new(u, v);
            }
        }
    }

    let mut uv = best_uv;
    for _ in 0..32 {
        let surface_point = surface.evaluate(uv.x, uv.y);
        let residual = surface_point - point;
        if residual.norm() <= tolerance {
            return Some(uv);
        }
        let (du, dv) = surface.partials(uv.x, uv.y);
        let a00 = du.dot(du);
        let a01 = du.dot(dv);
        let a11 = dv.dot(dv);
        let b0 = -du.dot(residual);
        let b1 = -dv.dot(residual);
        let det = a00 * a11 - a01 * a01;
        if det.abs() <= 1.0e-18 {
            break;
        }
        let step_u = (b0 * a11 - b1 * a01) / det;
        let step_v = (a00 * b1 - a01 * b0) / det;
        uv.x = (uv.x + step_u).clamp(u0, u1);
        uv.y = (uv.y + step_v).clamp(v0, v1);
        if step_u.hypot(step_v) <= tolerance * 0.1 {
            break;
        }
    }

    if surface.evaluate(uv.x, uv.y).distance(point) <= tolerance {
        Some(uv)
    } else {
        None
    }
}

fn edge_curve_has_span(curve: &EdgeCurve3D, tolerance: f64) -> bool {
    match curve {
        EdgeCurve3D::LineSegment { start, end } => start.distance(*end) > tolerance,
        EdgeCurve3D::CircularArc {
            radius,
            start_angle,
            end_angle,
            ..
        } => *radius > tolerance && (end_angle - start_angle).abs() > tolerance,
        EdgeCurve3D::Nurbs(curve) => {
            let (u0, u1) = curve.domain();
            if (u1 - u0).abs() <= f64::EPSILON {
                return false;
            }
            let start = curve.evaluate(u0);
            let middle = curve.evaluate((u0 + u1) * 0.5);
            let end = curve.evaluate(u1);
            start.distance(middle) > tolerance || start.distance(end) > tolerance
        }
        EdgeCurve3D::Polyline { points } => points
            .windows(2)
            .any(|edge| edge[0].distance(edge[1]) > tolerance),
        EdgeCurve3D::Unresolved => false,
    }
}

fn pcurve_from_uv_samples(points: &[Vec2], tolerance: f64) -> Option<TrimCurve2D> {
    if points.len() < 2 || points.iter().any(|point| !finite_vec2(*point)) {
        return None;
    }
    if points.len() == 2 {
        return Some(TrimCurve2D::LineSegment {
            start: points[0],
            end: points[1],
        });
    }
    let fit_points: Vec<Point3> = points
        .iter()
        .map(|point| Point3::new(point.x, point.y, 0.0))
        .collect();
    let curve = NurbsCurve::interpolate(&fit_points, 3, tolerance).ok()?;
    Some(TrimCurve2D::Nurbs(Box::new(curve)))
}

fn trim_loop_sample_points(
    trim_loop: &TrimLoop,
    samples_per_trim: usize,
    tolerance: f64,
) -> Option<Vec<Vec2>> {
    let mut points = Vec::new();
    for trim in &trim_loop.trims {
        let samples = trim.curve.sample_points(samples_per_trim)?;
        for point in samples {
            if points
                .last()
                .is_none_or(|last| vec2_distance(*last, point) > tolerance)
            {
                points.push(point);
            }
        }
    }
    if points.len() >= 2 && vec2_distance(points[0], points[points.len() - 1]) <= tolerance {
        points.pop();
    }
    Some(points)
}

fn signed_loop_area(points: &[Vec2]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..points.len() {
        area += points[i].cross(points[(i + 1) % points.len()]);
    }
    area * 0.5
}

fn robust_loop_orientation(points: &[Vec2]) -> TrimLoopOrientation {
    if points.len() < 3 {
        return TrimLoopOrientation::Degenerate;
    }
    let mut sum = Interval::point(0.0);
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let cross = Interval::point(a.x)
            .mul_i(Interval::point(b.y))
            .sub_i(Interval::point(a.y).mul_i(Interval::point(b.x)));
        sum = sum.add_i(cross);
    }
    match sum.sign() {
        RobustSign::Positive => TrimLoopOrientation::CounterClockwise,
        RobustSign::Negative => TrimLoopOrientation::Clockwise,
        RobustSign::Zero => TrimLoopOrientation::Degenerate,
        RobustSign::Uncertain => TrimLoopOrientation::Uncertain,
    }
}

fn interior_probe(points: &[Vec2]) -> Vec2 {
    let mut centroid = Vec2::new(0.0, 0.0);
    for point in points {
        centroid = centroid + *point;
    }
    centroid * (1.0 / points.len() as f64)
}

fn classify_point_in_loop(point: Vec2, loop_points: &[Vec2], tolerance: f64) -> LoopPointLocation {
    if loop_points.len() < 3 {
        return LoopPointLocation::Outside;
    }
    let mut inside = false;
    for i in 0..loop_points.len() {
        let a = loop_points[i];
        let b = loop_points[(i + 1) % loop_points.len()];
        if point_segment_distance(point, a, b) <= tolerance || point_on_segment_robust(point, a, b)
        {
            return LoopPointLocation::Boundary;
        }
        let crosses_y = (a.y > point.y) != (b.y > point.y);
        if crosses_y {
            let x_at_y = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if point.x < x_at_y {
                inside = !inside;
            }
        }
    }
    if inside {
        LoopPointLocation::Inside
    } else {
        LoopPointLocation::Outside
    }
}

fn point_on_segment_robust(point: Vec2, a: Vec2, b: Vec2) -> bool {
    orient2d(a, b, point) == RobustSign::Zero
        && point.x >= a.x.min(b.x)
        && point.x <= a.x.max(b.x)
        && point.y >= a.y.min(b.y)
        && point.y <= a.y.max(b.y)
}

fn point_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f64 {
    let edge = b - a;
    let len2 = edge.dot(edge);
    if len2 <= f64::EPSILON {
        return vec2_distance(point, a);
    }
    let t = ((point - a).dot(edge) / len2).clamp(0.0, 1.0);
    vec2_distance(point, a + edge * t)
}

fn nesting_depth(index: usize, records: &[TrimLoopNesting]) -> usize {
    let mut depth = 0;
    let mut cursor = records[index].parent;
    while let Some(parent) = cursor {
        depth += 1;
        cursor = records[parent].parent;
    }
    depth
}

fn resample_polyline2(points: &[Vec2], samples: usize) -> Option<Vec<Vec2>> {
    let lengths = cumulative_lengths2(points)?;
    let total = *lengths.last()?;
    if total <= f64::EPSILON {
        return None;
    }
    Some(
        (0..samples)
            .map(|index| {
                interpolate_polyline2(points, &lengths, total * normalized_index(index, samples))
            })
            .collect(),
    )
}

fn resample_polyline3(points: &[Point3], samples: usize) -> Option<Vec<Point3>> {
    let lengths = cumulative_lengths3(points)?;
    let total = *lengths.last()?;
    if total <= f64::EPSILON {
        return None;
    }
    Some(
        (0..samples)
            .map(|index| {
                interpolate_polyline3(points, &lengths, total * normalized_index(index, samples))
            })
            .collect(),
    )
}

fn cumulative_lengths2(points: &[Vec2]) -> Option<Vec<f64>> {
    if points.len() < 2 {
        return None;
    }
    let mut lengths = Vec::with_capacity(points.len());
    lengths.push(0.0);
    for edge in points.windows(2) {
        lengths.push(lengths.last().copied()? + vec2_distance(edge[0], edge[1]));
    }
    Some(lengths)
}

fn cumulative_lengths3(points: &[Point3]) -> Option<Vec<f64>> {
    if points.len() < 2 {
        return None;
    }
    let mut lengths = Vec::with_capacity(points.len());
    lengths.push(0.0);
    for edge in points.windows(2) {
        lengths.push(lengths.last().copied()? + edge[0].distance(edge[1]));
    }
    Some(lengths)
}

fn interpolate_polyline2(points: &[Vec2], lengths: &[f64], target: f64) -> Vec2 {
    for i in 0..lengths.len() - 1 {
        if target <= lengths[i + 1] {
            let span = lengths[i + 1] - lengths[i];
            let t = if span <= f64::EPSILON {
                0.0
            } else {
                (target - lengths[i]) / span
            };
            return points[i] * (1.0 - t) + points[i + 1] * t;
        }
    }
    *points.last().expect("polyline has points")
}

fn interpolate_polyline3(points: &[Point3], lengths: &[f64], target: f64) -> Point3 {
    for i in 0..lengths.len() - 1 {
        if target <= lengths[i + 1] {
            let span = lengths[i + 1] - lengths[i];
            let t = if span <= f64::EPSILON {
                0.0
            } else {
                (target - lengths[i]) / span
            };
            return points[i].lerp(points[i + 1], t);
        }
    }
    *points.last().expect("polyline has points")
}

fn vec2_distance(a: Vec2, b: Vec2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn lerp_scalar(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
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
