//! Half-edge topology for closed manifold B-reps.

use crate::geometry::{Circle, Cylinder, Plane};
use crate::math::{Point3, Vec2, Vec3};
use crate::nurbs::{NurbsCurve, NurbsSurface};
use crate::predicates::{orient2d, Interval, RobustSign};
use std::collections::{HashMap, HashSet};

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
/// Persistent topology revision identifier.
pub type TopologyRevisionId = u64;

/// Kind tag carried by a persistent topological id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PersistentTopologyKind {
    /// Vertex identity.
    Vertex,
    /// Directed half-edge identity.
    HalfEdge,
    /// Undirected edge identity.
    Edge,
    /// Staged split-edge identity.
    SplitEdge,
    /// Face identity.
    Face,
    /// Shell identity.
    Shell,
}

/// Stable id for a topological entity across local topology edits.
///
/// `VertexId`, `EdgeId`, and `FaceId` are compact snapshot indices. A
/// `PersistentId` is the longer-lived identity that survives metadata edits,
/// trim edits, p-curve generation, and staged split creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PersistentId {
    /// Entity kind.
    pub kind: PersistentTopologyKind,
    /// Monotonic serial number within the owning solid history.
    pub serial: u64,
}

impl PersistentId {
    /// Construct a persistent id from a kind and serial.
    pub fn new(kind: PersistentTopologyKind, serial: u64) -> Self {
        Self { kind, serial }
    }
}

/// Topology operation recorded in a solid's history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyOperation {
    /// Solid was constructed from an indexed triangle mesh.
    ConstructTriangleMesh,
    /// Solid was constructed after tolerance-aware sewing.
    SewTriangleMesh,
    /// A face support surface was replaced.
    SetFaceSurface,
    /// Face trim loops were replaced.
    SetFaceTrimLoops,
    /// An edge's model-space curve was replaced.
    SetEdgeCurve,
    /// A face-side p-curve was replaced.
    SetTrimCurve,
    /// Face p-curves were generated from model-space edge curves.
    GenerateFacePcurves,
    /// A staged split edge was installed on two faces.
    SplitFacesWithCurves,
    /// A trim-ready intersection curve was promoted to face topology.
    InstallTrimReadyFaceCurve,
}

/// One persistent-history event for a topology mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyHistoryEvent {
    /// Revision after the event was applied.
    pub revision: TopologyRevisionId,
    /// Operation that produced the event.
    pub operation: TopologyOperation,
    /// New persistent entities created by the event.
    pub created: Vec<PersistentId>,
    /// Existing persistent entities modified by the event.
    pub modified: Vec<PersistentId>,
    /// Persistent entities retired by the event.
    pub deleted: Vec<PersistentId>,
    /// Source entities used to derive created or modified topology.
    pub parents: Vec<PersistentId>,
}

/// Persistent ids and mutation history for one solid snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyIdentity {
    vertices: Vec<PersistentId>,
    halfedges: Vec<PersistentId>,
    edges: Vec<PersistentId>,
    split_edges: Vec<PersistentId>,
    faces: Vec<PersistentId>,
    shells: Vec<PersistentId>,
    next_serial: u64,
    revision: TopologyRevisionId,
    history: Vec<TopologyHistoryEvent>,
}

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

/// Boundary side of a finite surface parameter domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceBoundary {
    /// Minimum U boundary.
    UMin,
    /// Maximum U boundary.
    UMax,
    /// Minimum V boundary.
    VMin,
    /// Maximum V boundary.
    VMax,
}

/// Parameter-space metadata for a support surface.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceParameterization {
    /// Finite U domain, when the support has one.
    pub u_domain: Option<(f64, f64)>,
    /// Finite V domain, when the support has one.
    pub v_domain: Option<(f64, f64)>,
    /// U period for periodic surfaces.
    pub u_period: Option<f64>,
    /// V period for periodic surfaces.
    pub v_period: Option<f64>,
    /// Collapsed parameter-domain boundaries.
    pub singular_boundaries: Vec<SurfaceBoundary>,
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
    /// Report finite domains, periodic directions, and singular boundaries.
    pub fn parameterization(&self, tolerance: f64) -> SurfaceParameterization {
        match self {
            Self::Plane(_) | Self::Faceted => SurfaceParameterization {
                u_domain: None,
                v_domain: None,
                u_period: None,
                v_period: None,
                singular_boundaries: Vec::new(),
            },
            Self::Cylinder(_) => SurfaceParameterization {
                u_domain: Some((-core::f64::consts::PI, core::f64::consts::PI)),
                v_domain: None,
                u_period: Some(TWO_PI),
                v_period: None,
                singular_boundaries: Vec::new(),
            },
            Self::Nurbs(surface) => {
                let ((u0, u1), (v0, v1)) = surface.domain();
                let u_period = if nurbs_boundary_matches(
                    surface,
                    SurfaceBoundary::UMin,
                    SurfaceBoundary::UMax,
                    tolerance,
                ) {
                    Some(u1 - u0)
                } else {
                    None
                };
                let v_period = if nurbs_boundary_matches(
                    surface,
                    SurfaceBoundary::VMin,
                    SurfaceBoundary::VMax,
                    tolerance,
                ) {
                    Some(v1 - v0)
                } else {
                    None
                };
                SurfaceParameterization {
                    u_domain: Some((u0, u1)),
                    v_domain: Some((v0, v1)),
                    u_period,
                    v_period,
                    singular_boundaries: nurbs_singular_boundaries(surface, tolerance),
                }
            }
        }
    }

    /// Return the period for each parameter direction.
    pub fn parameter_periods(&self, tolerance: f64) -> (Option<f64>, Option<f64>) {
        let parameterization = self.parameterization(tolerance);
        (parameterization.u_period, parameterization.v_period)
    }

    /// Return collapsed parameter-domain boundaries.
    pub fn singular_boundaries(&self, tolerance: f64) -> Vec<SurfaceBoundary> {
        self.parameterization(tolerance).singular_boundaries
    }

    /// Detect whether a UV location is singular for this support surface.
    pub fn is_singular_at(&self, uv: Vec2, tolerance: f64) -> bool {
        match self {
            Self::Plane(_) | Self::Cylinder(_) | Self::Faceted => false,
            Self::Nurbs(_) => {
                let Some((du, dv)) = self.partials(uv) else {
                    return false;
                };
                du.cross(dv).norm() <= tolerance
                    || singular_boundary_at(self, uv, tolerance).is_some()
            }
        }
    }

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
            Self::Nurbs(surface) => {
                let uv = constrain_nurbs_uv(surface, uv, DEFAULT_TRIM_TOLERANCE);
                Some(surface.evaluate(uv.x, uv.y))
            }
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
            Self::Nurbs(surface) => {
                let uv = constrain_nurbs_uv(surface, uv, DEFAULT_TRIM_TOLERANCE);
                Some(surface.partials(uv.x, uv.y))
            }
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
        self.project_point_near(point, None, tolerance)
    }

    /// Project a model-space point using an optional nearby UV seed.
    ///
    /// The seed is ignored for direct analytic projections, but NURBS surfaces use
    /// it as the first Newton candidate. This keeps a projected curve continuous
    /// in parameter space instead of re-solving every sample from a global grid.
    pub fn project_point_near(
        &self,
        point: Point3,
        seed: Option<Vec2>,
        tolerance: f64,
    ) -> Option<Vec2> {
        match self {
            Self::Plane(plane) => {
                let (u_axis, v_axis) = plane_frame(*plane);
                let d = point - plane.origin;
                Some(Vec2::new(d.dot(u_axis), d.dot(v_axis)))
            }
            Self::Cylinder(cylinder) => {
                let d = point - cylinder.center;
                let angle = unwrap_periodic_value(d.y.atan2(d.x), seed.map(|uv| uv.x), TWO_PI);
                Some(Vec2::new(angle, d.z))
            }
            Self::Nurbs(surface) => {
                inverse_project_nurbs_with_seed(surface, point, seed, tolerance)
            }
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
    /// Persistent ids and mutation history for this topology snapshot.
    pub identity: TopologyIdentity,
}

impl TopologyIdentity {
    /// Construct identity records for a new triangle-mesh solid.
    pub fn from_counts(
        vertices: usize,
        halfedges: usize,
        edges: usize,
        split_edges: usize,
        faces: usize,
        shells: usize,
    ) -> Self {
        let mut identity = Self {
            vertices: Vec::new(),
            halfedges: Vec::new(),
            edges: Vec::new(),
            split_edges: Vec::new(),
            faces: Vec::new(),
            shells: Vec::new(),
            next_serial: 1,
            revision: 0,
            history: Vec::new(),
        };
        identity.vertices = identity.allocate_many(PersistentTopologyKind::Vertex, vertices);
        identity.halfedges = identity.allocate_many(PersistentTopologyKind::HalfEdge, halfedges);
        identity.edges = identity.allocate_many(PersistentTopologyKind::Edge, edges);
        identity.split_edges =
            identity.allocate_many(PersistentTopologyKind::SplitEdge, split_edges);
        identity.faces = identity.allocate_many(PersistentTopologyKind::Face, faces);
        identity.shells = identity.allocate_many(PersistentTopologyKind::Shell, shells);
        let created = identity.all_ids();
        identity.record(
            TopologyOperation::ConstructTriangleMesh,
            created,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        identity
    }

    /// Current topology revision.
    pub fn revision(&self) -> TopologyRevisionId {
        self.revision
    }

    /// Persistent history events in revision order.
    pub fn history(&self) -> &[TopologyHistoryEvent] {
        &self.history
    }

    /// Persistent vertex id for a snapshot vertex index.
    pub fn vertex(&self, vertex: VertexId) -> Option<PersistentId> {
        self.vertices.get(vertex).copied()
    }

    /// Persistent half-edge id for a snapshot half-edge index.
    pub fn halfedge(&self, halfedge: HalfEdgeId) -> Option<PersistentId> {
        self.halfedges.get(halfedge).copied()
    }

    /// Persistent edge id for a snapshot edge index.
    pub fn edge(&self, edge: EdgeId) -> Option<PersistentId> {
        self.edges.get(edge).copied()
    }

    /// Persistent staged split-edge id for a snapshot split-edge index.
    pub fn split_edge(&self, split_edge: SplitEdgeId) -> Option<PersistentId> {
        self.split_edges.get(split_edge).copied()
    }

    /// Persistent face id for a snapshot face index.
    pub fn face(&self, face: FaceId) -> Option<PersistentId> {
        self.faces.get(face).copied()
    }

    /// Persistent shell id for a snapshot shell index.
    pub fn shell(&self, shell: usize) -> Option<PersistentId> {
        self.shells.get(shell).copied()
    }

    /// Return every live persistent id in deterministic storage order.
    pub fn all_ids(&self) -> Vec<PersistentId> {
        self.vertices
            .iter()
            .chain(&self.halfedges)
            .chain(&self.edges)
            .chain(&self.split_edges)
            .chain(&self.faces)
            .chain(&self.shells)
            .copied()
            .collect()
    }

    fn allocate_many(&mut self, kind: PersistentTopologyKind, count: usize) -> Vec<PersistentId> {
        (0..count).map(|_| self.allocate(kind)).collect()
    }

    fn allocate(&mut self, kind: PersistentTopologyKind) -> PersistentId {
        let id = PersistentId::new(kind, self.next_serial);
        self.next_serial += 1;
        id
    }

    fn allocate_split_edge(&mut self) -> PersistentId {
        let id = self.allocate(PersistentTopologyKind::SplitEdge);
        self.split_edges.push(id);
        id
    }

    fn record(
        &mut self,
        operation: TopologyOperation,
        created: Vec<PersistentId>,
        modified: Vec<PersistentId>,
        deleted: Vec<PersistentId>,
        parents: Vec<PersistentId>,
    ) {
        self.revision += 1;
        self.history.push(TopologyHistoryEvent {
            revision: self.revision,
            operation,
            created,
            modified,
            deleted,
            parents,
        });
    }
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

/// Deterministic report from tolerance-aware mesh sewing.
#[derive(Clone, Debug, PartialEq)]
pub struct SewingReport {
    /// Sewing tolerance in model units.
    pub tolerance: f64,
    /// Number of input vertices.
    pub input_vertices: usize,
    /// Number of input triangles.
    pub input_triangles: usize,
    /// Number of vertices after tolerance clustering.
    pub output_vertices: usize,
    /// Number of triangles after removing collapsed faces.
    pub output_triangles: usize,
    /// Number of vertices merged into earlier tolerance clusters.
    pub merged_vertices: usize,
    /// Number of triangles removed because sewing collapsed at least two corners together.
    pub removed_degenerate_triangles: usize,
    /// Map from each input vertex index to the sewn output vertex index.
    pub vertex_map: Vec<VertexId>,
}

/// Triangle mesh after tolerance-aware sewing.
#[derive(Clone, Debug, PartialEq)]
pub struct SewnTriangleMesh {
    /// Sewn model-space vertices.
    pub vertices: Vec<Point3>,
    /// Triangles rewritten through the sewn vertex map.
    pub triangles: Vec<[usize; 3]>,
    /// Deterministic sewing diagnostics.
    pub report: SewingReport,
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

/// How a trim-ready SSI curve was promoted into face topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimReadyFaceConversionKind {
    /// Open curve installed as a staged face split.
    OpenSplit,
    /// Closed curve installed as inner trim loops on both faces.
    ClosedInnerLoops,
}

/// Result of promoting one trim-ready SSI curve into face trim topology.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimReadyFaceConversionReport {
    /// Conversion mode.
    pub kind: TrimReadyFaceConversionKind,
    /// Staged split report for open curves.
    pub split: Option<SplitFacesReport>,
    /// Installed or matched loop index on the first face for closed curves.
    pub a_loop: Option<usize>,
    /// Installed or matched loop index on the second face for closed curves.
    pub b_loop: Option<usize>,
    /// Number of p-curve endpoints snapped to existing trim boundaries.
    pub snapped_pcurve_endpoints: usize,
    /// True when an existing equivalent split/loop was reused.
    pub merged_existing: bool,
}

/// Topology construction or validation error.
#[derive(Clone, Debug, PartialEq)]
pub enum TopologyError {
    /// A vertex id or vertex coordinate is invalid.
    InvalidVertex(VertexId),
    /// A face does not contain three distinct vertices.
    DegenerateTriangle(usize),
    /// Sewing tolerance is not finite and nonnegative.
    InvalidSewingTolerance,
    /// Sewing removed every triangle or left no usable shell candidate.
    DegenerateSewnMesh,
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
    /// Persistent topology ids do not match the current topology arrays.
    InvalidPersistentIdentity,
}

impl Solid {
    /// Sew a triangle mesh by clustering vertices within `tolerance`.
    ///
    /// The returned triangles are rewritten through the sewn vertex map.
    /// Triangles that collapse to fewer than three distinct sewn vertices are
    /// removed and counted in the report. This does not require the result to
    /// be a closed manifold; use [`Solid::from_triangle_mesh_sewn`] when a
    /// validated B-rep shell is required.
    pub fn sew_triangle_mesh(
        points: Vec<Point3>,
        triangles: &[[usize; 3]],
        tolerance: f64,
    ) -> Result<SewnTriangleMesh, TopologyError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(TopologyError::InvalidSewingTolerance);
        }
        for (index, point) in points.iter().enumerate() {
            if !finite_point3(*point) {
                return Err(TopologyError::InvalidVertex(index));
            }
        }

        let mut sets = DisjointSets::new(points.len());
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if points[i].distance(points[j]) <= tolerance {
                    sets.union(i, j);
                }
            }
        }

        let mut root_to_output = HashMap::<usize, usize>::new();
        let mut vertex_map = vec![0; points.len()];
        let mut sums = Vec::<Point3>::new();
        let mut counts = Vec::<usize>::new();
        for (index, point) in points.iter().copied().enumerate() {
            let root = sets.find(index);
            let output = *root_to_output.entry(root).or_insert_with(|| {
                sums.push(Point3::ZERO);
                counts.push(0);
                sums.len() - 1
            });
            vertex_map[index] = output;
            sums[output] += point;
            counts[output] += 1;
        }

        let sewn_points: Vec<Point3> = sums
            .into_iter()
            .zip(counts.iter().copied())
            .map(|(sum, count)| sum / count as f64)
            .collect();

        let mut sewn_triangles = Vec::<[usize; 3]>::new();
        let mut removed_degenerate_triangles = 0;
        for tri in triangles.iter().copied() {
            for vertex in tri {
                if vertex >= vertex_map.len() {
                    return Err(TopologyError::InvalidVertex(vertex));
                }
            }
            let sewn = [vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]];
            if sewn[0] == sewn[1] || sewn[1] == sewn[2] || sewn[2] == sewn[0] {
                removed_degenerate_triangles += 1;
            } else {
                sewn_triangles.push(sewn);
            }
        }

        let report = SewingReport {
            tolerance,
            input_vertices: points.len(),
            input_triangles: triangles.len(),
            output_vertices: sewn_points.len(),
            output_triangles: sewn_triangles.len(),
            merged_vertices: points.len().saturating_sub(sewn_points.len()),
            removed_degenerate_triangles,
            vertex_map,
        };
        Ok(SewnTriangleMesh {
            vertices: sewn_points,
            triangles: sewn_triangles,
            report,
        })
    }

    /// Sew a triangle mesh with tolerance and validate the result as a B-rep solid.
    pub fn from_triangle_mesh_sewn(
        points: Vec<Point3>,
        triangles: &[[usize; 3]],
        tolerance: f64,
    ) -> Result<(Self, SewingReport), TopologyError> {
        let sewn = Self::sew_triangle_mesh(points, triangles, tolerance)?;
        if sewn.triangles.is_empty() {
            return Err(TopologyError::DegenerateSewnMesh);
        }
        let mut solid = Self::from_triangle_mesh(sewn.vertices, &sewn.triangles)?;
        let edge_tolerance = tolerance.max(DEFAULT_EDGE_TOLERANCE);
        for edge in &mut solid.edges {
            edge.tolerance = edge_tolerance;
        }
        solid.validate()?;
        let modified = solid.identity.all_ids();
        solid.identity.record(
            TopologyOperation::SewTriangleMesh,
            Vec::new(),
            modified,
            Vec::new(),
            Vec::new(),
        );
        Ok((solid, sewn.report))
    }

    /// Construct a closed half-edge solid from an indexed triangle mesh.
    pub fn from_triangle_mesh(
        points: Vec<Point3>,
        triangles: &[[usize; 3]],
    ) -> Result<Self, TopologyError> {
        let point_count = points.len();
        let triangle_count = triangles.len();
        let mut vertices: Vec<Vertex> = points
            .into_iter()
            .map(|point| Vertex {
                point,
                halfedge: None,
            })
            .collect();
        for (index, vertex) in vertices.iter().enumerate() {
            if !finite_point3(vertex.point) {
                return Err(TopologyError::InvalidVertex(index));
            }
        }
        let mut halfedges = Vec::<HalfEdge>::new();
        let mut faces = Vec::<Face>::new();
        let mut directed = HashMap::<(usize, usize), HalfEdgeId>::new();

        for (face_id, tri) in triangles.iter().copied().enumerate() {
            for vertex in tri {
                if vertex >= vertices.len() {
                    return Err(TopologyError::InvalidVertex(vertex));
                }
            }
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
            identity: TopologyIdentity::from_counts(
                point_count,
                triangle_count * 3,
                directed.len() / 2,
                0,
                triangle_count,
                1,
            ),
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
        self.validate_persistent_identity()?;
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

    /// Validate persistent topological ids and history bookkeeping.
    pub fn validate_persistent_identity(&self) -> Result<(), TopologyError> {
        if self.identity.vertices.len() != self.vertices.len()
            || self.identity.halfedges.len() != self.halfedges.len()
            || self.identity.edges.len() != self.edges.len()
            || self.identity.split_edges.len() != self.split_edges.len()
            || self.identity.faces.len() != self.faces.len()
            || self.identity.shells.len() != self.shells.len()
        {
            return Err(TopologyError::InvalidPersistentIdentity);
        }

        let mut seen = HashSet::<PersistentId>::new();
        validate_persistent_id_slice(
            &self.identity.vertices,
            PersistentTopologyKind::Vertex,
            self.identity.next_serial,
            &mut seen,
        )?;
        validate_persistent_id_slice(
            &self.identity.halfedges,
            PersistentTopologyKind::HalfEdge,
            self.identity.next_serial,
            &mut seen,
        )?;
        validate_persistent_id_slice(
            &self.identity.edges,
            PersistentTopologyKind::Edge,
            self.identity.next_serial,
            &mut seen,
        )?;
        validate_persistent_id_slice(
            &self.identity.split_edges,
            PersistentTopologyKind::SplitEdge,
            self.identity.next_serial,
            &mut seen,
        )?;
        validate_persistent_id_slice(
            &self.identity.faces,
            PersistentTopologyKind::Face,
            self.identity.next_serial,
            &mut seen,
        )?;
        validate_persistent_id_slice(
            &self.identity.shells,
            PersistentTopologyKind::Shell,
            self.identity.next_serial,
            &mut seen,
        )?;

        let mut expected_revision = 1;
        for event in &self.identity.history {
            if event.revision != expected_revision {
                return Err(TopologyError::InvalidPersistentIdentity);
            }
            for id in event
                .created
                .iter()
                .chain(&event.modified)
                .chain(&event.deleted)
                .chain(&event.parents)
            {
                if id.serial == 0 || id.serial >= self.identity.next_serial {
                    return Err(TopologyError::InvalidPersistentIdentity);
                }
            }
            expected_revision += 1;
        }
        if self.identity.revision + 1 != expected_revision {
            return Err(TopologyError::InvalidPersistentIdentity);
        }
        Ok(())
    }

    /// Borrow the persistent identity table for this solid.
    pub fn topology_identity(&self) -> &TopologyIdentity {
        &self.identity
    }

    /// Current persistent topology revision.
    pub fn topology_revision(&self) -> TopologyRevisionId {
        self.identity.revision()
    }

    /// Persistent topology history events in revision order.
    pub fn topology_history(&self) -> &[TopologyHistoryEvent] {
        self.identity.history()
    }

    /// Persistent vertex id for a snapshot vertex index.
    pub fn persistent_vertex_id(&self, vertex: VertexId) -> Option<PersistentId> {
        self.identity.vertex(vertex)
    }

    /// Persistent half-edge id for a snapshot half-edge index.
    pub fn persistent_halfedge_id(&self, halfedge: HalfEdgeId) -> Option<PersistentId> {
        self.identity.halfedge(halfedge)
    }

    /// Persistent edge id for a snapshot edge index.
    pub fn persistent_edge_id(&self, edge: EdgeId) -> Option<PersistentId> {
        self.identity.edge(edge)
    }

    /// Persistent staged split-edge id for a snapshot split-edge index.
    pub fn persistent_split_edge_id(&self, split_edge: SplitEdgeId) -> Option<PersistentId> {
        self.identity.split_edge(split_edge)
    }

    /// Persistent face id for a snapshot face index.
    pub fn persistent_face_id(&self, face: FaceId) -> Option<PersistentId> {
        self.identity.face(face)
    }

    /// Persistent shell id for a snapshot shell index.
    pub fn persistent_shell_id(&self, shell: usize) -> Option<PersistentId> {
        self.identity.shell(shell)
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
        let modified = vec![self
            .persistent_face_id(face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?];
        let old_face = self.faces[face].clone();
        self.faces[face].surface = surface;
        self.rebuild_face_trim_curves(face);
        if let Err(error) = self.validate_trim_topology() {
            self.faces[face] = old_face;
            return Err(error);
        }
        self.identity.record(
            TopologyOperation::SetFaceSurface,
            Vec::new(),
            modified.clone(),
            Vec::new(),
            modified,
        );
        Ok(())
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
        let modified = vec![self
            .persistent_face_id(face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?];
        let old = core::mem::replace(&mut self.faces[face].trim_loops, trim_loops);
        if let Err(error) = self.validate_trim_topology() {
            self.faces[face].trim_loops = old;
            return Err(error);
        }
        self.identity.record(
            TopologyOperation::SetFaceTrimLoops,
            Vec::new(),
            modified.clone(),
            Vec::new(),
            modified,
        );
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
        let modified = vec![self
            .persistent_edge_id(edge)
            .ok_or(TopologyError::InvalidPersistentIdentity)?];
        let old_curve = core::mem::replace(&mut self.edges[edge].curve, curve);
        let old_tolerance = core::mem::replace(&mut self.edges[edge].tolerance, tolerance);
        if let Err(error) = self.validate_edge_curves() {
            self.edges[edge].curve = old_curve;
            self.edges[edge].tolerance = old_tolerance;
            return Err(error);
        }
        self.identity.record(
            TopologyOperation::SetEdgeCurve,
            Vec::new(),
            modified.clone(),
            Vec::new(),
            modified,
        );
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
        let modified = vec![self
            .persistent_face_id(face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?];
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
        self.identity.record(
            TopologyOperation::SetTrimCurve,
            Vec::new(),
            modified.clone(),
            Vec::new(),
            modified,
        );
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
        let modified = vec![self
            .persistent_face_id(face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?];
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
        self.normalize_face_trim_loop_parameters(face, tolerance);

        if let Err(error) = self.validate_trim_topology() {
            self.faces[face].trim_loops = old_loops;
            return Err(error);
        }
        self.identity.record(
            TopologyOperation::GenerateFacePcurves,
            Vec::new(),
            modified.clone(),
            Vec::new(),
            modified,
        );
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

        let old_identity = self.identity.clone();
        let a_face_id = self
            .persistent_face_id(a_face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?;
        let b_face_id = self
            .persistent_face_id(b_face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?;
        let split_persistent_id = self.identity.allocate_split_edge();
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
            self.identity = old_identity;
            return Err(error);
        }
        self.identity.record(
            TopologyOperation::SplitFacesWithCurves,
            vec![split_persistent_id],
            vec![a_face_id, b_face_id],
            Vec::new(),
            vec![a_face_id, b_face_id],
        );

        Ok(SplitFacesReport {
            split_edge,
            a_face,
            b_face,
            a_split,
            b_split,
        })
    }

    /// Promote a trim-ready SSI curve into face trim topology.
    ///
    /// Closed SSI curves become inner trim loops on both faces. Open curves are
    /// gap-closed against nearby face boundaries and installed as staged face
    /// splits for later Boolean classification. Existing equivalent closed loops
    /// or open split edges are reused instead of duplicated.
    pub fn install_trim_ready_face_curve(
        &mut self,
        a_face: FaceId,
        b_face: FaceId,
        edge_curve: EdgeCurve3D,
        a_pcurve: TrimCurve2D,
        b_pcurve: TrimCurve2D,
        tolerance: f64,
    ) -> Result<TrimReadyFaceConversionReport, TopologyError> {
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

        let old_split_edges = self.split_edges.clone();
        let old_a_splits = self.faces[a_face].split_curves.clone();
        let old_b_splits = self.faces[b_face].split_curves.clone();
        let old_a_loops = self.faces[a_face].trim_loops.clone();
        let old_b_loops = self.faces[b_face].trim_loops.clone();
        let a_face_id = self
            .persistent_face_id(a_face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?;
        let b_face_id = self
            .persistent_face_id(b_face)
            .ok_or(TopologyError::InvalidPersistentIdentity)?;

        let (mut a_pcurve, a_snaps) =
            self.close_pcurve_gaps_to_face_boundary(a_face, a_pcurve, tolerance);
        let (mut b_pcurve, b_snaps) =
            self.close_pcurve_gaps_to_face_boundary(b_face, b_pcurve, tolerance);
        let snapped_pcurve_endpoints = a_snaps + b_snaps;

        if trim_curve_is_closed_on_surface(&self.faces[a_face].surface, &a_pcurve, tolerance)
            && trim_curve_is_closed_on_surface(&self.faces[b_face].surface, &b_pcurve, tolerance)
            && edge_curve.endpoints().is_some_and(|(start, end)| {
                start.distance(end) <= tolerance.max(DEFAULT_EDGE_TOLERANCE)
            })
        {
            a_pcurve = close_trim_curve(a_pcurve);
            b_pcurve = close_trim_curve(b_pcurve);

            let existing_a = self.find_matching_inner_trim_loop(a_face, &a_pcurve, tolerance);
            let existing_b = self.find_matching_inner_trim_loop(b_face, &b_pcurve, tolerance);
            if let (Some(a_loop), Some(b_loop)) = (existing_a, existing_b) {
                return Ok(TrimReadyFaceConversionReport {
                    kind: TrimReadyFaceConversionKind::ClosedInnerLoops,
                    split: None,
                    a_loop: Some(a_loop),
                    b_loop: Some(b_loop),
                    snapped_pcurve_endpoints,
                    merged_existing: true,
                });
            }

            let a_loop = self.faces[a_face].trim_loops.len();
            let b_loop = self.faces[b_face].trim_loops.len();
            self.faces[a_face].trim_loops.push(TrimLoop::new(
                TrimLoopKind::Inner,
                vec![Trim::curve(a_pcurve, tolerance)],
            ));
            self.faces[b_face].trim_loops.push(TrimLoop::new(
                TrimLoopKind::Inner,
                vec![Trim::curve(b_pcurve, tolerance)],
            ));

            if let Err(error) = self
                .validate_trim_topology()
                .and_then(|_| self.validate_trim_loop_nesting(a_face, tolerance))
                .and_then(|_| self.validate_trim_loop_nesting(b_face, tolerance))
            {
                self.faces[a_face].trim_loops = old_a_loops;
                self.faces[b_face].trim_loops = old_b_loops;
                self.faces[a_face].split_curves = old_a_splits;
                self.faces[b_face].split_curves = old_b_splits;
                self.split_edges = old_split_edges;
                return Err(error);
            }
            self.identity.record(
                TopologyOperation::InstallTrimReadyFaceCurve,
                Vec::new(),
                vec![a_face_id, b_face_id],
                Vec::new(),
                vec![a_face_id, b_face_id],
            );

            return Ok(TrimReadyFaceConversionReport {
                kind: TrimReadyFaceConversionKind::ClosedInnerLoops,
                split: None,
                a_loop: Some(a_loop),
                b_loop: Some(b_loop),
                snapped_pcurve_endpoints,
                merged_existing: false,
            });
        }

        let edge_curve =
            self.edge_curve_with_pcurve_endpoints(edge_curve, a_face, &a_pcurve, tolerance);
        if let Some(split) = self.find_matching_face_split(
            a_face,
            b_face,
            &edge_curve,
            &a_pcurve,
            &b_pcurve,
            tolerance,
        ) {
            return Ok(TrimReadyFaceConversionReport {
                kind: TrimReadyFaceConversionKind::OpenSplit,
                split: Some(split),
                a_loop: None,
                b_loop: None,
                snapped_pcurve_endpoints,
                merged_existing: true,
            });
        }

        match self
            .split_faces_with_curves(a_face, b_face, edge_curve, a_pcurve, b_pcurve, tolerance)
        {
            Ok(split) => Ok(TrimReadyFaceConversionReport {
                kind: TrimReadyFaceConversionKind::OpenSplit,
                split: Some(split),
                a_loop: None,
                b_loop: None,
                snapped_pcurve_endpoints,
                merged_existing: false,
            }),
            Err(error) => {
                self.faces[a_face].trim_loops = old_a_loops;
                self.faces[b_face].trim_loops = old_b_loops;
                self.faces[a_face].split_curves = old_a_splits;
                self.faces[b_face].split_curves = old_b_splits;
                self.split_edges = old_split_edges;
                Err(error)
            }
        }
    }

    fn close_pcurve_gaps_to_face_boundary(
        &self,
        face: FaceId,
        pcurve: TrimCurve2D,
        tolerance: f64,
    ) -> (TrimCurve2D, usize) {
        let Some((start, end)) = pcurve.endpoints() else {
            return (pcurve, 0);
        };
        let mut snapped = 0;
        let mut new_start = start;
        let mut new_end = end;
        if let Some(point) = self.nearest_face_trim_point(face, start, tolerance) {
            if surface_parameter_distance(&self.faces[face].surface, start, point, tolerance)
                <= tolerance
            {
                new_start = point;
                snapped += 1;
            }
        }
        if let Some(point) = self.nearest_face_trim_point(face, end, tolerance) {
            if surface_parameter_distance(&self.faces[face].surface, end, point, tolerance)
                <= tolerance
            {
                new_end = point;
                snapped += 1;
            }
        }
        (
            trim_curve_with_endpoints(pcurve, new_start, new_end),
            snapped,
        )
    }

    fn nearest_face_trim_point(&self, face: FaceId, point: Vec2, tolerance: f64) -> Option<Vec2> {
        let mut best = None;
        let mut best_distance = tolerance;
        for trim_loop in &self.faces.get(face)?.trim_loops {
            for trim in &trim_loop.trims {
                let samples = trim.curve.sample_points(33)?;
                for segment in samples.windows(2) {
                    let candidate = closest_point_on_segment(point, segment[0], segment[1]);
                    let distance = surface_parameter_distance(
                        &self.faces[face].surface,
                        point,
                        candidate,
                        tolerance,
                    );
                    if distance <= best_distance {
                        best = Some(candidate);
                        best_distance = distance;
                    }
                }
            }
        }
        best
    }

    fn find_matching_inner_trim_loop(
        &self,
        face: FaceId,
        pcurve: &TrimCurve2D,
        tolerance: f64,
    ) -> Option<usize> {
        let incoming = pcurve.sample_points(33)?;
        for (loop_index, trim_loop) in self.faces.get(face)?.trim_loops.iter().enumerate() {
            if trim_loop.kind != TrimLoopKind::Inner {
                continue;
            }
            let existing = trim_loop_sample_points(trim_loop, 33, tolerance)?;
            if closed_polylines_match(&incoming, &existing, tolerance) {
                return Some(loop_index);
            }
        }
        None
    }

    fn edge_curve_with_pcurve_endpoints(
        &self,
        edge_curve: EdgeCurve3D,
        face: FaceId,
        pcurve: &TrimCurve2D,
        tolerance: f64,
    ) -> EdgeCurve3D {
        let Some((start_uv, end_uv)) = pcurve.endpoints() else {
            return edge_curve;
        };
        let Some(start) = self.faces[face].surface.evaluate(start_uv) else {
            return edge_curve;
        };
        let Some(end) = self.faces[face].surface.evaluate(end_uv) else {
            return edge_curve;
        };
        let Some((old_start, old_end)) = edge_curve.endpoints() else {
            return edge_curve;
        };
        if old_start.distance(start) <= tolerance.max(DEFAULT_EDGE_TOLERANCE)
            && old_end.distance(end) <= tolerance.max(DEFAULT_EDGE_TOLERANCE)
        {
            trim_edge_curve_with_endpoints(edge_curve, start, end)
        } else {
            edge_curve
        }
    }

    fn find_matching_face_split(
        &self,
        a_face: FaceId,
        b_face: FaceId,
        edge_curve: &EdgeCurve3D,
        a_pcurve: &TrimCurve2D,
        b_pcurve: &TrimCurve2D,
        tolerance: f64,
    ) -> Option<SplitFacesReport> {
        for (a_split, a_use) in self.faces.get(a_face)?.split_curves.iter().enumerate() {
            let split_edge = self.split_edges.get(a_use.split_edge)?;
            if !edge_curves_match(&split_edge.curve, edge_curve, tolerance) {
                continue;
            }
            if !trim_curves_match(&a_use.pcurve, a_pcurve, tolerance) {
                continue;
            }
            for (b_split, b_use) in self.faces.get(b_face)?.split_curves.iter().enumerate() {
                if b_use.split_edge == a_use.split_edge
                    && trim_curves_match(&b_use.pcurve, b_pcurve, tolerance)
                {
                    return Some(SplitFacesReport {
                        split_edge: a_use.split_edge,
                        a_face,
                        b_face,
                        a_split,
                        b_split,
                    });
                }
            }
        }
        None
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
            if surface_parameter_distance(&self.faces[face_id].surface, end, next_start, tolerance)
                > tolerance
            {
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
        let mut uv_points = project_model_points_to_surface(surface, &model_points, tolerance)?;
        let start_uv = surface.project_point_near(start, uv_points.first().copied(), tolerance)?;
        let end_uv = surface.project_point_near(end, uv_points.last().copied(), tolerance)?;
        *uv_points.first_mut()? = start_uv;
        *uv_points.last_mut()? = end_uv;
        stabilize_singular_uv_samples(surface, &mut uv_points, tolerance);
        unwrap_periodic_uv_samples(surface, &mut uv_points, tolerance);
        if has_interior_singular_samples(surface, &uv_points, tolerance) {
            return None;
        }
        let start_uv = *uv_points.first()?;
        let end_uv = *uv_points.last()?;
        let pcurve = pcurve_from_uv_samples(&uv_points, tolerance)?;
        Some(trim_curve_with_endpoints(pcurve, start_uv, end_uv))
    }

    fn normalize_face_trim_loop_parameters(&mut self, face: FaceId, tolerance: f64) {
        let surface = self.faces[face].surface.clone();
        for trim_loop in &mut self.faces[face].trim_loops {
            normalize_trim_loop_parameters(trim_loop, &surface, tolerance);
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

const DEFAULT_EDGE_TOLERANCE: f64 = 1.0e-9;
const DEFAULT_TRIM_TOLERANCE: f64 = 1.0e-9;
const TWO_PI: f64 = core::f64::consts::PI * 2.0;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const HASH_GRID: f64 = 1_000_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopPointLocation {
    Inside,
    Boundary,
    Outside,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DisjointSets {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSets {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        let parent = self.parent[index];
        if parent != index {
            let root = self.find(parent);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut a_root = self.find(a);
        let mut b_root = self.find(b);
        if a_root == b_root {
            return;
        }
        if self.rank[a_root] < self.rank[b_root] {
            core::mem::swap(&mut a_root, &mut b_root);
        }
        self.parent[b_root] = a_root;
        if self.rank[a_root] == self.rank[b_root] {
            self.rank[a_root] += 1;
        }
    }
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

fn validate_persistent_id_slice(
    ids: &[PersistentId],
    kind: PersistentTopologyKind,
    next_serial: u64,
    seen: &mut HashSet<PersistentId>,
) -> Result<(), TopologyError> {
    for id in ids {
        if id.kind != kind || id.serial == 0 || id.serial >= next_serial || !seen.insert(*id) {
            return Err(TopologyError::InvalidPersistentIdentity);
        }
    }
    Ok(())
}

fn normalized_index(index: usize, samples: usize) -> f64 {
    if samples <= 1 {
        0.0
    } else {
        index as f64 / (samples - 1) as f64
    }
}

fn nurbs_boundary_matches(
    surface: &NurbsSurface,
    a: SurfaceBoundary,
    b: SurfaceBoundary,
    tolerance: f64,
) -> bool {
    let ((u0, u1), (v0, v1)) = surface.domain();
    for index in 0..=8 {
        let t = index as f64 / 8.0;
        let first = match a {
            SurfaceBoundary::UMin => surface.evaluate(u0, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::UMax => surface.evaluate(u1, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::VMin => surface.evaluate(lerp_scalar(u0, u1, t), v0),
            SurfaceBoundary::VMax => surface.evaluate(lerp_scalar(u0, u1, t), v1),
        };
        let second = match b {
            SurfaceBoundary::UMin => surface.evaluate(u0, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::UMax => surface.evaluate(u1, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::VMin => surface.evaluate(lerp_scalar(u0, u1, t), v0),
            SurfaceBoundary::VMax => surface.evaluate(lerp_scalar(u0, u1, t), v1),
        };
        if first.distance(second) > tolerance {
            return false;
        }
    }
    true
}

fn nurbs_singular_boundaries(surface: &NurbsSurface, tolerance: f64) -> Vec<SurfaceBoundary> {
    let mut boundaries = Vec::new();
    for boundary in [
        SurfaceBoundary::UMin,
        SurfaceBoundary::UMax,
        SurfaceBoundary::VMin,
        SurfaceBoundary::VMax,
    ] {
        if nurbs_boundary_is_collapsed(surface, boundary, tolerance) {
            boundaries.push(boundary);
        }
    }
    boundaries
}

fn nurbs_boundary_is_collapsed(
    surface: &NurbsSurface,
    boundary: SurfaceBoundary,
    tolerance: f64,
) -> bool {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let reference = match boundary {
        SurfaceBoundary::UMin => surface.evaluate(u0, v0),
        SurfaceBoundary::UMax => surface.evaluate(u1, v0),
        SurfaceBoundary::VMin => surface.evaluate(u0, v0),
        SurfaceBoundary::VMax => surface.evaluate(u0, v1),
    };
    for index in 1..=8 {
        let t = index as f64 / 8.0;
        let point = match boundary {
            SurfaceBoundary::UMin => surface.evaluate(u0, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::UMax => surface.evaluate(u1, lerp_scalar(v0, v1, t)),
            SurfaceBoundary::VMin => surface.evaluate(lerp_scalar(u0, u1, t), v0),
            SurfaceBoundary::VMax => surface.evaluate(lerp_scalar(u0, u1, t), v1),
        };
        if point.distance(reference) > tolerance {
            return false;
        }
    }
    true
}

fn singular_boundary_at(
    surface: &FaceSurface,
    uv: Vec2,
    tolerance: f64,
) -> Option<SurfaceBoundary> {
    let parameterization = surface.parameterization(tolerance);
    for boundary in &parameterization.singular_boundaries {
        if uv_on_boundary(uv, *boundary, &parameterization, tolerance) {
            return Some(*boundary);
        }
    }
    None
}

fn uv_on_boundary(
    uv: Vec2,
    boundary: SurfaceBoundary,
    parameterization: &SurfaceParameterization,
    tolerance: f64,
) -> bool {
    match boundary {
        SurfaceBoundary::UMin => parameterization
            .u_domain
            .is_some_and(|(u0, _)| (uv.x - u0).abs() <= tolerance),
        SurfaceBoundary::UMax => parameterization
            .u_domain
            .is_some_and(|(_, u1)| (uv.x - u1).abs() <= tolerance),
        SurfaceBoundary::VMin => parameterization
            .v_domain
            .is_some_and(|(v0, _)| (uv.y - v0).abs() <= tolerance),
        SurfaceBoundary::VMax => parameterization
            .v_domain
            .is_some_and(|(_, v1)| (uv.y - v1).abs() <= tolerance),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SurfaceProjection {
    uv: Vec2,
    residual: f64,
}

fn inverse_project_nurbs_with_seed(
    surface: &NurbsSurface,
    point: Point3,
    seed: Option<Vec2>,
    tolerance: f64,
) -> Option<Vec2> {
    let seed = seed.map(|uv| constrain_nurbs_uv(surface, uv, tolerance));
    if let Some(seed) = seed {
        if let Some(projected) = refine_nurbs_projection(surface, point, seed, tolerance) {
            if projected.residual <= tolerance {
                return Some(projected.uv);
            }
        }
    }

    let projected = global_nurbs_projection(surface, point, seed, tolerance)?;
    if projected.residual <= tolerance {
        Some(projected.uv)
    } else {
        None
    }
}

fn global_nurbs_projection(
    surface: &NurbsSurface,
    point: Point3,
    seed: Option<Vec2>,
    tolerance: f64,
) -> Option<SurfaceProjection> {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let (u_period, v_period) = nurbs_periods(surface, tolerance);
    let mut best = None;
    if let Some(seed) = seed {
        best = refine_nurbs_projection(
            surface,
            point,
            constrain_nurbs_uv(surface, seed, tolerance),
            tolerance,
        );
    }
    for j in 0..=4 {
        for i in 0..=4 {
            let u = lerp_scalar(u0, u1, i as f64 / 4.0);
            let v = lerp_scalar(v0, v1, j as f64 / 4.0);
            let candidate = refine_nurbs_projection(surface, point, Vec2::new(u, v), tolerance);
            best = choose_projection(best, candidate, seed, u_period, v_period);
        }
    }

    best
}

fn choose_projection(
    current: Option<SurfaceProjection>,
    candidate: Option<SurfaceProjection>,
    seed: Option<Vec2>,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Option<SurfaceProjection> {
    let Some(candidate) = candidate else {
        return current;
    };
    let Some(current) = current else {
        return Some(candidate);
    };
    let residual_delta = candidate.residual - current.residual;
    if residual_delta < -1.0e-12 {
        return Some(candidate);
    }
    if residual_delta.abs() <= 1.0e-12 {
        if let Some(seed) = seed {
            let current_distance = periodic_uv_distance(current.uv, seed, u_period, v_period);
            let candidate_distance = periodic_uv_distance(candidate.uv, seed, u_period, v_period);
            if candidate_distance < current_distance {
                return Some(candidate);
            }
        }
    }
    Some(current)
}

fn refine_nurbs_projection(
    surface: &NurbsSurface,
    point: Point3,
    seed: Vec2,
    tolerance: f64,
) -> Option<SurfaceProjection> {
    let mut uv = constrain_nurbs_uv(surface, seed, tolerance);
    let mut residual_norm = surface.evaluate(uv.x, uv.y).distance(point);
    if !residual_norm.is_finite() {
        return None;
    }

    for _ in 0..40 {
        let surface_point = surface.evaluate(uv.x, uv.y);
        let residual = surface_point - point;
        residual_norm = residual.norm();
        let (du, dv) = surface.partials(uv.x, uv.y);
        let a00 = du.dot(du);
        let a01 = du.dot(dv);
        let a11 = dv.dot(dv);
        let b0 = -du.dot(residual);
        let b1 = -dv.dot(residual);
        let det = a00 * a11 - a01 * a01;
        if residual_norm <= tolerance {
            return Some(SurfaceProjection {
                uv,
                residual: residual_norm,
            });
        }
        if det.abs() <= 1.0e-18 {
            break;
        }
        let step_u = (b0 * a11 - b1 * a01) / det;
        let step_v = (a00 * b1 - a01 * b0) / det;
        if !step_u.is_finite() || !step_v.is_finite() {
            break;
        }

        let mut accepted = None;
        let mut scale = 1.0;
        for _ in 0..8 {
            let trial = constrain_nurbs_uv(
                surface,
                Vec2::new(uv.x + step_u * scale, uv.y + step_v * scale),
                tolerance,
            );
            let trial_residual = surface.evaluate(trial.x, trial.y).distance(point);
            if trial_residual.is_finite() && trial_residual < residual_norm {
                accepted = Some((trial, trial_residual, scale));
                break;
            }
            scale *= 0.5;
        }

        let Some((trial, trial_residual, scale)) = accepted else {
            break;
        };
        uv = trial;
        residual_norm = trial_residual;
        if step_u.hypot(step_v) * scale <= tolerance.max(1.0e-12) * 0.1 {
            break;
        }
    }

    Some(SurfaceProjection {
        uv,
        residual: residual_norm,
    })
}

fn constrain_nurbs_uv(surface: &NurbsSurface, uv: Vec2, tolerance: f64) -> Vec2 {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let (u_period, v_period) = nurbs_periods(surface, tolerance);
    Vec2::new(
        match u_period {
            Some(period) => wrap_to_periodic_domain(uv.x, u0, u1, period, tolerance),
            None => uv.x.clamp(u0, u1),
        },
        match v_period {
            Some(period) => wrap_to_periodic_domain(uv.y, v0, v1, period, tolerance),
            None => uv.y.clamp(v0, v1),
        },
    )
}

fn nurbs_periods(surface: &NurbsSurface, tolerance: f64) -> (Option<f64>, Option<f64>) {
    let ((u0, u1), (v0, v1)) = surface.domain();
    let u_period = if nurbs_boundary_matches(
        surface,
        SurfaceBoundary::UMin,
        SurfaceBoundary::UMax,
        tolerance,
    ) {
        Some(u1 - u0)
    } else {
        None
    };
    let v_period = if nurbs_boundary_matches(
        surface,
        SurfaceBoundary::VMin,
        SurfaceBoundary::VMax,
        tolerance,
    ) {
        Some(v1 - v0)
    } else {
        None
    };
    (u_period, v_period)
}

fn wrap_to_periodic_domain(value: f64, min: f64, max: f64, period: f64, tolerance: f64) -> f64 {
    if (value - max).abs() <= tolerance {
        return max;
    }
    min + (value - min).rem_euclid(period)
}

fn unwrap_periodic_value(value: f64, seed: Option<f64>, period: f64) -> f64 {
    match seed {
        Some(seed) => value + ((seed - value) / period).round() * period,
        None => value,
    }
}

fn unwrap_uv_near(uv: Vec2, seed: Vec2, u_period: Option<f64>, v_period: Option<f64>) -> Vec2 {
    Vec2::new(
        match u_period {
            Some(period) => unwrap_periodic_value(uv.x, Some(seed.x), period),
            None => uv.x,
        },
        match v_period {
            Some(period) => unwrap_periodic_value(uv.y, Some(seed.y), period),
            None => uv.y,
        },
    )
}

fn periodic_uv_distance(a: Vec2, b: Vec2, u_period: Option<f64>, v_period: Option<f64>) -> f64 {
    let dx = match u_period {
        Some(period) => unwrap_periodic_value(a.x, Some(b.x), period) - b.x,
        None => a.x - b.x,
    };
    let dy = match v_period {
        Some(period) => unwrap_periodic_value(a.y, Some(b.y), period) - b.y,
        None => a.y - b.y,
    };
    dx.hypot(dy)
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

fn trim_curve_is_closed_on_surface(
    surface: &FaceSurface,
    curve: &TrimCurve2D,
    tolerance: f64,
) -> bool {
    curve.endpoints().is_some_and(|(start, end)| {
        surface_parameter_distance(surface, start, end, tolerance) <= tolerance
    })
}

fn close_trim_curve(curve: TrimCurve2D) -> TrimCurve2D {
    let Some((start, _)) = curve.endpoints() else {
        return curve;
    };
    trim_curve_with_endpoints(curve, start, start)
}

fn trim_edge_curve_with_endpoints(curve: EdgeCurve3D, start: Point3, end: Point3) -> EdgeCurve3D {
    match curve {
        EdgeCurve3D::LineSegment { .. } => EdgeCurve3D::LineSegment { start, end },
        EdgeCurve3D::Polyline { mut points } => {
            if let Some(first) = points.first_mut() {
                *first = start;
            }
            if let Some(last) = points.last_mut() {
                *last = end;
            }
            EdgeCurve3D::Polyline { points }
        }
        EdgeCurve3D::Nurbs(mut curve) => {
            if let Some(first) = curve.control_points.first_mut() {
                *first = start;
            }
            if let Some(last) = curve.control_points.last_mut() {
                *last = end;
            }
            EdgeCurve3D::Nurbs(curve)
        }
        other => other,
    }
}

fn edge_curves_match(a: &EdgeCurve3D, b: &EdgeCurve3D, tolerance: f64) -> bool {
    let Some(a_points) = a.sample_points(17) else {
        return false;
    };
    let Some(b_points) = b.sample_points(17) else {
        return false;
    };
    open_point3_polylines_match(&a_points, &b_points, tolerance)
}

fn trim_curves_match(a: &TrimCurve2D, b: &TrimCurve2D, tolerance: f64) -> bool {
    let Some(a_points) = a.sample_points(17) else {
        return false;
    };
    let Some(b_points) = b.sample_points(17) else {
        return false;
    };
    open_vec2_polylines_match(&a_points, &b_points, tolerance)
}

fn open_point3_polylines_match(a: &[Point3], b: &[Point3], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let forward = a
        .iter()
        .zip(b.iter())
        .all(|(a, b)| a.distance(*b) <= tolerance);
    let reverse = a
        .iter()
        .zip(b.iter().rev())
        .all(|(a, b)| a.distance(*b) <= tolerance);
    forward || reverse
}

fn open_vec2_polylines_match(a: &[Vec2], b: &[Vec2], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let forward = a
        .iter()
        .zip(b.iter())
        .all(|(a, b)| vec2_distance(*a, *b) <= tolerance);
    let reverse = a
        .iter()
        .zip(b.iter().rev())
        .all(|(a, b)| vec2_distance(*a, *b) <= tolerance);
    forward || reverse
}

fn closed_polylines_match(a: &[Vec2], b: &[Vec2], tolerance: f64) -> bool {
    let a = normalized_closed_polyline(a, tolerance);
    let b = normalized_closed_polyline(b, tolerance);
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    for offset in 0..b.len() {
        let forward = (0..a.len())
            .all(|index| vec2_distance(a[index], b[(index + offset) % b.len()]) <= tolerance);
        if forward {
            return true;
        }
        let reverse = (0..a.len()).all(|index| {
            let b_index = (offset + b.len() - index % b.len()) % b.len();
            vec2_distance(a[index], b[b_index]) <= tolerance
        });
        if reverse {
            return true;
        }
    }
    false
}

fn normalized_closed_polyline(points: &[Vec2], tolerance: f64) -> Vec<Vec2> {
    let mut points = compact_uv_samples(points, tolerance);
    if points.len() >= 2 && vec2_distance(points[0], points[points.len() - 1]) <= tolerance {
        points.pop();
    }
    points
}

fn closest_point_on_segment(point: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let edge = b - a;
    let len2 = edge.dot(edge);
    if len2 <= f64::EPSILON {
        return a;
    }
    let t = ((point - a).dot(edge) / len2).clamp(0.0, 1.0);
    a + edge * t
}

fn pcurve_from_uv_samples(points: &[Vec2], tolerance: f64) -> Option<TrimCurve2D> {
    let points = compact_uv_samples(points, tolerance);
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

fn project_model_points_to_surface(
    surface: &FaceSurface,
    model_points: &[Point3],
    tolerance: f64,
) -> Option<Vec<Vec2>> {
    let mut uv_points = Vec::with_capacity(model_points.len());
    let mut seed = None;
    for point in model_points {
        let mut uv = surface.project_point_near(*point, seed, tolerance)?;
        if let Some(previous) = uv_points.last().copied() {
            let (u_period, v_period) = surface.parameter_periods(tolerance);
            uv = unwrap_uv_near(uv, previous, u_period, v_period);
        }
        uv_points.push(uv);
        seed = Some(uv);
    }
    stabilize_singular_uv_samples(surface, &mut uv_points, tolerance);
    unwrap_periodic_uv_samples(surface, &mut uv_points, tolerance);
    if has_interior_singular_samples(surface, &uv_points, tolerance) {
        return None;
    }
    Some(uv_points)
}

fn surface_parameter_distance(surface: &FaceSurface, a: Vec2, b: Vec2, tolerance: f64) -> f64 {
    if singular_parameter_equivalent(surface, a, b, tolerance) {
        return 0.0;
    }
    let (u_period, v_period) = surface.parameter_periods(tolerance);
    periodic_uv_distance(a, b, u_period, v_period)
}

fn singular_parameter_equivalent(surface: &FaceSurface, a: Vec2, b: Vec2, tolerance: f64) -> bool {
    let parameterization = surface.parameterization(tolerance);
    parameterization.singular_boundaries.iter().any(|boundary| {
        uv_on_boundary(a, *boundary, &parameterization, tolerance)
            && uv_on_boundary(b, *boundary, &parameterization, tolerance)
    })
}

fn unwrap_periodic_uv_samples(surface: &FaceSurface, points: &mut [Vec2], tolerance: f64) {
    let (u_period, v_period) = surface.parameter_periods(tolerance);
    for index in 1..points.len() {
        points[index] = unwrap_uv_near(points[index], points[index - 1], u_period, v_period);
    }
}

fn stabilize_singular_uv_samples(surface: &FaceSurface, points: &mut [Vec2], tolerance: f64) {
    if points.len() < 2 {
        return;
    }
    for index in 0..points.len() {
        let neighbor = if index == 0 {
            points[1]
        } else {
            points[index - 1]
        };
        points[index] = stabilize_singular_uv(surface, points[index], neighbor, tolerance);
    }
}

fn stabilize_singular_uv(
    surface: &FaceSurface,
    mut uv: Vec2,
    neighbor: Vec2,
    tolerance: f64,
) -> Vec2 {
    let Some(boundary) = singular_boundary_at(surface, uv, tolerance) else {
        return uv;
    };
    match boundary {
        SurfaceBoundary::UMin | SurfaceBoundary::UMax => uv.y = neighbor.y,
        SurfaceBoundary::VMin | SurfaceBoundary::VMax => uv.x = neighbor.x,
    }
    uv
}

fn has_interior_singular_samples(surface: &FaceSurface, points: &[Vec2], tolerance: f64) -> bool {
    points.len() > 2
        && points[1..points.len() - 1]
            .iter()
            .any(|uv| surface.is_singular_at(*uv, tolerance))
}

fn normalize_trim_loop_parameters(trim_loop: &mut TrimLoop, surface: &FaceSurface, tolerance: f64) {
    if trim_loop.trims.len() < 2 {
        return;
    }
    let (u_period, v_period) = surface.parameter_periods(tolerance);
    if u_period.is_none() && v_period.is_none() {
        return;
    }
    for index in 1..trim_loop.trims.len() {
        let Some((_, previous_end)) = trim_loop.trims[index - 1].curve.endpoints() else {
            continue;
        };
        let Some((start, _)) = trim_loop.trims[index].curve.endpoints() else {
            continue;
        };
        let offset = periodic_offset_to_near(start, previous_end, u_period, v_period);
        if offset.x.abs() > tolerance || offset.y.abs() > tolerance {
            translate_trim_curve(&mut trim_loop.trims[index].curve, offset);
        }
    }
}

fn periodic_offset_to_near(
    value: Vec2,
    target: Vec2,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Vec2 {
    Vec2::new(
        u_period
            .map(|period| ((target.x - value.x) / period).round() * period)
            .unwrap_or(0.0),
        v_period
            .map(|period| ((target.y - value.y) / period).round() * period)
            .unwrap_or(0.0),
    )
}

fn translate_trim_curve(curve: &mut TrimCurve2D, offset: Vec2) {
    match curve {
        TrimCurve2D::LineSegment { start, end } => {
            *start = *start + offset;
            *end = *end + offset;
        }
        TrimCurve2D::CircularArc { center, .. } => {
            *center = *center + offset;
        }
        TrimCurve2D::Polyline { points } => {
            for point in points {
                *point = *point + offset;
            }
        }
        TrimCurve2D::Nurbs(curve) => {
            for point in &mut curve.control_points {
                point.x += offset.x;
                point.y += offset.y;
            }
        }
        TrimCurve2D::Unresolved => {}
    }
}

fn compact_uv_samples(points: &[Vec2], tolerance: f64) -> Vec<Vec2> {
    let mut compact = Vec::with_capacity(points.len());
    for point in points {
        if compact
            .last()
            .is_none_or(|last| vec2_distance(*last, *point) > tolerance)
        {
            compact.push(*point);
        }
    }
    compact
}

fn trim_curve_with_endpoints(curve: TrimCurve2D, start: Vec2, end: Vec2) -> TrimCurve2D {
    match curve {
        TrimCurve2D::LineSegment { .. } => TrimCurve2D::LineSegment { start, end },
        TrimCurve2D::Polyline { mut points } => {
            if let Some(first) = points.first_mut() {
                *first = start;
            }
            if let Some(last) = points.last_mut() {
                *last = end;
            }
            TrimCurve2D::Polyline { points }
        }
        TrimCurve2D::Nurbs(mut curve) => {
            if let Some(first) = curve.control_points.first_mut() {
                *first = Point3::new(start.x, start.y, 0.0);
            }
            if let Some(last) = curve.control_points.last_mut() {
                *last = Point3::new(end.x, end.y, 0.0);
            }
            TrimCurve2D::Nurbs(curve)
        }
        other => other,
    }
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
