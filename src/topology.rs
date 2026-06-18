//! Half-edge topology for closed manifold B-reps.

use crate::geometry::Plane;
use crate::math::{Point3, Vec3};
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

/// Face geometry tag.
#[derive(Clone, Debug, PartialEq)]
pub enum FaceGeometry {
    /// Planar face.
    Plane(Plane),
    /// Faceted or not yet classified.
    Faceted,
}

/// Topological face.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// One half-edge on the outer loop.
    pub halfedge: HalfEdgeId,
    /// Geometric support.
    pub geometry: FaceGeometry,
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
                geometry: FaceGeometry::Faceted,
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
        Ok(())
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

    fn classify_planar_faces(&mut self) {
        for face_id in 0..self.faces.len() {
            let tri = self.face_vertices(face_id);
            let a = self.vertices[tri[0]].point;
            let b = self.vertices[tri[1]].point;
            let c = self.vertices[tri[2]].point;
            if let Some(plane) = Plane::from_points(a, b, c) {
                self.faces[face_id].geometry = FaceGeometry::Plane(plane);
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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const HASH_GRID: f64 = 1_000_000_000.0;

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
