//! Euler operators for constructive B-rep topology.
//!
//! The existing [`crate::topology::Solid`] type is the validated half-edge
//! boundary representation. This module adds a mutable construction layer with
//! classic make operators:
//!
//! - [`EulerBuilder::mvfs`] - make vertex, face, shell
//! - [`EulerBuilder::mev`] - make edge and vertex
//! - [`EulerBuilder::mef`] - make edge and face
//!
//! The builder tracks polygonal face loops and Euler counts. It can emit a
//! triangulated [`crate::topology::Solid`], which then goes through the same
//! closed-manifold validation path as every other kernel output.

use crate::math::Point3;
use crate::topology::{FaceId, Solid, TopologyError, VertexId};

/// Construction edge identifier.
pub type EulerEdgeId = usize;
/// Construction shell identifier.
pub type EulerShellId = usize;

/// Stable counts for the Euler construction state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EulerCounts {
    /// Number of vertices.
    pub vertices: usize,
    /// Number of edges.
    pub edges: usize,
    /// Number of faces.
    pub faces: usize,
    /// Number of shells.
    pub shells: usize,
}

impl EulerCounts {
    /// Euler characteristic `V - E + F`.
    pub fn euler_characteristic(self) -> isize {
        self.vertices as isize - self.edges as isize + self.faces as isize
    }
}

/// Error returned by Euler construction operators.
#[derive(Clone, Debug, PartialEq)]
pub enum EulerError {
    /// `MVFS` can only be called on an empty builder.
    BuilderAlreadyInitialized,
    /// An operator requiring an initialized shell was called too early.
    EmptyBuilder,
    /// Face id is not present.
    InvalidFace(FaceId),
    /// Vertex id is not present.
    InvalidVertex(VertexId),
    /// A vertex is not in the requested face loop.
    VertexNotInFace {
        /// Face that was searched.
        face: FaceId,
        /// Vertex that was not found.
        vertex: VertexId,
    },
    /// A requested edge already exists.
    EdgeAlreadyExists(VertexId, VertexId),
    /// `MEF` cannot split or close the requested loop.
    DegenerateFaceSplit,
    /// Conversion to a half-edge solid found an open construction face.
    OpenConstructionFace(FaceId),
    /// Final half-edge topology validation failed.
    Topology(TopologyError),
}

impl From<TopologyError> for EulerError {
    fn from(value: TopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Mutable Euler-operator construction state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EulerBuilder {
    vertices: Vec<Point3>,
    edges: Vec<EdgeKey>,
    faces: Vec<Vec<VertexId>>,
    shells: usize,
}

impl EulerBuilder {
    /// Create an empty Euler builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Make vertex, face, shell.
    ///
    /// This is the usual `MVFS` seed operation. It creates one vertex, one
    /// shell, and one construction face loop containing that single vertex.
    pub fn mvfs(&mut self, point: Point3) -> Result<(VertexId, FaceId, EulerShellId), EulerError> {
        if !self.vertices.is_empty() || !self.edges.is_empty() || !self.faces.is_empty() {
            return Err(EulerError::BuilderAlreadyInitialized);
        }
        self.vertices.push(point);
        self.faces.push(vec![0]);
        self.shells = 1;
        Ok((0, 0, 0))
    }

    /// Make edge and vertex.
    ///
    /// The new vertex is inserted into `face` immediately after `from` in that
    /// face loop, and a construction edge from `from` to the new vertex is
    /// recorded.
    pub fn mev(
        &mut self,
        face: FaceId,
        from: VertexId,
        point: Point3,
    ) -> Result<VertexId, EulerError> {
        self.ensure_initialized()?;
        self.ensure_vertex(from)?;
        let insert_after = self.vertex_position(face, from)?;
        let new_vertex = self.vertices.len();
        self.vertices.push(point);
        self.add_edge(from, new_vertex)?;
        self.faces[face].insert(insert_after + 1, new_vertex);
        Ok(new_vertex)
    }

    /// Make edge and face.
    ///
    /// `from` and `to` must be distinct vertices in the same face loop. The
    /// operator inserts the edge `from-to`, then either splits a polygonal face
    /// into two loops or closes a wire loop into two oppositely oriented faces.
    pub fn mef(
        &mut self,
        face: FaceId,
        from: VertexId,
        to: VertexId,
    ) -> Result<FaceId, EulerError> {
        self.ensure_initialized()?;
        self.ensure_vertex(from)?;
        self.ensure_vertex(to)?;
        if from == to {
            return Err(EulerError::DegenerateFaceSplit);
        }
        if self.edge_exists(from, to) {
            return Err(EulerError::EdgeAlreadyExists(from, to));
        }
        let from_pos = self.vertex_position(face, from)?;
        let to_pos = self.vertex_position(face, to)?;
        let loop_vertices = self.faces[face].clone();
        let from_to = cyclic_path(&loop_vertices, from_pos, to_pos);
        let to_from = cyclic_path(&loop_vertices, to_pos, from_pos);

        let (retained_loop, new_loop) = match (from_to.len() >= 3, to_from.len() >= 3) {
            (true, true) => (to_from, from_to),
            (true, false) => (reversed(&from_to), from_to),
            (false, true) => (reversed(&to_from), to_from),
            (false, false) => return Err(EulerError::DegenerateFaceSplit),
        };

        self.add_edge(from, to)?;
        self.faces[face] = retained_loop;
        let new_face = self.faces.len();
        self.faces.push(new_loop);
        Ok(new_face)
    }

    /// Return current construction counts.
    pub fn counts(&self) -> EulerCounts {
        EulerCounts {
            vertices: self.vertices.len(),
            edges: self.edges.len(),
            faces: self.faces.len(),
            shells: self.shells,
        }
    }

    /// True when the construction state satisfies the Euler count invariant.
    pub fn satisfies_euler_invariant(&self) -> bool {
        if self.shells == 0 {
            return self.vertices.is_empty() && self.edges.is_empty() && self.faces.is_empty();
        }
        self.counts().euler_characteristic() == 2 * self.shells as isize
    }

    /// Borrow a construction face loop.
    pub fn face_loop(&self, face: FaceId) -> Result<&[VertexId], EulerError> {
        self.faces
            .get(face)
            .map(Vec::as_slice)
            .ok_or(EulerError::InvalidFace(face))
    }

    /// Borrow construction vertices.
    pub fn vertices(&self) -> &[Point3] {
        &self.vertices
    }

    /// Triangulate construction faces and validate the result as a half-edge solid.
    ///
    /// Each construction face must have at least three vertices. Polygonal
    /// loops are triangulated with a fan from the first vertex.
    pub fn to_solid(&self) -> Result<Solid, EulerError> {
        self.ensure_initialized()?;
        let mut triangles = Vec::<[usize; 3]>::new();
        for (face_id, face) in self.faces.iter().enumerate() {
            if face.len() < 3 {
                return Err(EulerError::OpenConstructionFace(face_id));
            }
            for i in 1..face.len() - 1 {
                triangles.push([face[0], face[i], face[i + 1]]);
            }
        }
        Solid::from_triangle_mesh(self.vertices.clone(), &triangles).map_err(EulerError::Topology)
    }

    fn ensure_initialized(&self) -> Result<(), EulerError> {
        if self.shells == 0 {
            Err(EulerError::EmptyBuilder)
        } else {
            Ok(())
        }
    }

    fn ensure_vertex(&self, vertex: VertexId) -> Result<(), EulerError> {
        if vertex < self.vertices.len() {
            Ok(())
        } else {
            Err(EulerError::InvalidVertex(vertex))
        }
    }

    fn vertex_position(&self, face: FaceId, vertex: VertexId) -> Result<usize, EulerError> {
        let face_loop = self.faces.get(face).ok_or(EulerError::InvalidFace(face))?;
        face_loop
            .iter()
            .position(|candidate| *candidate == vertex)
            .ok_or(EulerError::VertexNotInFace { face, vertex })
    }

    fn edge_exists(&self, a: VertexId, b: VertexId) -> bool {
        let key = EdgeKey::new(a, b);
        self.edges.contains(&key)
    }

    fn add_edge(&mut self, a: VertexId, b: VertexId) -> Result<EulerEdgeId, EulerError> {
        let key = EdgeKey::new(a, b);
        if self.edges.contains(&key) {
            return Err(EulerError::EdgeAlreadyExists(a, b));
        }
        let edge = self.edges.len();
        self.edges.push(key);
        Ok(edge)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EdgeKey {
    a: VertexId,
    b: VertexId,
}

impl EdgeKey {
    fn new(a: VertexId, b: VertexId) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

fn cyclic_path(loop_vertices: &[VertexId], start: usize, end: usize) -> Vec<VertexId> {
    let mut out = Vec::new();
    let mut index = start;
    loop {
        out.push(loop_vertices[index]);
        if index == end {
            break;
        }
        index = (index + 1) % loop_vertices.len();
    }
    out
}

fn reversed(vertices: &[VertexId]) -> Vec<VertexId> {
    vertices.iter().rev().copied().collect()
}
