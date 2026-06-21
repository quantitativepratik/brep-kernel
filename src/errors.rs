//! Structured kernel diagnostics and errors.

use crate::boolean::BooleanError;
use crate::euler::EulerError;
use crate::features::FeatureError;
use crate::nurbs::NurbsError;
use crate::topology::{
    EdgeId, FaceId, HalfEdgeId, PersistentId, SplitEdgeId, TopologyError, VertexId,
};
use std::fmt;

/// Kernel result alias.
pub type KernelResult<T> = Result<T, KernelError>;

/// Subsystem that produced a diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelSubsystem {
    /// Topology and half-edge validation.
    Topology,
    /// Geometry primitives and evaluations.
    Geometry,
    /// NURBS evaluation and fitting.
    Nurbs,
    /// Surface/curve intersections.
    Intersection,
    /// Boolean classification and output generation.
    Boolean,
    /// Euler construction operators.
    Euler,
    /// Parametric feature layer.
    Feature,
    /// File exchange import/export.
    Exchange,
    /// GPU or tessellation layer.
    Tessellation,
}

/// Diagnostic severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Informational diagnostic.
    Info,
    /// Non-fatal warning.
    Warning,
    /// Recoverable error.
    Error,
    /// Fatal error that invalidates the operation.
    Fatal,
}

/// Broad error category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelErrorKind {
    /// Caller supplied invalid input.
    InvalidInput,
    /// Requested operation is outside the supported kernel scope.
    Unsupported,
    /// Text or exchange data could not be parsed.
    Parse,
    /// Topology or geometry validation failed.
    Validation,
    /// Numerical routine did not converge or could not certify a result.
    Numerical,
    /// Topology is open, non-manifold, or otherwise inconsistent.
    Topology,
    /// Import/export format is malformed or unsupported.
    Exchange,
    /// Unexpected internal invariant failure.
    Internal,
}

/// Optional entity reference attached to a diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelEntityRef {
    /// Snapshot vertex id.
    Vertex(VertexId),
    /// Snapshot half-edge/coedge id.
    HalfEdge(HalfEdgeId),
    /// Snapshot edge id.
    Edge(EdgeId),
    /// Staged split-edge id.
    SplitEdge(SplitEdgeId),
    /// Snapshot face id.
    Face(FaceId),
    /// Persistent topology id.
    Persistent(PersistentId),
    /// Exchange entity id, such as a STEP `#42` record.
    ExchangeEntity {
        /// Exchange format label.
        format: String,
        /// Entity number.
        id: usize,
    },
    /// Named entity or external reference.
    Named(String),
}

/// One structured diagnostic record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelDiagnostic {
    /// Diagnostic severity.
    pub severity: DiagnosticSeverity,
    /// Subsystem that produced the diagnostic.
    pub subsystem: KernelSubsystem,
    /// Broad category.
    pub kind: KernelErrorKind,
    /// Stable short code.
    pub code: String,
    /// Human-readable diagnostic.
    pub message: String,
    /// Operation being performed.
    pub operation: Option<String>,
    /// Related entity, when available.
    pub entity: Option<KernelEntityRef>,
    /// Source error text from a lower-level subsystem.
    pub source: Option<String>,
    /// Additional notes.
    pub notes: Vec<String>,
}

impl KernelDiagnostic {
    /// Construct a diagnostic.
    pub fn new(
        severity: DiagnosticSeverity,
        subsystem: KernelSubsystem,
        kind: KernelErrorKind,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            subsystem,
            kind,
            code: code.into(),
            message: message.into(),
            operation: None,
            entity: None,
            source: None,
            notes: Vec::new(),
        }
    }
}

/// Structured kernel error with a primary diagnostic and related diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelError {
    /// Primary diagnostic.
    pub primary: Box<KernelDiagnostic>,
    /// Related diagnostics.
    pub related: Vec<KernelDiagnostic>,
}

impl KernelError {
    /// Construct a new structured error.
    pub fn new(
        subsystem: KernelSubsystem,
        kind: KernelErrorKind,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            primary: Box::new(KernelDiagnostic::new(
                DiagnosticSeverity::Error,
                subsystem,
                kind,
                code,
                message,
            )),
            related: Vec::new(),
        }
    }

    /// Construct an unsupported-operation error.
    pub fn unsupported(
        subsystem: KernelSubsystem,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(
            subsystem,
            KernelErrorKind::Unsupported,
            "unsupported",
            message,
        )
        .with_operation(operation)
    }

    /// Attach an operation name.
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.primary.operation = Some(operation.into());
        self
    }

    /// Attach an entity reference.
    pub fn with_entity(mut self, entity: KernelEntityRef) -> Self {
        self.primary.entity = Some(entity);
        self
    }

    /// Attach a lower-level source error string.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.primary.source = Some(source.into());
        self
    }

    /// Attach an additional note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.primary.notes.push(note.into());
        self
    }

    /// Add a related diagnostic.
    pub fn with_related(mut self, diagnostic: KernelDiagnostic) -> Self {
        self.related.push(diagnostic);
        self
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.primary.code, self.primary.message)?;
        if let Some(operation) = &self.primary.operation {
            write!(f, " while running `{operation}`")?;
        }
        if let Some(source) = &self.primary.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl std::error::Error for KernelError {}

impl From<TopologyError> for KernelError {
    fn from(value: TopologyError) -> Self {
        Self::new(
            KernelSubsystem::Topology,
            KernelErrorKind::Topology,
            "topology.error",
            "topology operation failed",
        )
        .with_source(format!("{value:?}"))
    }
}

impl From<BooleanError> for KernelError {
    fn from(value: BooleanError) -> Self {
        Self::new(
            KernelSubsystem::Boolean,
            KernelErrorKind::Validation,
            "boolean.error",
            "boolean operation failed",
        )
        .with_source(format!("{value:?}"))
    }
}

impl From<EulerError> for KernelError {
    fn from(value: EulerError) -> Self {
        Self::new(
            KernelSubsystem::Euler,
            KernelErrorKind::Validation,
            "euler.error",
            "Euler operation failed",
        )
        .with_source(format!("{value:?}"))
    }
}

impl From<FeatureError> for KernelError {
    fn from(value: FeatureError) -> Self {
        Self::new(
            KernelSubsystem::Feature,
            KernelErrorKind::InvalidInput,
            "feature.error",
            "feature operation failed",
        )
        .with_source(format!("{value:?}"))
    }
}

impl From<NurbsError> for KernelError {
    fn from(value: NurbsError) -> Self {
        Self::new(
            KernelSubsystem::Nurbs,
            KernelErrorKind::Numerical,
            "nurbs.error",
            "NURBS operation failed",
        )
        .with_source(format!("{value:?}"))
    }
}
