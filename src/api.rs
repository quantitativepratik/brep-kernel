//! Stable public API facade.
//!
//! The crate's implementation modules remain public while the kernel is young,
//! but this facade is the compatibility boundary callers should prefer. Items
//! re-exported here are the first candidates for semver-protected API stability;
//! lower-level modules can still evolve as algorithms move from prototype to
//! supported behavior.

pub use crate::boolean::{subtract_cube_cylinder, BooleanError, BooleanOp, BooleanReport};
pub use crate::errors::{
    DiagnosticSeverity, KernelDiagnostic, KernelDiagnosticReport, KernelEntityRef, KernelError,
    KernelErrorKind, KernelResult, KernelSubsystem,
};
pub use crate::exchange::{
    export_iges_faceted_brep, export_step_faceted_brep, import_iges_faceted_brep,
    import_step_faceted_brep, ExchangeFormat,
};
pub use crate::features::{
    build_plate_with_holes, execute_feature_tree, parse_feature_prompt, BasePlate, FeatureError,
    FeatureNode, FeatureOperation, FeatureTree, ThroughHole, Unit,
};
pub use crate::geometry::{AxisAlignedBox, Circle, Cylinder, Line, Plane};
pub use crate::math::{Point3, Vec2, Vec3};
pub use crate::nurbs::{KnotVector, NurbsCurve, NurbsError, NurbsSurface};
pub use crate::predicates::{incircle2d, orient2d, orient3d, RobustSign};
pub use crate::tessellation::{tessellate_nurbs_surface, TessVertex, Tessellation};
pub use crate::topology::{
    CoedgeTolerance, EdgeCurve3D, EdgeId, EdgeTolerance, FaceId, FaceSurface, FaceTolerance,
    HalfEdgeId, PersistentId, PersistentTopologyKind, SewingReport, Solid, SplitEdgeId,
    TopologyCommitReport, TopologyCounts, TopologyError, TopologyHistoryEvent, TopologyIdentity,
    TopologyOperation, TopologyRevisionId, TopologyRollbackEntry, TopologyRollbackReport,
    TopologyToleranceModel, Trim, TrimCurve2D, TrimLoop, TrimLoopAnalysis, TrimLoopKind,
    TrimLoopNesting, TrimLoopOrientation, VertexId, VertexTolerance,
};

/// Current revision of the Rust API facade.
///
/// This number is intentionally separate from the crate semver. It increments
/// when the curated facade gains, removes, or materially changes an item.
pub const API_REVISION: u32 = 1;

/// Current raw WASM ABI revision used by the browser viewer exports.
pub const WASM_ABI_REVISION: u32 = 1;

/// Minimum supported Rust version documented for this crate.
pub const MINIMUM_SUPPORTED_RUST_VERSION: &str = "1.87";

/// Version metadata for the public API and browser ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApiVersion {
    /// Cargo package name.
    pub crate_name: &'static str,
    /// Cargo package version.
    pub crate_version: &'static str,
    /// Cargo package major version component.
    pub crate_version_major: &'static str,
    /// Cargo package minor version component.
    pub crate_version_minor: &'static str,
    /// Cargo package patch version component.
    pub crate_version_patch: &'static str,
    /// Revision of the curated Rust facade.
    pub api_revision: u32,
    /// Revision of the raw WASM ABI.
    pub wasm_abi_revision: u32,
    /// Minimum supported Rust version.
    pub msrv: &'static str,
}

/// Return version metadata for runtime diagnostics and compatibility checks.
pub fn version() -> ApiVersion {
    ApiVersion {
        crate_name: env!("CARGO_PKG_NAME"),
        crate_version: env!("CARGO_PKG_VERSION"),
        crate_version_major: env!("CARGO_PKG_VERSION_MAJOR"),
        crate_version_minor: env!("CARGO_PKG_VERSION_MINOR"),
        crate_version_patch: env!("CARGO_PKG_VERSION_PATCH"),
        api_revision: API_REVISION,
        wasm_abi_revision: WASM_ABI_REVISION,
        msrv: MINIMUM_SUPPORTED_RUST_VERSION,
    }
}

/// Common imports for applications using the supported facade.
pub mod prelude {
    pub use super::{
        build_plate_with_holes, execute_feature_tree, export_iges_faceted_brep,
        export_step_faceted_brep, import_iges_faceted_brep, import_step_faceted_brep, incircle2d,
        orient2d, orient3d, parse_feature_prompt, subtract_cube_cylinder, tessellate_nurbs_surface,
        version, ApiVersion, AxisAlignedBox, BasePlate, BooleanError, BooleanOp, BooleanReport,
        Circle, CoedgeTolerance, Cylinder, DiagnosticSeverity, EdgeCurve3D, EdgeId, EdgeTolerance,
        ExchangeFormat, FaceId, FaceSurface, FaceTolerance, FeatureError, FeatureNode,
        FeatureOperation, FeatureTree, HalfEdgeId, KernelDiagnostic, KernelDiagnosticReport,
        KernelEntityRef, KernelError, KernelErrorKind, KernelResult, KernelSubsystem, KnotVector,
        Line, NurbsCurve, NurbsError, NurbsSurface, PersistentId, PersistentTopologyKind, Plane,
        Point3, RobustSign, SewingReport, Solid, SplitEdgeId, TessVertex, Tessellation,
        ThroughHole, TopologyCommitReport, TopologyCounts, TopologyError, TopologyHistoryEvent,
        TopologyIdentity, TopologyOperation, TopologyRevisionId, TopologyRollbackEntry,
        TopologyRollbackReport, TopologyToleranceModel, Trim, TrimCurve2D, TrimLoop,
        TrimLoopAnalysis, TrimLoopKind, TrimLoopNesting, TrimLoopOrientation, Unit, Vec2, Vec3,
        VertexId, VertexTolerance, API_REVISION, MINIMUM_SUPPORTED_RUST_VERSION, WASM_ABI_REVISION,
    };
}
