#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
//! Boundary-representation kernel prototype.
//!
//! This crate is intentionally dependency-light. The core modules implement:
//! half-edge topology for closed manifold triangle B-reps, analytic primitives,
//! NURBS curves/surfaces, interval-filtered predicates, representative
//! intersection routines, a cube-minus-cylinder boolean, CPU tessellation, and
//! raw WASM exports for the browser viewer.
//!
//! Prefer [`api`] or [`prelude`] for application code. The direct subsystem
//! modules are public for experimentation and focused kernel work, but the API
//! facade is the compatibility boundary documented by the versioning policy.

pub mod api;
pub mod boolean;
pub mod errors;
pub mod euler;
pub mod exchange;
pub mod features;
pub mod geometry;
pub mod intersection;
pub mod math;
pub mod nurbs;
pub mod predicates;
pub mod tessellation;
pub mod topology;
pub mod wasm;

/// Common imports from the stable public API facade.
pub mod prelude {
    pub use crate::api::prelude::*;
}

pub use math::{Point3, Vec2, Vec3};
