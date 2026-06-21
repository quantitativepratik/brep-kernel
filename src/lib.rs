#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
//! Boundary-representation kernel prototype.
//!
//! This crate is intentionally dependency-light. The core modules implement:
//! half-edge topology for closed manifold triangle B-reps, analytic primitives,
//! NURBS curves/surfaces, interval-filtered predicates, representative
//! intersection routines, a cube-minus-cylinder boolean, CPU tessellation, and
//! raw WASM exports for the browser viewer.

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

pub use math::{Point3, Vec2, Vec3};
