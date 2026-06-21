# Changelog

## Unreleased

- Added the `brep_kernel::api` facade and top-level `brep_kernel::prelude` as the curated public compatibility surface.
- Added runtime version metadata: crate version components, `API_REVISION`, `WASM_ABI_REVISION`, and `MINIMUM_SUPPORTED_RUST_VERSION`.
- Added public API policy documentation, facade regression tests, and rustdoc checks in CI.
- Routed the raw WASM `brep_version()` export through the shared ABI revision constant.

## 0.1.0

Initial public prototype:

- half-edge topology and Euler construction operators
- analytic face support and trim-loop topology
- 3D edge curves plus per-face 2D p-curves
- robust trim-loop orientation/nesting analysis and NURBS p-curve generation
- periodic seam unwrapping and singular surface-boundary handling for p-curves
- topology-level tolerance-aware triangle mesh sewing with deterministic reports
- staged face splitting from trim-ready SSI curves
- gap-closing trim-ready SSI promotion into open splits or closed inner trim loops
- Boolean classification over staged split faces
- healed Boolean trim-region generation for supported split faces
- NURBS curve/surface evaluation, derivatives, and curve interpolation
- interval-filtered robust predicates
- representative curve/surface intersections and trim-ready NURBS/NURBS SSI output with NURBS curve fitting
- cube-minus-cylinder boolean reference case
- natural-language to parametric feature-tree layer
- golden reference models and property-based tests
- browser WebGPU/WASM viewer path
- native `wgpu` viewer binary
