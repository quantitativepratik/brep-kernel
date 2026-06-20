# Boundary-Representation Kernel Prototype

This repository contains a compact, dependency-light B-rep kernel written in Rust, plus a WebGPU browser viewer.

It is not an industrial CAD kernel. It is a defensible core that exercises the hard parts directly: topology, rational geometry, robustness, intersections, booleans, GPU tessellation, and WASM interop.

## Status

This is a research/portfolio prototype for learning and demonstrating CAD-kernel internals. It is useful for studying topology, NURBS evaluation, robustness, representative intersections, regression testing, and WebGPU/WASM integration. It is not ready to replace OpenCascade, Parasolid, ACIS, or other production kernels.

The supported scope is intentionally explicit: each implemented operation has tests, and unsupported cases are called out instead of hidden behind broad claims.

## Quick Start

```sh
cargo test
cargo run --features native-viewer --bin native-viewer
```

For the browser viewer:

```sh
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
python3 -m http.server 8080
```

Then open `http://localhost:8080/web/`.

## What Is Implemented

- Half-edge topology for closed manifold triangle B-reps: vertices, half-edges, edges, faces, shells, solids, Euler/genus validation, signed volume.
- Edge and face geometry: `EdgeCurve3D` model-space curves on topological edges, `FaceSurface` support surfaces, outer/inner `TrimLoop`s with per-face 2D p-curves, trim-loop orientation/nesting analysis, seeded NURBS-surface p-curve generation, periodic seam unwrapping, and singular-boundary handling.
- Tolerance-aware sewing: clusters near-coincident mesh vertices, removes triangles collapsed by sewing, reports the vertex remap, and validates the sewn mesh as half-edge topology.
- Euler construction operators: `MVFS`, `MEV`, and `MEF`, with count invariants and conversion into validated half-edge solids.
- Analytic primitives: lines, planes, circles, boxes, cylinders.
- NURBS curves and surfaces: clamped/nonuniform knot vectors, rational evaluation, first derivatives, surface normals, curve knot insertion, and global curve interpolation.
- Numerical robustness layer: outward-rounded interval predicates for `orient2d`, `orient3d`, and `incircle2d`.
- Intersections:
  - line-plane
  - plane-plane
  - NURBS-curve/plane via bracketing and bisection
  - plane/NURBS-surface via marching squares over the parametric domain
  - NURBS/NURBS surface intersection via tessellated discovery, NURBS residual refinement, trim-ready 3D/p-curve output, and conservative NURBS fitting of SSI polylines
- Face splitting:
  - staged `SplitEdge` records with one model-space curve and per-face p-curves
  - rollback-safe installation of trim-ready SSI output onto two analytic faces
  - trim-ready SSI promotion that gap-closes open curves to existing trims, merges duplicate split edges, and installs closed curves as inner trim loops on both faces
- Boolean classification:
  - classifies staged face splits by trim-domain location and local inside/outside side
  - maps split sides to Union/Intersect/Subtract keep, discard, and reversed-keep actions
- Healed Boolean output:
  - promotes supported classified split sides into new trim loops
  - triangulates healed regions, runs tolerance-aware sewing, and validates a new half-edge solid when the generated shell closes
- Boolean operation:
  - cube minus Z-cylinder as a validated genus-1 half-edge solid
- Thin parametric feature layer:
  - parses prompts such as `10mm bracket with two M4 holes`
  - emits an explicit base-plate plus through-hole feature tree
  - executes the tree as a validated B-rep using `earcutr` for polygon-with-holes triangulation
- Tessellation:
  - CPU regular-grid NURBS surface tessellation
  - WGSL compute shader for rational cubic patch tessellation
- Native viewer:
  - `wgpu`/`winit` binary rendering a tessellated NURBS patch
- Browser frontend:
  - WebGPU render path
  - GPU NURBS patch mode
  - B-rep boolean mesh mode via Rust WASM export, with JS fallback
- Regression corpus under `corpus/regression`.
- Versioned reference models under `corpus/reference/v1` with golden mesh hashes, topology counts, volume, and surface-area invariants.
- Property-based tests using `proptest` to generate valid solids and assert closed-manifold invariants.

## Run The Core Tests

```sh
cargo test
```

The tests cover topology closure, Euler operators, split-face Boolean classification and healing, boolean genus, NURBS evaluation and knot insertion, robust predicate uncertainty, intersections, tessellation, golden reference models, natural-language feature execution, property-generated solids, and the regression corpus.

To print updated reference outputs after an intentional mesh change:

```sh
cargo test --test reference_models dump_reference_golden_outputs -- --ignored --nocapture
```

## Run The Browser Viewer

The viewer is static, but it should be served over HTTP so it can fetch the WGSL shader and optional WASM artifact.

```sh
cargo build --target wasm32-unknown-unknown --release
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080/web/
```

If the WASM target is not installed, the B-rep mode still renders through the JavaScript fallback. Install the target with:

```sh
rustup target add wasm32-unknown-unknown
```

## Run The Native Viewer

The native viewer is feature-gated so normal kernel tests and WASM builds do not pull in desktop windowing dependencies.

```sh
cargo run --features native-viewer --bin native-viewer
```

It opens a `wgpu` window and renders a CPU-tessellated rational NURBS patch with depth shading and a rotating camera. Press `Esc` to close it.

## Quality Gates

The GitHub Actions workflow runs:

```sh
cargo fmt -- --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo build --target wasm32-unknown-unknown --release --locked
cargo check --features native-viewer --bin native-viewer --locked
```

## Design Boundaries

The boolean module deliberately supports one hard representative closed-output case instead of pretending to solve all solid modeling. The result is a real closed half-edge solid for cube-minus-cylinder with `V - E + F = 0`, which is the expected genus-1 topology. For the general pipeline, trim-ready SSI curves can now be gap-closed and promoted into staged split records or closed inner trim loops, staged split faces can be classified into local Boolean keep/discard decisions, and supported boundary-to-boundary splits can be promoted into healed trim regions, triangulated, sewn, and submitted to the half-edge validator. Broad multi-split region graphs, coincident overlaps, and automatic classification of unsplit faces are still outside the supported scope.

The intersection module has exact analytic line/plane and plane/plane routines, plus marching/bracketing routines for NURBS cases. The NURBS/NURBS SSI path emits trim-ready `EdgeCurve3D` plus two `TrimCurve2D` p-curves, and the topology layer can install those curves as staged face splits or closed inner trim loops. It is not yet a full CAD face-intersection engine: coplanar overlap classification, coincident face merging, multi-split region graphs, and global shell rewrites are the next major layers.

The predicates are conservative interval filters. When a determinant cannot be certified, the API returns `Uncertain`; it does not silently trust an unstable sign.

## Roadmap

- Broaden Euler operators beyond the current `MVFS`, `MEV`, and `MEF` construction layer.
- Promote broader staged face-split graphs into healed trim loops and rewritten shell topology.
- Extend NURBS/NURBS SSI with coplanar overlap classification and higher-order curve fitting.
- Generalize healed Boolean output to multi-split region graphs, unsplit-face classification, and coincident overlap cases.
- Add viewer overlays for topology, normals, residuals, and golden-reference inspection.

## Repository Map

- `src/topology.rs` - half-edge B-rep data structure and validation
- `src/euler.rs` - Euler construction operators
- `src/nurbs.rs` - NURBS curves/surfaces, derivatives, knot insertion
- `src/predicates.rs` - interval-filtered robust predicates
- `src/intersection.rs` - curve/surface and surface/surface intersection routines
- `src/boolean.rs` - supported boolean pipeline
- `src/features.rs` - natural-language to parametric-feature tree layer
- `src/tessellation.rs` - CPU tessellation
- `src/wasm.rs` - raw WASM exports
- `src/bin/native_viewer.rs` - native `wgpu`/`winit` viewer binary
- `assets/shaders/nurbs_tessellate.wgsl` - WebGPU compute tessellator
- `web/` - browser viewer
- `tests/` - executable regression tests
- `docs/EulerOperators.md` - Euler operator scope and invariants
- `docs/FeatureLayer.md` - prompt-to-feature-tree design and dependency choices
- `corpus/regression/` - text corpus for bug and degeneracy cases
- `corpus/reference/v1/` - versioned reference models and golden outputs

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New geometry behavior should include focused tests and, when appropriate, corpus or golden-file coverage.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
