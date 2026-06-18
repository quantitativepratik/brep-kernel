# Kernel Architecture

The kernel is arranged around a small set of boundaries.

## Topology

`src/topology.rs` owns the half-edge structure:

- `Vertex`
- `HalfEdge`
- `Edge`
- `Face`
- `Shell`
- `Solid`

The constructor builds twin adjacency from indexed triangles and rejects boundary, duplicate, and nonmanifold directed edges. The validator checks `next`, `prev`, and `twin` consistency. Euler characteristic and genus are derived from the resulting graph.

Edges now carry an `EdgeCurve3D` support curve in model space. Supported edge curves are line segments, circular arcs, NURBS curves, polylines, and unresolved placeholders. Edge validation checks that explicit curve endpoints match the topological edge vertices within tolerance.

Faces carry an analytic `FaceSurface` support tag and ordered trim loops. Supported face surfaces are planes, Z-aligned cylinders, NURBS surfaces, and faceted fallbacks. Each `TrimLoop` is either outer or inner, contains ordered trims, and each trim can carry a 2D p-curve in that face's parameter domain. A single topological edge can therefore have one 3D curve and two distinct p-curves, one for each adjacent face. Triangle-mesh construction automatically gives every edge a 3D line segment and every planar face one projected outer trim loop.

This is a topology representation layer, not yet a full trimming engine. The next layers are face splitting, fitted trim-curve creation from SSI output, coedge-level tolerance metadata, and healing/sewing.

`src/euler.rs` provides the constructive Euler-operator layer above this final topology:

- `MVFS`
- `MEV`
- `MEF`

The Euler builder tracks polygonal construction loops and count invariants, then triangulates into `Solid::from_triangle_mesh` for final half-edge validation. This keeps incremental topological construction separate from the closed-manifold representation used by booleans, golden files, and rendering.

## Geometry

`src/geometry.rs` contains analytic primitives. `src/nurbs.rs` contains rational B-spline curves and tensor-product surfaces.

NURBS evaluation is done in homogeneous coordinates. First derivatives use basis derivatives and the rational quotient rule. Surface normals are computed from `du x dv`.

Knot insertion is implemented for curves as a shape-preserving refinement operation.

## Robustness

`src/predicates.rs` is the numerical gate. Code that wants to make topology-changing decisions should use these signs rather than raw determinants.

The current filter can certify many cases and explicitly reports uncertainty for degenerate ones. That is enough to prevent silent misclassification in the implemented operations.

## Intersections

`src/intersection.rs` includes exact analytic intersections for linear primitives and marching/bracketing routines for NURBS cases.

The plane/NURBS-surface routine marches over parameter-space cells and emits short polyline segments. The NURBS/NURBS surface routine takes the next step: it tessellates both parametric surfaces for discovery, intersects candidate triangle pairs, then refines segment endpoints against the original NURBS evaluations with a small Gauss-Newton residual solve. It still intentionally stops before trim-curve fitting, coplanar overlap handling, and topology insertion.

## Booleans

`src/boolean.rs` implements cube-minus-cylinder as a supported analytic boolean. It emits:

- top and bottom square annuli
- inner cylindrical wall with subtraction-facing normals
- outer cube side walls

The result is fed through the same half-edge constructor as every other solid. The regression test asserts closure and genus.

## GPU And Browser

`assets/shaders/nurbs_tessellate.wgsl` evaluates a rational cubic patch in a compute shader. `web/app.js` dispatches the compute pass, renders the resulting vertex buffer, and can load the Rust-generated WASM boolean mesh.

`src/bin/native_viewer.rs` is a feature-gated native `wgpu`/`winit` viewer. It builds a rational bicubic NURBS patch through the kernel, tessellates it on the CPU, uploads vertex/index buffers, and renders it with a depth buffer and rotating camera. The binary is opt-in through the `native-viewer` feature so the core library and WASM target do not inherit desktop windowing dependencies.
