# Numerical Robustness Notes

Naive floating-point predicates fail in CAD kernels because topological decisions depend on signs that can be smaller than rounding error.

Examples:

- A nearly collinear trim edge can be classified on the wrong side of a face.
- Coincident or nearly coincident faces can flicker between overlap and disjoint classifications.
- A near-tangent curve/surface intersection can disappear if the solver only looks for sign changes.
- Boolean splitting can create nonmanifold sliver edges when two vertices that should be identical are classified differently.

This kernel uses interval-filtered predicates in `src/predicates.rs`.

Each arithmetic operation expands the result to adjacent representable `f64` values. If the final interval is strictly positive or strictly negative, the sign is certified. If it straddles zero, the result is `RobustSign::Uncertain`.

That behavior is intentionally conservative. An industrial kernel would normally continue from `Uncertain` into one of:

- Shewchuk-style adaptive expansion arithmetic for exact signs of floating inputs.
- Rational arithmetic for CAD data that originates in exact construction history.
- Symbolic perturbation to force consistent decisions across a whole arrangement.
- Broader topology-aware healing that merges analytic edges, vertices, and trim loops inside a model tolerance after certified classification. The current topology layer includes tolerance-aware triangle mesh sewing for shell candidates, but not yet full analytic edge/trim sewing.

The important rule is that uncertain signs must remain explicit. They are not ordinary negative or positive signs with a smaller epsilon.

## Where This Shows Up

- `orient2d` protects planar loop and triangle decisions.
- `orient3d` protects face-side and tetrahedral volume decisions.
- `incircle2d` protects local triangulation decisions.
- line-plane intersection checks a certified denominator before dividing.
- curve-plane and plane-surface intersections include tangency handling instead of only relying on sign changes.

The regression corpus includes cases that should be classified as uncertain instead of guessed.
