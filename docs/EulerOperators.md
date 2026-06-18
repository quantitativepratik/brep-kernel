# Euler Operators

The Euler operator layer lives in `src/euler.rs`.

It is a mutable construction layer above the validated half-edge `Solid` type. The builder tracks polygonal face loops and supports:

- `MVFS`: make vertex, face, shell
- `MEV`: make edge and vertex
- `MEF`: make edge and face

These are the constructive make-side operators most useful for building and testing topology incrementally. Each operation preserves the expected Euler characteristic for a single shell:

```text
V - E + F = 2S
```

For a single shell, `S = 1`, so the count invariant remains `2` after `MVFS`, `MEV`, and `MEF`.

## Relationship To Half-Edge Solids

The Euler builder is not a second final topology type. It is a construction surface.

When a construction is ready, `EulerBuilder::to_solid()` triangulates the construction face loops and calls `Solid::from_triangle_mesh`. That means the final result must still satisfy the same closed-manifold half-edge checks as every other kernel output:

- every directed half-edge has a twin
- `next` and `prev` links are consistent
- no boundary half-edges remain
- Euler/genus invariants can be computed on the final shell

This keeps the architecture honest: Euler operators build topology, and the half-edge kernel validates topology.

## Current Scope

Implemented:

- `MVFS`
- `MEV`
- `MEF`
- count snapshots
- Euler invariant checks
- conversion to validated half-edge solids
- proptest coverage over generated triangle-sheet constructions

Not yet implemented:

- inverse kill operators such as `KEV` and `KEF`
- multi-shell Euler operators
- ring/hole operators such as `KEMR`
- full non-triangular face retention in the final `Solid`

Those are natural next additions if the kernel needs a full classic Euler-operator editing API rather than a constructive foundation.
