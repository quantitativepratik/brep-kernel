# Reference Models And Golden Outputs

Reference models live under `corpus/reference/v1`.

Each model has two files:

- `*.model` describes the construction input.
- `*.golden` locks the expected output.

Model inputs can be direct analytic/kernel constructors, or higher-level feature prompts such as `kind=feature_prompt`.

The current golden fields are:

- `mesh_hash`: stable FNV-1a hash of quantized vertex coordinates and triangle indices.
- `vertices`, `edges`, `halfedges`, `faces`, `shells`, `triangles`.
- `boundary_halfedges`.
- `euler`, `genus`.
- `volume`, `surface_area`.
- `tolerance` for floating invariant comparisons.

The hash is intentionally order-sensitive. If a boolean or tessellation change emits the same shape with a different vertex or face order, the reference test fails. That is useful for kernel work because downstream trimming, serialization, and browser interop often depend on deterministic output.

The `nl_bracket_two_m4` reference pins the natural-language feature layer: the prompt is parsed into a feature tree, executed into a B-rep, then checked like any other model.

Run:

```sh
cargo test --test reference_models
```

After an intentional mesh-output change, inspect new values with:

```sh
cargo test --test reference_models dump_reference_golden_outputs -- --ignored --nocapture
```

Only update golden files when the geometry/topology change is intended and the invariants still make sense.

## Property Tests

`tests/property_solids.rs` uses `proptest` to generate valid solids:

- random rectangular boxes
- random cube-minus-cylinder solids
- a mixed valid-solid strategy

For every generated solid, the tests assert:

- half-edge validation succeeds
- no boundary half-edges exist
- `halfedges == edges * 2`
- Euler characteristic matches the expected genus
- surface area is positive and finite
- volume is finite and nonnegative

The generators are intentionally conservative: they only emit solids known to be valid through the current kernel constructors. This keeps failures meaningful; a failing property should indicate a kernel regression, not a bad test generator.
