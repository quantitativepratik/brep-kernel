# Tutorials

These tutorials are intentionally short and executable. Each command below is
backed by a checked Rust example or test target.

## 1. Inspect A Closed Solid

Run the exchange example:

```sh
cargo run --example exchange_roundtrip
```

This builds a cube, exports it to the repository's faceted STEP and IGES
subsets, imports both files again, and prints topology counts, volume, surface
area, stable mesh hashes, and exchange payload sizes.

The key invariant is that the imported solids validate as closed half-edge
shells and preserve the stable mesh hash:

```text
original hash: ...
STEP hash:     ...
IGES hash:     ...
```

## 2. Use Structured Diagnostics

Run:

```sh
cargo run --example boolean_diagnostics
```

The example intentionally asks for an invalid cube-minus-cylinder subtraction.
Instead of panicking or returning an untyped string, the kernel produces a
structured diagnostic with:

- subsystem
- broad error kind
- stable diagnostic code
- operation name
- lower-level source text
- optional notes

This is the path new public APIs should use when they need to expose unsupported
or numerically ambiguous cases.

## 3. Mutate Topology Transactionally

Run:

```sh
cargo run --example transactional_topology
```

The example creates a cube, edits vertex and face tolerances inside a
`TopologyTransaction`, commits the edit, then performs a second edit and rolls
it back. The transaction report records revision movement and rollback entries.

This is the pattern for future topology rewrites: stage a mutation, validate the
solid, commit only when invariants hold, and retain enough log data to diagnose
what changed.

## 4. Exercise The Reference Corpus

Run:

```sh
cargo test --test reference_models
cargo test --test regression_corpus
```

Reference models lock down stable mesh hashes, topology counts, volumes, and
areas. Regression corpus tests capture named edge cases so behavior changes are
explicit.

When a geometry change intentionally updates a golden value, print the proposed
outputs with:

```sh
cargo test --test reference_models dump_reference_golden_outputs -- --ignored --nocapture
```

Review the diffs carefully before committing a new golden file.

## 5. Open The Viewers

Native:

```sh
cargo run --features native-viewer --bin native-viewer
```

Browser:

```sh
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
python3 -m http.server 8080
```

Open `http://localhost:8080/web/`.

The native and browser viewers are not modeling applications yet. They exist to
prove the kernel can expose tessellated geometry through desktop `wgpu`, WASM,
and WebGPU-facing assets.
