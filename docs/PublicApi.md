# Public API And Versioning Policy

This crate is still pre-1.0, but it now has an explicit public API discipline.
The goal is to let examples, downstream experiments, and browser integration use
a coherent facade while the lower-level CAD-kernel internals continue to evolve.

## API Tiers

### Stable Facade

Use:

```rust
use brep_kernel::prelude::*;
```

or import from:

```rust
brep_kernel::api
```

The facade re-exports the supported application-facing surface:

- math primitives
- structured diagnostics
- validated `Solid` construction and inspection
- topology tolerances and transactions
- NURBS evaluation and tessellation
- faceted STEP/IGES exchange
- deterministic feature parsing/execution
- the supported cube-minus-cylinder boolean reference case

Changes to this facade should be handled with semver discipline, documented in
`CHANGELOG.md`, and covered by tests in `tests/public_api.rs`.

### Expert Modules

Direct modules such as `brep_kernel::topology`, `brep_kernel::boolean`, and
`brep_kernel::intersection` remain public because this is a kernel project and
kernel engineers need access to the machinery. Until individual items are
promoted through `brep_kernel::api`, these modules are considered expert APIs:
useful, documented, tested, but still allowed to change as algorithms mature.

### Raw WASM ABI

The browser ABI is separate from the Rust facade. It is tracked by
`WASM_ABI_REVISION` in `brep_kernel::api` and by the exported `brep_version()`
function. Any incompatible change to raw WASM function signatures, buffer
layout, or units must increment that ABI revision and update the web viewer.

## SemVer Rules

While the crate is `0.x`:

- Patch releases must not intentionally break the `brep_kernel::api` facade.
- Minor releases may expand the facade and may make direct expert-module changes.
- Removing or renaming facade items requires either a deprecation period or a
  clearly documented pre-1.0 breaking minor release.
- Diagnostic codes are treated as compatibility surface. New codes can be added;
  existing codes should not be silently repurposed.
- Golden reference model changes must be documented when they reflect intentional
  geometry or topology behavior changes.

At `1.0`, all items in `brep_kernel::api` become semver-stable unless explicitly
marked otherwise.

## MSRV Policy

The minimum supported Rust version is exposed as
`MINIMUM_SUPPORTED_RUST_VERSION` and currently matches `Cargo.toml`:

```text
1.87
```

Increasing MSRV requires a minor version bump and a changelog entry.

## Feature Flags

The default library build stays dependency-light. Optional runtime surfaces are
feature-gated:

- `native-viewer`: enables the desktop `wgpu`/`winit` viewer binary

Adding a feature flag is non-breaking. Removing or changing the meaning of an
existing feature flag is a breaking API change.

## Release Checklist

Before tagging a release:

```sh
cargo fmt -- --check
cargo test --locked
cargo check --examples --locked
cargo check --benches --locked
cargo clippy --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
cargo build --target wasm32-unknown-unknown --release --locked
cargo check --features native-viewer --bin native-viewer --locked
```

Then review:

- `CHANGELOG.md` has an entry for facade, behavior, diagnostic-code, MSRV, and
  golden-output changes.
- `README.md` and `docs/` match the implemented scope.
- `tests/public_api.rs` covers any newly promoted facade item.
- Reference golden files changed only when the geometry/topology change is
  intentional.
- `API_REVISION` or `WASM_ABI_REVISION` was incremented when the relevant
  compatibility boundary changed.
