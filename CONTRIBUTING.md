# Contributing

Thanks for taking a look. This is a compact research/prototype B-rep kernel, so contributions are most useful when they keep the mathematical scope explicit and add tests for new behavior.

## Development Setup

Install stable Rust and the WASM target:

```sh
rustup toolchain install stable
rustup target add wasm32-unknown-unknown
```

Run the main quality gates:

```sh
cargo fmt -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo build --target wasm32-unknown-unknown --release
cargo check --features native-viewer --bin native-viewer
```

## Contribution Guidelines

- Keep geometric algorithms honest about their scope and tolerances.
- Add regression tests for every new degeneracy or supported modeling case.
- Prefer small, focused changes over broad rewrites.
- Do not remove golden-file assertions unless the reference model change is intentional and documented.
- Update `README.md` and `docs/` when behavior or supported scope changes.

## Public API Discipline

Prefer adding application-facing API through `brep_kernel::api` and `brep_kernel::prelude`.
Direct subsystem modules are public for kernel engineering, but the facade is the
compatibility boundary described in `docs/PublicApi.md`.

When changing the facade:

- Add or update tests in `tests/public_api.rs`.
- Update `CHANGELOG.md`.
- Increment `API_REVISION` when the curated facade changes materially.
- Increment `WASM_ABI_REVISION` when raw browser ABI signatures, buffer layout, or units change.
- Do not increase the MSRV without a minor-version rationale and changelog note.

## Reference Outputs

If a reference model intentionally changes, regenerate and inspect the golden values:

```sh
cargo test --test reference_models dump_reference_golden_outputs -- --ignored --nocapture
```

Then update the matching files under `corpus/reference/v1`.
