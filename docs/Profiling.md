# Benchmarks And Profiling

The repository includes a dependency-light benchmark harness:

```sh
cargo bench --bench kernel_bench --locked
```

It times representative kernel paths:

- closed-solid validation
- cube-minus-cylinder boolean generation
- faceted STEP export/import round-trip
- prompt-to-feature execution
- CPU NURBS surface tessellation

The harness is intentionally simple so it runs on stable Rust and can be checked
in CI. Treat it as a smoke benchmark and profiling entry point, not a substitute
for a statistically rigorous Criterion suite.

## CI Compile Check

CI verifies the benchmark target compiles:

```sh
cargo check --benches --locked
```

For local release-code profiling, run:

```sh
cargo bench --bench kernel_bench --locked
```

## macOS Instruments

Build the benchmark binary without running it:

```sh
cargo bench --bench kernel_bench --no-run --locked
```

Then open the emitted binary under `target/release/deps/` in Instruments with
the Time Profiler template.

## flamegraph

If `cargo-flamegraph` is installed:

```sh
cargo flamegraph --bench kernel_bench
```

On Linux, this usually requires `perf` permissions. On macOS, the command uses
DTrace and may require elevated permissions depending on the machine policy.

## What To Profile Next

Useful next profiling targets are:

- SSI sampling and NURBS residual refinement
- trim-loop nesting and point-in-loop classification
- tolerance-aware sewing for large triangle shells
- boolean face-pair graph construction
- WebGPU tessellation dispatch and readback costs

When adding a new benchmark, prefer a fixed deterministic model and print a
stable guard value so accidental dead-code elimination is visible.
