# Natural-Language Feature Layer

The thin feature layer lives in `src/features.rs`.

It supports prompts such as:

```text
10mm bracket with two M4 holes
60x20x10mm plate with 2 M4 holes
```

The parser emits an explicit feature tree:

- root: rectangular base plate
- children: vertical through-hole features

Execution converts the feature tree into a B-rep by triangulating a 2D plate profile with holes, extruding it to top/bottom faces, adding side walls, and then passing the mesh through the half-edge topology validator.

## Why Use A Library?

We should not reinvent every layer. A CAD kernel portfolio piece should show judgment about which parts are core research/engineering and which parts are commodity infrastructure.

This layer uses `earcutr`, a Rust port of Mapbox Earcut, for polygon-with-holes triangulation. That is a good dependency because:

- polygon triangulation is not the interesting differentiator for this repo
- the triangulation output is still validated by the kernel
- the golden-reference corpus pins the resulting mesh hash and topology counts
- keeping the natural-language parser deterministic makes tests explainable

In a production version, this same boundary could lean on larger libraries:

- a grammar parser such as `pest` or `nom`
- a constraint solver for sketches
- a real CAD interchange layer for standards and units
- an LLM or small classifier that proposes feature trees, followed by deterministic validation

The important boundary is that learned or heuristic layers should emit explicit feature trees. The kernel should execute and validate geometry; it should not silently trust vague text.

## Current Defaults

`M4` is interpreted as a normal clearance through-hole with a 4.5 mm diameter.

If no full `LxWxT` size is provided, the parser treats the first `Nmm` value as plate thickness and derives a conservative default plate length/width from the hole count and diameter.
