mod common;

use brep_kernel::boolean::subtract_cube_cylinder;
use brep_kernel::features::parse_feature_prompt;
use brep_kernel::topology::Solid;
use common::rectangular_box;
use std::collections::HashMap;

const REFERENCES: &[(&str, &str, &str)] = &[
    (
        "cube_2",
        include_str!("../corpus/reference/v1/cube_2.model"),
        include_str!("../corpus/reference/v1/cube_2.golden"),
    ),
    (
        "rectangular_box_1_25_2_5_0_75",
        include_str!("../corpus/reference/v1/rectangular_box_1_25_2_5_0_75.model"),
        include_str!("../corpus/reference/v1/rectangular_box_1_25_2_5_0_75.golden"),
    ),
    (
        "cube_minus_cylinder_64",
        include_str!("../corpus/reference/v1/cube_minus_cylinder_64.model"),
        include_str!("../corpus/reference/v1/cube_minus_cylinder_64.golden"),
    ),
    (
        "nl_bracket_two_m4",
        include_str!("../corpus/reference/v1/nl_bracket_two_m4.model"),
        include_str!("../corpus/reference/v1/nl_bracket_two_m4.golden"),
    ),
];

#[test]
fn reference_models_match_golden_outputs() {
    for (name, model, golden) in REFERENCES {
        let model = parse_kv(model);
        let golden = parse_kv(golden);
        let solid = build_model(&model);
        common::assert_closed_manifold(&solid);

        let counts = solid.topology_counts();
        assert_eq_field(
            name,
            "mesh_hash",
            format!("{:#018x}", solid.stable_mesh_hash()),
            &golden,
        );
        assert_eq_field(name, "vertices", counts.vertices.to_string(), &golden);
        assert_eq_field(name, "edges", counts.edges.to_string(), &golden);
        assert_eq_field(name, "halfedges", counts.halfedges.to_string(), &golden);
        assert_eq_field(name, "faces", counts.faces.to_string(), &golden);
        assert_eq_field(name, "shells", counts.shells.to_string(), &golden);
        assert_eq_field(name, "triangles", counts.triangles.to_string(), &golden);
        assert_eq_field(
            name,
            "boundary_halfedges",
            counts.boundary_halfedges.to_string(),
            &golden,
        );
        assert_eq_field(
            name,
            "euler",
            solid.euler_characteristic().to_string(),
            &golden,
        );
        assert_eq_field(
            name,
            "genus",
            solid.genus().expect("single-shell genus").to_string(),
            &golden,
        );

        let tolerance = parse_f64(&golden, "tolerance");
        assert_close(
            name,
            "volume",
            solid.volume(),
            parse_f64(&golden, "volume"),
            tolerance,
        );
        assert_close(
            name,
            "surface_area",
            solid.surface_area(),
            parse_f64(&golden, "surface_area"),
            tolerance,
        );
    }
}

#[test]
#[ignore = "prints current golden files for intentional reference-model updates"]
fn dump_reference_golden_outputs() {
    for (name, model, _) in REFERENCES {
        let model = parse_kv(model);
        let solid = build_model(&model);
        let counts = solid.topology_counts();
        println!("# {name}");
        println!("mesh_hash={:#018x}", solid.stable_mesh_hash());
        println!("vertices={}", counts.vertices);
        println!("edges={}", counts.edges);
        println!("halfedges={}", counts.halfedges);
        println!("faces={}", counts.faces);
        println!("shells={}", counts.shells);
        println!("triangles={}", counts.triangles);
        println!("boundary_halfedges={}", counts.boundary_halfedges);
        println!("euler={}", solid.euler_characteristic());
        println!("genus={}", solid.genus().expect("single-shell genus"));
        println!("volume={:.15}", solid.volume());
        println!("surface_area={:.15}", solid.surface_area());
        println!();
    }
}

fn build_model(model: &HashMap<String, String>) -> Solid {
    match required(model, "kind").as_str() {
        "cube" => Solid::cube(parse_f64(model, "size")).expect("valid cube reference"),
        "box" => rectangular_box(parse_vec3(model, "size")),
        "cube_minus_cylinder" => {
            let report = subtract_cube_cylinder(
                parse_f64(model, "cube_size"),
                parse_f64(model, "radius"),
                parse_usize(model, "segments"),
            )
            .expect("valid cube-minus-cylinder reference");
            report.solid
        }
        "feature_prompt" => {
            let tree = parse_feature_prompt(&required(model, "prompt"))
                .expect("valid feature prompt reference");
            tree.execute().expect("feature prompt executes")
        }
        kind => panic!("unsupported reference model kind: {kind}"),
    }
}

fn parse_kv(input: &str) -> HashMap<String, String> {
    input
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

fn required(map: &HashMap<String, String>, key: &str) -> String {
    map.get(key)
        .unwrap_or_else(|| panic!("missing key `{key}`"))
        .to_owned()
}

fn parse_f64(map: &HashMap<String, String>, key: &str) -> f64 {
    required(map, key).parse().expect("valid f64")
}

fn parse_usize(map: &HashMap<String, String>, key: &str) -> usize {
    required(map, key).parse().expect("valid usize")
}

fn parse_vec3(map: &HashMap<String, String>, key: &str) -> [f64; 3] {
    let raw = required(map, key);
    let values: Vec<f64> = raw
        .split(',')
        .map(|part| part.trim().parse().expect("valid vector component"))
        .collect();
    assert_eq!(values.len(), 3, "`{key}` must contain three values");
    [values[0], values[1], values[2]]
}

fn assert_eq_field(name: &str, field: &str, actual: String, expected: &HashMap<String, String>) {
    let expected = required(expected, field);
    assert_eq!(
        actual, expected,
        "{name}.{field}: actual value changed; update the golden only if the mesh change is intentional"
    );
}

fn assert_close(name: &str, field: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{name}.{field}: actual={actual:.15}, expected={expected:.15}, tolerance={tolerance}"
    );
}
