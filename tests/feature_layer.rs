use brep_kernel::features::{
    build_plate_with_holes, parse_feature_prompt, BasePlate, FeatureError, FeatureOperation,
    ThroughHole,
};

#[test]
fn parses_bracket_prompt_into_feature_tree() {
    let tree = parse_feature_prompt("10mm bracket with two M4 holes").unwrap();
    let FeatureOperation::BasePlate(plate) = tree.root.operation else {
        panic!("expected base plate");
    };
    assert_eq!(plate.thickness_mm, 10.0);
    assert_eq!(tree.root.children.len(), 2);

    for child in &tree.root.children {
        let FeatureOperation::ThroughHole(hole) = &child.operation else {
            panic!("expected through hole");
        };
        assert_eq!(hole.standard.as_deref(), Some("M4"));
        assert_eq!(hole.diameter_mm, 4.5);
    }
}

#[test]
fn executes_prompt_as_closed_genus_two_brep() {
    let tree = parse_feature_prompt("10mm bracket with two M4 holes").unwrap();
    let solid = tree.execute().unwrap();
    let counts = solid.topology_counts();
    solid.validate().unwrap();

    assert_eq!(counts.boundary_halfedges, 0);
    assert_eq!(counts.halfedges, counts.edges * 2);
    assert_eq!(solid.euler_characteristic(), -2);
    assert_eq!(solid.genus(), Some(2));
    assert!(solid.volume() > 0.0);
    assert!(solid.surface_area() > 0.0);
}

#[test]
fn parses_explicit_plate_dimensions() {
    let tree = parse_feature_prompt("60x20x10mm plate with 2 M4 holes").unwrap();
    let FeatureOperation::BasePlate(plate) = tree.root.operation else {
        panic!("expected base plate");
    };
    assert_eq!(plate.length_mm, 60.0);
    assert_eq!(plate.width_mm, 20.0);
    assert_eq!(plate.thickness_mm, 10.0);
    assert_eq!(tree.root.children.len(), 2);
}

#[test]
fn rejects_overlapping_holes() {
    let plate = BasePlate {
        length_mm: 40.0,
        width_mm: 20.0,
        thickness_mm: 4.0,
    };
    let holes = vec![
        ThroughHole {
            x_mm: -1.0,
            y_mm: 0.0,
            diameter_mm: 6.0,
            segments: 24,
            standard: None,
        },
        ThroughHole {
            x_mm: 1.0,
            y_mm: 0.0,
            diameter_mm: 6.0,
            segments: 24,
            standard: None,
        },
    ];

    assert!(matches!(
        build_plate_with_holes(plate, &holes),
        Err(FeatureError::InvalidFeature("holes overlap"))
    ));
}
