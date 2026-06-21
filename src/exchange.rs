//! STEP and IGES exchange helpers.
//!
//! This module implements a deliberately scoped faceted B-rep exchange layer.
//! STEP export/import uses a small ISO-10303-21 subset around
//! `FACETED_BREP`, `CLOSED_SHELL`, `FACE`, `FACE_OUTER_BOUND`,
//! `POLY_LOOP`, and `CARTESIAN_POINT`. IGES export/import uses a compact
//! faceted subset with point records and triangle records. These routines are
//! suitable for deterministic regression interchange of the current triangle
//! B-rep solids; they are not a general STEP/IGES translator.

use crate::errors::{KernelEntityRef, KernelError, KernelErrorKind, KernelResult, KernelSubsystem};
use crate::math::Point3;
use crate::topology::Solid;
use std::collections::{BTreeMap, HashMap};

/// Supported exchange format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExchangeFormat {
    /// STEP Part 21 faceted B-rep subset.
    Step,
    /// Compact IGES faceted subset.
    Iges,
}

/// Export a solid as a STEP Part 21 faceted B-rep subset.
pub fn export_step_faceted_brep(solid: &Solid, product_name: &str) -> KernelResult<String> {
    solid.validate().map_err(|error| {
        KernelError::from(error)
            .with_operation("export_step_faceted_brep")
            .with_note("solid must validate before exchange export")
    })?;

    let safe_name = step_string(product_name);
    let mut next_id = 1usize;
    let mut point_ids = Vec::with_capacity(solid.vertices.len());
    let mut data = Vec::<String>::new();
    for vertex in &solid.vertices {
        let id = next_id;
        next_id += 1;
        point_ids.push(id);
        data.push(format!(
            "#{id}=CARTESIAN_POINT('',({}, {}, {}));",
            format_float(vertex.point.x),
            format_float(vertex.point.y),
            format_float(vertex.point.z)
        ));
    }

    let mut face_ids = Vec::<usize>::with_capacity(solid.faces.len());
    for (face_id, triangle) in solid.triangles().into_iter().enumerate() {
        let loop_id = next_id;
        next_id += 1;
        let bound_id = next_id;
        next_id += 1;
        let step_face_id = next_id;
        next_id += 1;
        data.push(format!(
            "#{loop_id}=POLY_LOOP('',(#{},#{},#{}));",
            point_ids[triangle[0]], point_ids[triangle[1]], point_ids[triangle[2]]
        ));
        data.push(format!("#{bound_id}=FACE_OUTER_BOUND('',#{loop_id},.T.);"));
        data.push(format!("#{step_face_id}=FACE('F{face_id}',(#{bound_id}));"));
        face_ids.push(step_face_id);
    }

    let shell_id = next_id;
    next_id += 1;
    let brep_id = next_id;
    data.push(format!(
        "#{shell_id}=CLOSED_SHELL('',({}));",
        face_ids
            .iter()
            .map(|id| format!("#{id}"))
            .collect::<Vec<_>>()
            .join(",")
    ));
    data.push(format!(
        "#{brep_id}=FACETED_BREP('{safe_name}',#{shell_id});"
    ));

    let mut output = String::new();
    output.push_str("ISO-10303-21;\n");
    output.push_str("HEADER;\n");
    output.push_str("FILE_DESCRIPTION(('brep-kernel faceted B-rep subset'),'2;1');\n");
    output.push_str(&format!(
        "FILE_NAME('{safe_name}','',('brep-kernel'),('brep-kernel'),'brep-kernel','brep-kernel','');\n"
    ));
    output.push_str("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n");
    output.push_str("ENDSEC;\n");
    output.push_str("DATA;\n");
    for entity in data {
        output.push_str(&entity);
        output.push('\n');
    }
    output.push_str("ENDSEC;\n");
    output.push_str("END-ISO-10303-21;\n");
    Ok(output)
}

/// Import a solid from the STEP faceted B-rep subset emitted by this crate.
pub fn import_step_faceted_brep(input: &str) -> KernelResult<Solid> {
    let entities = parse_step_entities(input)?;
    let mut points = BTreeMap::<usize, Point3>::new();
    let mut loops = HashMap::<usize, Vec<usize>>::new();
    let mut bounds = HashMap::<usize, usize>::new();
    let mut face_bounds = Vec::<usize>::new();

    for (id, body) in &entities {
        let upper = body.to_ascii_uppercase();
        if upper.starts_with("CARTESIAN_POINT") {
            points.insert(
                *id,
                parse_step_point(body).map_err(|error| {
                    error
                        .with_operation("import_step_faceted_brep")
                        .with_entity(KernelEntityRef::ExchangeEntity {
                            format: "STEP".to_owned(),
                            id: *id,
                        })
                })?,
            );
        } else if upper.starts_with("POLY_LOOP") {
            loops.insert(*id, parse_step_refs(body));
        } else if upper.starts_with("FACE_OUTER_BOUND") {
            let refs = parse_step_refs(body);
            let Some(loop_id) = refs.first().copied() else {
                return Err(exchange_parse_error(
                    "STEP face bound is missing its poly-loop reference",
                )
                .with_operation("import_step_faceted_brep")
                .with_entity(KernelEntityRef::ExchangeEntity {
                    format: "STEP".to_owned(),
                    id: *id,
                }));
            };
            bounds.insert(*id, loop_id);
        } else if upper.starts_with("FACE(") {
            let refs = parse_step_refs(body);
            let Some(bound_id) = refs.first().copied() else {
                return Err(
                    exchange_parse_error("STEP face is missing its outer-bound reference")
                        .with_operation("import_step_faceted_brep")
                        .with_entity(KernelEntityRef::ExchangeEntity {
                            format: "STEP".to_owned(),
                            id: *id,
                        }),
                );
            };
            face_bounds.push(bound_id);
        }
    }

    if points.is_empty() || face_bounds.is_empty() {
        return Err(
            exchange_parse_error("STEP input did not contain points and faces")
                .with_operation("import_step_faceted_brep"),
        );
    }

    let mut point_index = HashMap::<usize, usize>::new();
    let mut solid_points = Vec::<Point3>::with_capacity(points.len());
    for (step_id, point) in points {
        point_index.insert(step_id, solid_points.len());
        solid_points.push(point);
    }

    let mut triangles = Vec::<[usize; 3]>::new();
    for bound_id in face_bounds {
        let loop_id = *bounds.get(&bound_id).ok_or_else(|| {
            exchange_parse_error("STEP face references a missing face bound")
                .with_operation("import_step_faceted_brep")
                .with_entity(KernelEntityRef::ExchangeEntity {
                    format: "STEP".to_owned(),
                    id: bound_id,
                })
        })?;
        let point_refs = loops.get(&loop_id).ok_or_else(|| {
            exchange_parse_error("STEP face bound references a missing poly-loop")
                .with_operation("import_step_faceted_brep")
                .with_entity(KernelEntityRef::ExchangeEntity {
                    format: "STEP".to_owned(),
                    id: loop_id,
                })
        })?;
        let loop_vertices = normalize_loop_refs(point_refs);
        if loop_vertices.len() < 3 {
            return Err(
                exchange_parse_error("STEP poly-loop has fewer than three vertices")
                    .with_operation("import_step_faceted_brep")
                    .with_entity(KernelEntityRef::ExchangeEntity {
                        format: "STEP".to_owned(),
                        id: loop_id,
                    }),
            );
        }
        let mut vertex_indices = Vec::<usize>::with_capacity(loop_vertices.len());
        for point_ref in loop_vertices {
            let index = *point_index.get(point_ref).ok_or_else(|| {
                exchange_parse_error("STEP poly-loop references a missing point")
                    .with_operation("import_step_faceted_brep")
                    .with_entity(KernelEntityRef::ExchangeEntity {
                        format: "STEP".to_owned(),
                        id: *point_ref,
                    })
            })?;
            vertex_indices.push(index);
        }
        for index in 1..vertex_indices.len() - 1 {
            triangles.push([
                vertex_indices[0],
                vertex_indices[index],
                vertex_indices[index + 1],
            ]);
        }
    }

    Solid::from_triangle_mesh(solid_points, &triangles).map_err(|error| {
        KernelError::from(error)
            .with_operation("import_step_faceted_brep")
            .with_note("STEP geometry was parsed but did not validate as a closed solid")
    })
}

/// Export a solid as a compact IGES faceted subset.
pub fn export_iges_faceted_brep(solid: &Solid, product_name: &str) -> KernelResult<String> {
    solid.validate().map_err(|error| {
        KernelError::from(error)
            .with_operation("export_iges_faceted_brep")
            .with_note("solid must validate before exchange export")
    })?;

    let mut output = String::new();
    output.push_str("BREP_KERNEL_IGES_FACETED_SUBSET,1;S      1\n");
    output.push_str(&format!("PRODUCT,{};G      1\n", iges_token(product_name)));
    for (index, vertex) in solid.vertices.iter().enumerate() {
        output.push_str(&format!(
            "116,{},{},{},{};P{:>7}\n",
            index + 1,
            format_float(vertex.point.x),
            format_float(vertex.point.y),
            format_float(vertex.point.z),
            index + 1
        ));
    }
    let face_offset = solid.vertices.len();
    for (index, triangle) in solid.triangles().into_iter().enumerate() {
        output.push_str(&format!(
            "106,{},{},{},{};P{:>7}\n",
            index + 1,
            triangle[0] + 1,
            triangle[1] + 1,
            triangle[2] + 1,
            face_offset + index + 1
        ));
    }
    output.push_str(&format!(
        "T,{},{},0;T      1\n",
        solid.vertices.len(),
        solid.faces.len()
    ));
    Ok(output)
}

/// Import a solid from the compact IGES faceted subset emitted by this crate.
pub fn import_iges_faceted_brep(input: &str) -> KernelResult<Solid> {
    let mut points = BTreeMap::<usize, Point3>::new();
    let mut faces = Vec::<[usize; 3]>::new();

    for (line_number, line) in input.lines().enumerate() {
        let record = line.split(';').next().unwrap_or("").trim();
        if record.is_empty() {
            continue;
        }
        let fields: Vec<&str> = record.split(',').map(str::trim).collect();
        match fields.first().copied() {
            Some("116") => {
                if fields.len() != 5 {
                    return Err(exchange_parse_error("IGES point record must have id,x,y,z")
                        .with_operation("import_iges_faceted_brep")
                        .with_note(format!("line {}", line_number + 1)));
                }
                let id = parse_usize(fields[1], "IGES point id")?;
                let x = parse_f64(fields[2], "IGES point x")?;
                let y = parse_f64(fields[3], "IGES point y")?;
                let z = parse_f64(fields[4], "IGES point z")?;
                points.insert(id, Point3::new(x, y, z));
            }
            Some("106") => {
                if fields.len() != 5 {
                    return Err(exchange_parse_error(
                        "IGES triangle record must have id,vertex_a,vertex_b,vertex_c",
                    )
                    .with_operation("import_iges_faceted_brep")
                    .with_note(format!("line {}", line_number + 1)));
                }
                let a = parse_usize(fields[2], "IGES triangle vertex a")?;
                let b = parse_usize(fields[3], "IGES triangle vertex b")?;
                let c = parse_usize(fields[4], "IGES triangle vertex c")?;
                faces.push([a, b, c]);
            }
            _ => {}
        }
    }

    if points.is_empty() || faces.is_empty() {
        return Err(
            exchange_parse_error("IGES input did not contain point and triangle records")
                .with_operation("import_iges_faceted_brep"),
        );
    }

    let mut point_index = HashMap::<usize, usize>::new();
    let mut solid_points = Vec::<Point3>::with_capacity(points.len());
    for (iges_id, point) in points {
        point_index.insert(iges_id, solid_points.len());
        solid_points.push(point);
    }

    let mut triangles = Vec::<[usize; 3]>::with_capacity(faces.len());
    for face in faces {
        let a = *point_index.get(&face[0]).ok_or_else(|| {
            exchange_parse_error("IGES triangle references a missing point")
                .with_operation("import_iges_faceted_brep")
        })?;
        let b = *point_index.get(&face[1]).ok_or_else(|| {
            exchange_parse_error("IGES triangle references a missing point")
                .with_operation("import_iges_faceted_brep")
        })?;
        let c = *point_index.get(&face[2]).ok_or_else(|| {
            exchange_parse_error("IGES triangle references a missing point")
                .with_operation("import_iges_faceted_brep")
        })?;
        triangles.push([a, b, c]);
    }

    Solid::from_triangle_mesh(solid_points, &triangles).map_err(|error| {
        KernelError::from(error)
            .with_operation("import_iges_faceted_brep")
            .with_note("IGES geometry was parsed but did not validate as a closed solid")
    })
}

fn parse_step_entities(input: &str) -> KernelResult<BTreeMap<usize, String>> {
    let upper = input.to_ascii_uppercase();
    let data_start = upper.find("DATA;").ok_or_else(|| {
        exchange_parse_error("STEP input is missing DATA section")
            .with_operation("import_step_faceted_brep")
    })?;
    let data_end = upper[data_start..]
        .find("ENDSEC;")
        .map(|offset| data_start + offset)
        .ok_or_else(|| {
            exchange_parse_error("STEP input is missing DATA ENDSEC")
                .with_operation("import_step_faceted_brep")
        })?;
    let data = &input[data_start + "DATA;".len()..data_end];
    let mut entities = BTreeMap::<usize, String>::new();
    for chunk in data.split(';') {
        let entity = chunk.trim();
        if entity.is_empty() {
            continue;
        }
        let Some(rest) = entity.strip_prefix('#') else {
            continue;
        };
        let Some((id_text, body)) = rest.split_once('=') else {
            return Err(exchange_parse_error("STEP entity is missing `=`")
                .with_operation("import_step_faceted_brep"));
        };
        let id = parse_usize(id_text.trim(), "STEP entity id")?;
        entities.insert(id, body.trim().to_owned());
    }
    if entities.is_empty() {
        return Err(
            exchange_parse_error("STEP DATA section contains no entities")
                .with_operation("import_step_faceted_brep"),
        );
    }
    Ok(entities)
}

fn parse_step_point(body: &str) -> KernelResult<Point3> {
    let start = body
        .rfind('(')
        .ok_or_else(|| exchange_parse_error("STEP point is missing coordinate tuple"))?;
    let end = body[start..]
        .find(')')
        .map(|offset| start + offset)
        .ok_or_else(|| exchange_parse_error("STEP point coordinate tuple is not closed"))?;
    let coords: Vec<&str> = body[start + 1..end].split(',').map(str::trim).collect();
    if coords.len() != 3 {
        return Err(exchange_parse_error(
            "STEP point must contain exactly three coordinates",
        ));
    }
    Ok(Point3::new(
        parse_f64(coords[0], "STEP point x")?,
        parse_f64(coords[1], "STEP point y")?,
        parse_f64(coords[2], "STEP point z")?,
    ))
}

fn parse_step_refs(body: &str) -> Vec<usize> {
    let mut refs = Vec::new();
    let mut chars = body.char_indices().peekable();
    while let Some((_, ch)) = chars.next() {
        if ch != '#' {
            continue;
        }
        let mut digits = String::new();
        while let Some((_, next)) = chars.peek().copied() {
            if next.is_ascii_digit() {
                digits.push(next);
                chars.next();
            } else {
                break;
            }
        }
        if let Ok(id) = digits.parse::<usize>() {
            refs.push(id);
        }
    }
    refs
}

fn normalize_loop_refs(point_refs: &[usize]) -> &[usize] {
    if point_refs.len() >= 2 && point_refs.first() == point_refs.last() {
        &point_refs[..point_refs.len() - 1]
    } else {
        point_refs
    }
}

fn parse_usize(text: &str, field: &str) -> KernelResult<usize> {
    text.parse::<usize>().map_err(|error| {
        exchange_parse_error(format!("could not parse {field} as unsigned integer"))
            .with_source(error.to_string())
    })
}

fn parse_f64(text: &str, field: &str) -> KernelResult<f64> {
    let value = text.parse::<f64>().map_err(|error| {
        exchange_parse_error(format!("could not parse {field} as floating-point value"))
            .with_source(error.to_string())
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(exchange_parse_error(format!("{field} must be finite")))
    }
}

fn exchange_parse_error(message: impl Into<String>) -> KernelError {
    KernelError::new(
        KernelSubsystem::Exchange,
        KernelErrorKind::Parse,
        "exchange.parse",
        message,
    )
}

fn step_string(value: &str) -> String {
    value.replace('\'', "''")
}

fn iges_token(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            ',' | ';' => '_',
            _ => ch,
        })
        .collect()
}

fn format_float(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.17}")
    }
}
