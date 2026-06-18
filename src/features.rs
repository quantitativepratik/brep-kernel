//! Thin natural-language to parametric-feature layer.
//!
//! The parser is intentionally small and deterministic. It recognizes bracket
//! and plate prompts such as `"10mm bracket with two M4 holes"` or
//! `"60x20x10mm plate with 2 M4 holes"`, emits an explicit feature tree, and
//! executes that tree by generating a validated half-edge solid.

use crate::math::{Point3, Vec3};
use crate::topology::{Solid, TopologyError};

/// Unit system for feature-tree dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Millimeters.
    Millimeter,
}

/// A parametric feature tree.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureTree {
    /// Original prompt, retained for traceability.
    pub source: String,
    /// Unit system.
    pub units: Unit,
    /// Root feature node.
    pub root: FeatureNode,
}

impl FeatureTree {
    /// Execute the feature tree into a validated B-rep solid.
    pub fn execute(&self) -> Result<Solid, FeatureError> {
        execute_feature_tree(self)
    }
}

/// A node in the feature tree.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureNode {
    /// Stable node id.
    pub id: String,
    /// Feature operation.
    pub operation: FeatureOperation,
    /// Child features applied to this node.
    pub children: Vec<FeatureNode>,
}

/// Supported parametric feature operations.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureOperation {
    /// Create the initial rectangular plate.
    BasePlate(BasePlate),
    /// Cut a vertical through-hole.
    ThroughHole(ThroughHole),
}

/// Rectangular base plate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasePlate {
    /// Length along X in millimeters.
    pub length_mm: f64,
    /// Width along Y in millimeters.
    pub width_mm: f64,
    /// Thickness along Z in millimeters.
    pub thickness_mm: f64,
}

/// Through-hole feature.
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughHole {
    /// X center in millimeters.
    pub x_mm: f64,
    /// Y center in millimeters.
    pub y_mm: f64,
    /// Hole diameter in millimeters.
    pub diameter_mm: f64,
    /// Polygon segments used to approximate the circle.
    pub segments: usize,
    /// Optional source standard, such as `M4`.
    pub standard: Option<String>,
}

/// Feature parser or execution error.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureError {
    /// Prompt is not recognized by the deterministic parser.
    UnrecognizedPrompt,
    /// A dimension is invalid.
    InvalidDimension(&'static str),
    /// A requested feature cannot be applied.
    InvalidFeature(&'static str),
    /// Polygon triangulation failed.
    Triangulation,
    /// Topology validation failed.
    Topology(TopologyError),
}

impl From<TopologyError> for FeatureError {
    fn from(value: TopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Parse a small natural-language bracket/plate prompt into a feature tree.
pub fn parse_feature_prompt(input: &str) -> Result<FeatureTree, FeatureError> {
    let normalized = normalize_prompt(input);
    if !normalized.contains("bracket") && !normalized.contains("plate") {
        return Err(FeatureError::UnrecognizedPrompt);
    }

    let dimensions = parse_dimensions(&normalized);
    let thickness = dimensions
        .map(|dims| dims[2])
        .or_else(|| parse_first_mm_value(&normalized))
        .unwrap_or(5.0);
    let hole_count = parse_hole_count(&normalized).unwrap_or(0);
    let hole_spec = parse_hole_spec(&normalized);

    let diameter = hole_spec
        .as_ref()
        .map(|spec| spec.clearance_diameter_mm)
        .or_else(|| parse_hole_diameter_mm(&normalized))
        .unwrap_or(4.5);

    let length = dimensions
        .map(|dims| dims[0])
        .unwrap_or_else(|| default_length_mm(hole_count, diameter));
    let width = dimensions
        .map(|dims| dims[1])
        .unwrap_or_else(|| default_width_mm(diameter));

    let plate = BasePlate {
        length_mm: length,
        width_mm: width,
        thickness_mm: thickness,
    };
    validate_plate(plate)?;

    let holes = distribute_holes(hole_count, diameter, hole_spec.as_ref(), plate)?;
    Ok(FeatureTree {
        source: input.to_owned(),
        units: Unit::Millimeter,
        root: FeatureNode {
            id: "base_plate".to_owned(),
            operation: FeatureOperation::BasePlate(plate),
            children: holes,
        },
    })
}

/// Execute a feature tree into a validated B-rep solid.
pub fn execute_feature_tree(tree: &FeatureTree) -> Result<Solid, FeatureError> {
    let FeatureOperation::BasePlate(plate) = &tree.root.operation else {
        return Err(FeatureError::InvalidFeature("root must be a base plate"));
    };
    let plate = *plate;

    let mut holes = Vec::<ThroughHole>::new();
    for child in &tree.root.children {
        match &child.operation {
            FeatureOperation::ThroughHole(hole) => holes.push(hole.clone()),
            FeatureOperation::BasePlate(_) => {
                return Err(FeatureError::InvalidFeature(
                    "base plates cannot be children",
                ));
            }
        }
    }
    build_plate_with_holes(plate, &holes)
}

/// Build a rectangular plate with vertical through-holes.
pub fn build_plate_with_holes(
    plate: BasePlate,
    holes: &[ThroughHole],
) -> Result<Solid, FeatureError> {
    validate_plate(plate)?;
    validate_holes(plate, holes)?;

    let mut rings = Vec::<Ring>::new();
    rings.push(Ring::outer_rectangle(plate.length_mm, plate.width_mm));
    for hole in holes {
        rings.push(Ring::hole(hole.clone()));
    }

    let mut flat = Vec::<f64>::new();
    let mut hole_indices = Vec::<usize>::new();
    let mut ranges = Vec::<std::ops::Range<usize>>::new();
    for (ring_id, ring) in rings.iter().enumerate() {
        if ring_id > 0 {
            hole_indices.push(flat.len() / 2);
        }
        let start = flat.len() / 2;
        for point in &ring.points {
            flat.push(point[0]);
            flat.push(point[1]);
        }
        let end = flat.len() / 2;
        ranges.push(start..end);
    }

    let top_tris =
        earcutr::earcut(&flat, &hole_indices, 2).map_err(|_| FeatureError::Triangulation)?;
    if top_tris.len() % 3 != 0 {
        return Err(FeatureError::Triangulation);
    }

    let h = plate.thickness_mm * 0.5;
    let vertex_count_2d = flat.len() / 2;
    let mut points = Vec::<Point3>::with_capacity(vertex_count_2d * 2);
    for i in 0..vertex_count_2d {
        points.push(Point3::new(flat[i * 2], flat[i * 2 + 1], h));
    }
    for i in 0..vertex_count_2d {
        points.push(Point3::new(flat[i * 2], flat[i * 2 + 1], -h));
    }

    let mut triangles = Vec::<[usize; 3]>::new();
    for chunk in top_tris.chunks_exact(3) {
        push_oriented(
            &points,
            &mut triangles,
            [chunk[0], chunk[1], chunk[2]],
            Vec3::new(0.0, 0.0, 1.0),
        );
        push_oriented(
            &points,
            &mut triangles,
            [
                chunk[0] + vertex_count_2d,
                chunk[1] + vertex_count_2d,
                chunk[2] + vertex_count_2d,
            ],
            Vec3::new(0.0, 0.0, -1.0),
        );
    }

    for (ring, range) in rings.iter().zip(ranges.iter()) {
        for i in range.clone() {
            let j = if i + 1 == range.end {
                range.start
            } else {
                i + 1
            };
            let top_i = i;
            let top_j = j;
            let bottom_i = i + vertex_count_2d;
            let bottom_j = j + vertex_count_2d;
            let mid = (points[top_i] + points[top_j]) * 0.5;
            let desired = ring.outward_at(mid);
            push_oriented(&points, &mut triangles, [top_i, bottom_j, top_j], desired);
            push_oriented(
                &points,
                &mut triangles,
                [top_i, bottom_i, bottom_j],
                desired,
            );
        }
    }

    Solid::from_triangle_mesh(points, &triangles).map_err(FeatureError::Topology)
}

#[derive(Clone, Debug)]
struct MetricHoleSpec {
    standard: String,
    clearance_diameter_mm: f64,
}

#[derive(Clone, Debug)]
struct Ring {
    points: Vec<[f64; 2]>,
    kind: RingKind,
}

#[derive(Clone, Debug)]
enum RingKind {
    Outer,
    Hole { center: [f64; 2] },
}

impl Ring {
    fn outer_rectangle(length: f64, width: f64) -> Self {
        let hx = length * 0.5;
        let hy = width * 0.5;
        Self {
            points: vec![[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]],
            kind: RingKind::Outer,
        }
    }

    fn hole(hole: ThroughHole) -> Self {
        let radius = hole.diameter_mm * 0.5;
        let segments = normalize_segments(hole.segments);
        let mut points = Vec::with_capacity(segments);
        for i in 0..segments {
            let theta = -core::f64::consts::TAU * i as f64 / segments as f64;
            points.push([
                hole.x_mm + radius * theta.cos(),
                hole.y_mm + radius * theta.sin(),
            ]);
        }
        Self {
            points,
            kind: RingKind::Hole {
                center: [hole.x_mm, hole.y_mm],
            },
        }
    }

    fn outward_at(&self, point: Point3) -> Vec3 {
        match self.kind {
            RingKind::Outer => Vec3::new(point.x, point.y, 0.0).normalized(),
            RingKind::Hole { center } => {
                Vec3::new(center[0] - point.x, center[1] - point.y, 0.0).normalized()
            }
        }
    }
}

fn distribute_holes(
    count: usize,
    diameter_mm: f64,
    metric: Option<&MetricHoleSpec>,
    plate: BasePlate,
) -> Result<Vec<FeatureNode>, FeatureError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let radius = diameter_mm * 0.5;
    if radius <= 0.0 {
        return Err(FeatureError::InvalidDimension(
            "hole diameter must be positive",
        ));
    }
    let usable = plate.length_mm - 4.0 * radius;
    if usable <= 0.0 {
        return Err(FeatureError::InvalidFeature(
            "plate is too short for requested holes",
        ));
    }

    let positions = if count == 1 {
        vec![0.0]
    } else {
        let pitch = (usable / (count - 1) as f64).min(plate.length_mm * 0.5);
        let start = -pitch * (count - 1) as f64 * 0.5;
        (0..count).map(|i| start + pitch * i as f64).collect()
    };

    Ok(positions
        .into_iter()
        .enumerate()
        .map(|(i, x_mm)| FeatureNode {
            id: format!("hole_{}", i + 1),
            operation: FeatureOperation::ThroughHole(ThroughHole {
                x_mm,
                y_mm: 0.0,
                diameter_mm,
                segments: 32,
                standard: metric.map(|spec| spec.standard.clone()),
            }),
            children: Vec::new(),
        })
        .collect())
}

fn validate_plate(plate: BasePlate) -> Result<(), FeatureError> {
    if plate.length_mm <= 0.0 {
        return Err(FeatureError::InvalidDimension(
            "plate length must be positive",
        ));
    }
    if plate.width_mm <= 0.0 {
        return Err(FeatureError::InvalidDimension(
            "plate width must be positive",
        ));
    }
    if plate.thickness_mm <= 0.0 {
        return Err(FeatureError::InvalidDimension(
            "plate thickness must be positive",
        ));
    }
    Ok(())
}

fn validate_holes(plate: BasePlate, holes: &[ThroughHole]) -> Result<(), FeatureError> {
    for hole in holes {
        let radius = hole.diameter_mm * 0.5;
        if radius <= 0.0 {
            return Err(FeatureError::InvalidDimension(
                "hole diameter must be positive",
            ));
        }
        if hole.segments < 8 {
            return Err(FeatureError::InvalidDimension(
                "hole segments must be at least 8",
            ));
        }
        if hole.x_mm.abs() + radius >= plate.length_mm * 0.5 {
            return Err(FeatureError::InvalidFeature("hole crosses plate length"));
        }
        if hole.y_mm.abs() + radius >= plate.width_mm * 0.5 {
            return Err(FeatureError::InvalidFeature("hole crosses plate width"));
        }
    }

    for i in 0..holes.len() {
        for j in (i + 1)..holes.len() {
            let a = &holes[i];
            let b = &holes[j];
            let distance = ((a.x_mm - b.x_mm).powi(2) + (a.y_mm - b.y_mm).powi(2)).sqrt();
            if distance <= (a.diameter_mm + b.diameter_mm) * 0.5 {
                return Err(FeatureError::InvalidFeature("holes overlap"));
            }
        }
    }
    Ok(())
}

fn push_oriented(
    points: &[Point3],
    triangles: &mut Vec<[usize; 3]>,
    tri: [usize; 3],
    desired_normal: Vec3,
) {
    let normal = triangle_normal(points, tri);
    if normal.dot(desired_normal) >= 0.0 {
        triangles.push(tri);
    } else {
        triangles.push([tri[0], tri[2], tri[1]]);
    }
}

fn triangle_normal(points: &[Point3], tri: [usize; 3]) -> Vec3 {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    (b - a).cross(c - a).normalized()
}

fn normalize_prompt(input: &str) -> String {
    input.to_ascii_lowercase().replace(['-', '_', ','], " ")
}

fn parse_dimensions(input: &str) -> Option<[f64; 3]> {
    for token in input.split_whitespace() {
        let token = token.trim_end_matches("mm");
        if !token.contains('x') {
            continue;
        }
        let parts: Vec<f64> = token
            .split('x')
            .filter_map(|part| part.trim().parse::<f64>().ok())
            .collect();
        if parts.len() == 3 {
            return Some([parts[0], parts[1], parts[2]]);
        }
    }
    None
}

fn parse_first_mm_value(input: &str) -> Option<f64> {
    input
        .split_whitespace()
        .find_map(|token| token.strip_suffix("mm")?.parse::<f64>().ok())
}

fn parse_hole_count(input: &str) -> Option<usize> {
    let words: Vec<&str> = input.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if !word.starts_with("hole") {
            continue;
        }
        for candidate in words[..i].iter().rev() {
            if candidate.starts_with('m') {
                continue;
            }
            if let Some(count) =
                parse_count_word(candidate).or_else(|| candidate.parse::<usize>().ok())
            {
                return Some(count);
            }
        }
        return Some(1);
    }
    None
}

fn parse_count_word(word: &str) -> Option<usize> {
    match word {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        _ => None,
    }
}

fn parse_hole_spec(input: &str) -> Option<MetricHoleSpec> {
    for token in input.split_whitespace() {
        let Some(rest) = token.strip_prefix('m') else {
            continue;
        };
        let nominal = rest.parse::<f64>().ok()?;
        let clearance = metric_clearance_diameter(nominal);
        return Some(MetricHoleSpec {
            standard: format!("M{}", rest),
            clearance_diameter_mm: clearance,
        });
    }
    None
}

fn parse_hole_diameter_mm(input: &str) -> Option<f64> {
    let words: Vec<&str> = input.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if !word.starts_with("hole") || i == 0 {
            continue;
        }
        if let Some(value) = words[i - 1].strip_suffix("mm") {
            return value.parse::<f64>().ok();
        }
    }
    None
}

fn metric_clearance_diameter(nominal_mm: f64) -> f64 {
    match (nominal_mm * 10.0).round() as i32 {
        30 => 3.4,
        40 => 4.5,
        50 => 5.5,
        60 => 6.6,
        80 => 9.0,
        _ => nominal_mm * 1.125,
    }
}

fn default_length_mm(hole_count: usize, diameter_mm: f64) -> f64 {
    let count = hole_count.max(1) as f64;
    (count + 2.0) * diameter_mm.max(4.0) * 4.0
}

fn default_width_mm(diameter_mm: f64) -> f64 {
    diameter_mm.max(4.0) * 5.0
}

fn normalize_segments(segments: usize) -> usize {
    segments.max(8).div_ceil(4) * 4
}
