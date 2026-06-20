//! Boolean operations on supported analytic solids.
//!
//! The module implements a production-style pipeline shape for one deliberately
//! scoped case: subtracting a vertical cylinder from a cube. It also includes a
//! staged classifier for split faces: SSI output installed as topology
//! [`crate::topology::SplitEdge`] records can be classified into Boolean
//! keep/discard side decisions before a later trim-healing pass rewrites the
//! shell topology.

use crate::math::{Point3, Vec2, Vec3};
use crate::topology::{
    FaceId, SewingReport, Solid, SplitEdgeId, TopologyError, TrimCurve2D, TrimLoopKind,
};

/// Boolean operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union.
    Union,
    /// Subtract right from left.
    Subtract,
    /// Intersection.
    Intersect,
}

/// Operand ownership for a face participating in staged Boolean classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOperand {
    /// Left-hand Boolean operand.
    Left,
    /// Right-hand Boolean operand.
    Right,
}

/// Classification of a split p-curve against its face trim domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimDomainStatus {
    /// The sampled split curve lies in the visible face region.
    Inside,
    /// The sampled split curve lies on existing trim boundaries.
    Boundary,
    /// The sampled split curve lies outside the visible face region.
    Outside,
    /// The sampled split curve crosses from visible to invisible face regions.
    Crossing,
    /// The trim domain could not be classified, usually because a curve is unresolved.
    Unknown,
}

/// Local side of a split curve relative to the opposite operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanRegionSide {
    /// The local side is inside the opposite operand.
    InsideOther,
    /// The local side is outside the opposite operand.
    OutsideOther,
    /// The side could not be certified, usually for tangency or missing derivatives.
    Ambiguous,
}

/// Boolean action assigned to one local side of a split curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanRegionAction {
    /// Keep the region with its existing orientation.
    Keep,
    /// Keep the region, reversing orientation because it came from the subtracted operand.
    KeepReversed,
    /// Discard the region.
    Discard,
    /// Ignore this split because it does not separate the two Boolean operands.
    Ignored,
    /// A robust keep/discard decision is not available.
    Ambiguous,
}

/// Overall status for one staged split edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanSplitStatus {
    /// The split separates left and right operands and received concrete side actions.
    Active,
    /// Both face uses belong to the same operand, so this is not a Boolean separator.
    InactiveSameOperand,
    /// Classification could not produce concrete side actions.
    Ambiguous,
}

/// Classification for one face-side use of a staged split edge.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanFaceSplitClassification {
    /// Face carrying this split use.
    pub face: FaceId,
    /// Index into `solid.faces[face].split_curves`.
    pub split_index: usize,
    /// Boolean operand that owns the face.
    pub operand: BooleanOperand,
    /// Relationship between the split p-curve and this face's trim domain.
    pub trim_domain: TrimDomainStatus,
    /// Classification of the UV-left side of the oriented split p-curve.
    pub left_of_curve: BooleanRegionSide,
    /// Classification of the UV-right side of the oriented split p-curve.
    pub right_of_curve: BooleanRegionSide,
    /// Boolean action for the UV-left side of the oriented split p-curve.
    pub left_action: BooleanRegionAction,
    /// Boolean action for the UV-right side of the oriented split p-curve.
    pub right_action: BooleanRegionAction,
}

/// Classification for one staged split edge and its two face-side uses.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanSplitClassification {
    /// Shared staged split edge.
    pub split_edge: SplitEdgeId,
    /// Overall classification status.
    pub status: BooleanSplitStatus,
    /// First face-side use.
    pub a: BooleanFaceSplitClassification,
    /// Second face-side use.
    pub b: BooleanFaceSplitClassification,
}

/// Report returned by staged Boolean split classification.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanClassificationReport {
    /// Boolean operation being classified.
    pub operation: BooleanOp,
    /// Number of staged split edges considered.
    pub split_count: usize,
    /// Number of split edges that became active Boolean separators.
    pub active_split_count: usize,
    /// Per-split classification records.
    pub classifications: Vec<BooleanSplitClassification>,
}

/// Local side selected for a healed face region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealedRegionSide {
    /// Region on the UV-left side of the oriented split p-curve.
    LeftOfSplit,
    /// Region on the UV-right side of the oriented split p-curve.
    RightOfSplit,
}

/// One healed trim region produced from a classified split face.
#[derive(Clone, Debug, PartialEq)]
pub struct HealedFaceRegion {
    /// Source face in the staged input solid.
    pub source_face: FaceId,
    /// Operand that owns the source face.
    pub operand: BooleanOperand,
    /// Boolean action that kept this region.
    pub action: BooleanRegionAction,
    /// Side of the split p-curve represented by this region.
    pub side: HealedRegionSide,
    /// Closed outer loop in the source face parameter domain.
    pub uv_loop: Vec<Vec2>,
    /// Model-space vertices sampled from `uv_loop`.
    pub vertices: Vec<Point3>,
    /// Source face normal at the first loop point.
    pub normal: Vec3,
    /// True when output triangles should reverse the source face orientation.
    pub orientation_reversed: bool,
}

/// Triangle mesh emitted by the healing pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HealedTriangleMesh {
    /// Welded model-space vertices.
    pub vertices: Vec<Point3>,
    /// Indexed triangles.
    pub triangles: Vec<[usize; 3]>,
}

/// Output from promoting classified split faces into healed regions.
#[derive(Clone, Debug, PartialEq)]
pub struct HealedBooleanOutput {
    /// Boolean operation used by the classification report.
    pub operation: BooleanOp,
    /// Healed kept regions.
    pub regions: Vec<HealedFaceRegion>,
    /// Triangulated and welded mesh for the healed regions.
    pub mesh: HealedTriangleMesh,
    /// Validated closed half-edge solid when the healed mesh forms a closed shell.
    pub solid: Option<Solid>,
    /// Validation error when the healed mesh is only a partial/open shell.
    pub solid_error: Option<TopologyError>,
    /// Tolerance-aware sewing report for the healed mesh, when meshing ran.
    pub sewing_report: Option<SewingReport>,
}

/// Boolean error.
#[derive(Clone, Debug, PartialEq)]
pub enum BooleanError {
    /// Unsupported operation or operand pair.
    Unsupported,
    /// Invalid input parameters.
    InvalidInput(&'static str),
    /// Topology validation failed.
    Topology(TopologyError),
}

impl From<TopologyError> for BooleanError {
    fn from(value: TopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Diagnostics returned with a boolean result.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanReport {
    /// Resulting solid.
    pub solid: Solid,
    /// Number of generated triangles.
    pub triangle_count: usize,
    /// Euler characteristic of the result.
    pub euler_characteristic: isize,
    /// Genus estimate.
    pub genus: Option<isize>,
}

/// Classify staged face splits into Boolean keep/discard side decisions.
///
/// `face_operands` must contain one [`BooleanOperand`] per face in `solid`.
/// Each staged split edge must have exactly two face uses, normally one from
/// the left operand and one from the right operand. The classifier samples each
/// split p-curve against the owning face's trim loops, then uses local surface
/// differentials and the opposite face normal to decide which side of the
/// oriented p-curve is inside the opposite operand.
pub fn classify_split_faces(
    solid: &Solid,
    face_operands: &[BooleanOperand],
    operation: BooleanOp,
    tolerance: f64,
) -> Result<BooleanClassificationReport, BooleanError> {
    if face_operands.len() != solid.faces.len() {
        return Err(BooleanError::InvalidInput(
            "face_operands must contain one entry per face",
        ));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }

    solid.validate()?;
    let tolerance = tolerance.max(1.0e-9);
    let uses = split_uses(solid);
    let mut classifications = Vec::with_capacity(solid.split_edges.len());
    for (split_edge, split_uses) in uses.iter().enumerate() {
        if split_uses.len() != 2 {
            return Err(BooleanError::Topology(TopologyError::InvalidSplitEdge(
                split_edge,
            )));
        }
        let a_use = split_uses[0];
        let b_use = split_uses[1];
        let a = classify_face_split(
            solid,
            a_use,
            b_use,
            face_operands[a_use.face],
            face_operands[b_use.face],
            operation,
            tolerance,
        );
        let b = classify_face_split(
            solid,
            b_use,
            a_use,
            face_operands[b_use.face],
            face_operands[a_use.face],
            operation,
            tolerance,
        );
        let status = split_status(&a, &b);
        classifications.push(BooleanSplitClassification {
            split_edge,
            status,
            a,
            b,
        });
    }
    let active_split_count = classifications
        .iter()
        .filter(|classification| classification.status == BooleanSplitStatus::Active)
        .count();
    Ok(BooleanClassificationReport {
        operation,
        split_count: solid.split_edges.len(),
        active_split_count,
        classifications,
    })
}

/// Promote classified split faces into healed trim regions and a triangulated shell candidate.
///
/// This is the first output-generation step after split classification. It
/// supports boundary-to-boundary split p-curves on faces whose outer trim loop
/// can be flattened to a polyline. Kept sides are sewn into new trim loops,
/// sampled back onto the analytic face, triangulated, and welded. If the
/// triangulated result is watertight, `solid` contains a validated half-edge
/// shell; otherwise `mesh` and `regions` still expose the healed partial output.
pub fn heal_classified_split_faces(
    solid: &Solid,
    classification: &BooleanClassificationReport,
    tolerance: f64,
) -> Result<HealedBooleanOutput, BooleanError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }
    solid.validate()?;
    let tolerance = tolerance.max(1.0e-9);
    let mut regions = Vec::new();
    for split in &classification.classifications {
        if split.status != BooleanSplitStatus::Active {
            continue;
        }
        push_healed_regions_for_face(solid, &split.a, tolerance, &mut regions)?;
        push_healed_regions_for_face(solid, &split.b, tolerance, &mut regions)?;
    }
    let mesh = triangulate_healed_regions(&regions, tolerance)?;
    let (solid, solid_error, sewing_report) = if mesh.triangles.is_empty() {
        (None, None, None)
    } else {
        match Solid::sew_triangle_mesh(mesh.vertices.clone(), &mesh.triangles, tolerance) {
            Ok(sewn) => {
                if sewn.triangles.is_empty() {
                    (
                        None,
                        Some(TopologyError::DegenerateSewnMesh),
                        Some(sewn.report),
                    )
                } else {
                    let sewn_vertices = sewn.vertices;
                    let sewn_triangles = sewn.triangles;
                    let report = sewn.report;
                    match Solid::from_triangle_mesh(sewn_vertices, &sewn_triangles) {
                        Ok(mut solid) => {
                            for edge in &mut solid.edges {
                                edge.tolerance = tolerance;
                            }
                            (Some(solid), None, Some(report))
                        }
                        Err(error) => (None, Some(error), Some(report)),
                    }
                }
            }
            Err(error) => (None, Some(error), None),
        }
    };
    Ok(HealedBooleanOutput {
        operation: classification.operation,
        regions,
        mesh,
        solid,
        solid_error,
        sewing_report,
    })
}

/// Subtract a Z-aligned cylinder from a cube, yielding a genus-1 closed solid.
pub fn subtract_cube_cylinder(
    cube_size: f64,
    cylinder_radius: f64,
    requested_segments: usize,
) -> Result<BooleanReport, BooleanError> {
    if cube_size <= 0.0 {
        return Err(BooleanError::InvalidInput("cube_size must be positive"));
    }
    if cylinder_radius <= 0.0 || cylinder_radius >= cube_size * 0.5 {
        return Err(BooleanError::InvalidInput(
            "cylinder_radius must be inside the cube",
        ));
    }
    let segments = requested_segments.max(8).div_ceil(8) * 8;
    let h = cube_size * 0.5;
    let r = cylinder_radius;

    let mut points = Vec::<Point3>::with_capacity(segments * 4);
    let mut inner_top = Vec::with_capacity(segments);
    let mut inner_bottom = Vec::with_capacity(segments);
    let mut outer_top = Vec::with_capacity(segments);
    let mut outer_bottom = Vec::with_capacity(segments);

    for i in 0..segments {
        let theta = core::f64::consts::TAU * i as f64 / segments as f64;
        let c = theta.cos();
        let s = theta.sin();
        let scale = h / c.abs().max(s.abs());
        let outer = (scale * c, scale * s);
        let inner = (r * c, r * s);

        inner_top.push(push_point(&mut points, Point3::new(inner.0, inner.1, h)));
        inner_bottom.push(push_point(&mut points, Point3::new(inner.0, inner.1, -h)));
        outer_top.push(push_point(&mut points, Point3::new(outer.0, outer.1, h)));
        outer_bottom.push(push_point(&mut points, Point3::new(outer.0, outer.1, -h)));
    }

    let mut triangles = Vec::<[usize; 3]>::with_capacity(segments * 8);
    for i in 0..segments {
        let j = (i + 1) % segments;

        // Top square annulus, normal +Z.
        triangles.push([inner_top[i], outer_top[i], outer_top[j]]);
        triangles.push([inner_top[i], outer_top[j], inner_top[j]]);

        // Bottom square annulus, normal -Z.
        triangles.push([inner_bottom[i], outer_bottom[j], outer_bottom[i]]);
        triangles.push([inner_bottom[i], inner_bottom[j], outer_bottom[j]]);

        // Inner cylindrical wall. This is a subtraction boundary, so the
        // outward normal points into the removed cylinder.
        triangles.push([inner_bottom[i], inner_top[i], inner_top[j]]);
        triangles.push([inner_bottom[i], inner_top[j], inner_bottom[j]]);

        // Outer cube side wall.
        triangles.push([outer_bottom[i], outer_bottom[j], outer_top[j]]);
        triangles.push([outer_bottom[i], outer_top[j], outer_top[i]]);
    }

    let solid = Solid::from_triangle_mesh(points, &triangles)?;
    Ok(BooleanReport {
        triangle_count: triangles.len(),
        euler_characteristic: solid.euler_characteristic(),
        genus: solid.genus(),
        solid,
    })
}

fn push_point(points: &mut Vec<Point3>, point: Point3) -> usize {
    points.push(point);
    points.len() - 1
}

fn push_healed_regions_for_face(
    solid: &Solid,
    classification: &BooleanFaceSplitClassification,
    tolerance: f64,
    regions: &mut Vec<HealedFaceRegion>,
) -> Result<(), BooleanError> {
    if action_keeps_region(classification.left_action) {
        regions.push(healed_region_for_side(
            solid,
            classification,
            HealedRegionSide::LeftOfSplit,
            classification.left_action,
            tolerance,
        )?);
    }
    if action_keeps_region(classification.right_action) {
        regions.push(healed_region_for_side(
            solid,
            classification,
            HealedRegionSide::RightOfSplit,
            classification.right_action,
            tolerance,
        )?);
    }
    Ok(())
}

fn healed_region_for_side(
    solid: &Solid,
    classification: &BooleanFaceSplitClassification,
    side: HealedRegionSide,
    action: BooleanRegionAction,
    tolerance: f64,
) -> Result<HealedFaceRegion, BooleanError> {
    let face = solid
        .faces
        .get(classification.face)
        .ok_or(BooleanError::Unsupported)?;
    let split = face
        .split_curves
        .get(classification.split_index)
        .ok_or(BooleanError::Unsupported)?;
    if face
        .trim_loops
        .iter()
        .any(|trim_loop| trim_loop.kind == TrimLoopKind::Inner)
    {
        return Err(BooleanError::Unsupported);
    }
    let outer_loop = face
        .trim_loops
        .iter()
        .find(|trim_loop| trim_loop.kind == TrimLoopKind::Outer)
        .ok_or(BooleanError::Unsupported)?;
    let mut outer = trim_loop_polyline(outer_loop).ok_or(BooleanError::Unsupported)?;
    cleanup_loop(&mut outer, tolerance);
    if outer.len() < 3 {
        return Err(BooleanError::Unsupported);
    }
    if polygon_signed_area(&outer) < 0.0 {
        outer.reverse();
    }

    let mut split_points = trim_curve_polyline(&split.pcurve).ok_or(BooleanError::Unsupported)?;
    cleanup_open_polyline(&mut split_points, tolerance);
    if split_points.len() < 2 {
        return Err(BooleanError::Unsupported);
    }
    let start = find_boundary_location(&outer, split_points[0], tolerance)
        .ok_or(BooleanError::Unsupported)?;
    let end = find_boundary_location(&outer, split_points[split_points.len() - 1], tolerance)
        .ok_or(BooleanError::Unsupported)?;
    if vec2_distance(start.point, end.point) <= tolerance {
        return Err(BooleanError::Unsupported);
    }
    split_points[0] = start.point;
    let last = split_points.len() - 1;
    split_points[last] = end.point;

    let mut uv_loop = match side {
        HealedRegionSide::LeftOfSplit => {
            let mut loop_points = split_points.clone();
            append_path_without_duplicate(
                &mut loop_points,
                &boundary_path(&outer, end, start, tolerance),
                tolerance,
            );
            loop_points
        }
        HealedRegionSide::RightOfSplit => {
            let mut loop_points = split_points;
            loop_points.reverse();
            append_path_without_duplicate(
                &mut loop_points,
                &boundary_path(&outer, start, end, tolerance),
                tolerance,
            );
            loop_points
        }
    };
    cleanup_loop(&mut uv_loop, tolerance);
    if uv_loop.len() < 3 || polygon_signed_area(&uv_loop).abs() <= tolerance * tolerance {
        return Err(BooleanError::Unsupported);
    }
    if polygon_signed_area(&uv_loop) < 0.0 {
        uv_loop.reverse();
    }

    let mut vertices = Vec::with_capacity(uv_loop.len());
    for uv in &uv_loop {
        vertices.push(
            face.surface
                .evaluate(*uv)
                .ok_or(BooleanError::Unsupported)?,
        );
    }
    let normal = face
        .surface
        .normal_at(uv_loop[0])
        .ok_or(BooleanError::Unsupported)?;
    Ok(HealedFaceRegion {
        source_face: classification.face,
        operand: classification.operand,
        action,
        side,
        uv_loop,
        vertices,
        normal,
        orientation_reversed: action == BooleanRegionAction::KeepReversed,
    })
}

fn triangulate_healed_regions(
    regions: &[HealedFaceRegion],
    tolerance: f64,
) -> Result<HealedTriangleMesh, BooleanError> {
    let mut mesh = HealedTriangleMesh::default();
    for region in regions {
        if region.uv_loop.len() < 3 || region.uv_loop.len() != region.vertices.len() {
            return Err(BooleanError::Unsupported);
        }
        let mut flat = Vec::with_capacity(region.uv_loop.len() * 2);
        for uv in &region.uv_loop {
            flat.push(uv.x);
            flat.push(uv.y);
        }
        let local_triangles =
            earcutr::earcut(&flat, &[], 2).map_err(|_| BooleanError::Unsupported)?;
        if local_triangles.len() % 3 != 0 {
            return Err(BooleanError::Unsupported);
        }
        let indices: Vec<usize> = region
            .vertices
            .iter()
            .map(|vertex| weld_mesh_vertex(&mut mesh.vertices, *vertex, tolerance))
            .collect();
        let desired = if region.orientation_reversed {
            -region.normal
        } else {
            region.normal
        };
        for chunk in local_triangles.chunks_exact(3) {
            push_oriented_triangle(
                &mesh.vertices,
                &mut mesh.triangles,
                [indices[chunk[0]], indices[chunk[1]], indices[chunk[2]]],
                desired,
            );
        }
    }
    Ok(mesh)
}

#[derive(Clone, Copy)]
struct SplitUse {
    face: FaceId,
    split_index: usize,
}

#[derive(Clone, Copy)]
struct BoundaryLocation {
    segment: usize,
    t: f64,
    point: Vec2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopPointLocation {
    Inside,
    Boundary,
    Outside,
}

fn split_uses(solid: &Solid) -> Vec<Vec<SplitUse>> {
    let mut uses = vec![Vec::new(); solid.split_edges.len()];
    for (face, face_record) in solid.faces.iter().enumerate() {
        for (split_index, split) in face_record.split_curves.iter().enumerate() {
            if let Some(split_uses) = uses.get_mut(split.split_edge) {
                split_uses.push(SplitUse { face, split_index });
            }
        }
    }
    uses
}

fn classify_face_split(
    solid: &Solid,
    this_use: SplitUse,
    other_use: SplitUse,
    this_operand: BooleanOperand,
    other_operand: BooleanOperand,
    operation: BooleanOp,
    tolerance: f64,
) -> BooleanFaceSplitClassification {
    let split = &solid.faces[this_use.face].split_curves[this_use.split_index];
    let other_split = &solid.faces[other_use.face].split_curves[other_use.split_index];
    let trim_domain = classify_pcurve_domain(
        solid,
        this_use.face,
        &split.pcurve,
        tolerance.max(split.tolerance),
    );

    let (left_of_curve, right_of_curve) = if this_operand == other_operand {
        (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous)
    } else {
        classify_local_sides(
            &solid.faces[this_use.face].surface,
            &split.pcurve,
            &solid.faces[other_use.face].surface,
            &other_split.pcurve,
            tolerance.max(split.tolerance).max(other_split.tolerance),
        )
    };
    let left_action = action_for_side(operation, this_operand, other_operand, left_of_curve);
    let right_action = action_for_side(operation, this_operand, other_operand, right_of_curve);
    BooleanFaceSplitClassification {
        face: this_use.face,
        split_index: this_use.split_index,
        operand: this_operand,
        trim_domain,
        left_of_curve,
        right_of_curve,
        left_action,
        right_action,
    }
}

fn split_status(
    a: &BooleanFaceSplitClassification,
    b: &BooleanFaceSplitClassification,
) -> BooleanSplitStatus {
    if a.operand == b.operand {
        return BooleanSplitStatus::InactiveSameOperand;
    }
    if !trim_domain_is_usable(a.trim_domain)
        || !trim_domain_is_usable(b.trim_domain)
        || !action_is_concrete(a.left_action)
        || !action_is_concrete(a.right_action)
        || !action_is_concrete(b.left_action)
        || !action_is_concrete(b.right_action)
    {
        return BooleanSplitStatus::Ambiguous;
    }
    BooleanSplitStatus::Active
}

fn action_for_side(
    operation: BooleanOp,
    this_operand: BooleanOperand,
    other_operand: BooleanOperand,
    side: BooleanRegionSide,
) -> BooleanRegionAction {
    if this_operand == other_operand {
        return BooleanRegionAction::Ignored;
    }
    match side {
        BooleanRegionSide::Ambiguous => BooleanRegionAction::Ambiguous,
        BooleanRegionSide::InsideOther => match operation {
            BooleanOp::Union => BooleanRegionAction::Discard,
            BooleanOp::Intersect => BooleanRegionAction::Keep,
            BooleanOp::Subtract => match this_operand {
                BooleanOperand::Left => BooleanRegionAction::Discard,
                BooleanOperand::Right => BooleanRegionAction::KeepReversed,
            },
        },
        BooleanRegionSide::OutsideOther => match operation {
            BooleanOp::Union => BooleanRegionAction::Keep,
            BooleanOp::Intersect => BooleanRegionAction::Discard,
            BooleanOp::Subtract => match this_operand {
                BooleanOperand::Left => BooleanRegionAction::Keep,
                BooleanOperand::Right => BooleanRegionAction::Discard,
            },
        },
    }
}

fn action_is_concrete(action: BooleanRegionAction) -> bool {
    matches!(
        action,
        BooleanRegionAction::Keep
            | BooleanRegionAction::KeepReversed
            | BooleanRegionAction::Discard
    )
}

fn action_keeps_region(action: BooleanRegionAction) -> bool {
    matches!(
        action,
        BooleanRegionAction::Keep | BooleanRegionAction::KeepReversed
    )
}

fn trim_domain_is_usable(status: TrimDomainStatus) -> bool {
    matches!(
        status,
        TrimDomainStatus::Inside | TrimDomainStatus::Boundary
    )
}

fn trim_curve_polyline(curve: &TrimCurve2D) -> Option<Vec<Vec2>> {
    let mut points = Vec::new();
    append_trim_curve_points(&mut points, curve)?;
    Some(points)
}

fn classify_local_sides(
    surface: &crate::topology::FaceSurface,
    pcurve: &TrimCurve2D,
    other_surface: &crate::topology::FaceSurface,
    other_pcurve: &TrimCurve2D,
    tolerance: f64,
) -> (BooleanRegionSide, BooleanRegionSide) {
    let Some((uv, tangent)) = representative_pcurve_sample(pcurve, tolerance) else {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    };
    let Some((du, dv)) = surface.partials(uv) else {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    };
    let Some((other_uv, _)) = representative_pcurve_sample(other_pcurve, tolerance) else {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    };
    let Some(other_normal) = other_surface.normal_at(other_uv) else {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    };
    let uv_left = Vec2::new(-tangent.y, tangent.x);
    let model_left = du * uv_left.x + dv * uv_left.y;
    if model_left.norm() <= tolerance || other_normal.norm() <= f64::EPSILON {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    }
    let signed_change = other_normal.normalized().dot(model_left.normalized());
    if signed_change.abs() <= tolerance.max(1.0e-12) {
        return (BooleanRegionSide::Ambiguous, BooleanRegionSide::Ambiguous);
    }
    if signed_change < 0.0 {
        (
            BooleanRegionSide::InsideOther,
            BooleanRegionSide::OutsideOther,
        )
    } else {
        (
            BooleanRegionSide::OutsideOther,
            BooleanRegionSide::InsideOther,
        )
    }
}

fn find_boundary_location(
    boundary: &[Vec2],
    point: Vec2,
    tolerance: f64,
) -> Option<BoundaryLocation> {
    let mut best = None;
    let mut best_distance = tolerance;
    for segment in 0..boundary.len() {
        let a = boundary[segment];
        let b = boundary[(segment + 1) % boundary.len()];
        let edge = b - a;
        let len2 = edge.dot(edge);
        if len2 <= f64::EPSILON {
            continue;
        }
        let t = ((point - a).dot(edge) / len2).clamp(0.0, 1.0);
        let projected = a + edge * t;
        let distance = vec2_distance(point, projected);
        if distance <= best_distance {
            best_distance = distance;
            best = Some(BoundaryLocation {
                segment,
                t,
                point: projected,
            });
        }
    }
    best
}

fn boundary_path(
    boundary: &[Vec2],
    from: BoundaryLocation,
    to: BoundaryLocation,
    tolerance: f64,
) -> Vec<Vec2> {
    let mut path = vec![from.point];
    if from.segment == to.segment && to.t + tolerance >= from.t {
        path.push(to.point);
        return path;
    }

    let mut vertex = (from.segment + 1) % boundary.len();
    loop {
        path.push(boundary[vertex]);
        if vertex == to.segment {
            break;
        }
        vertex = (vertex + 1) % boundary.len();
    }
    path.push(to.point);
    path
}

fn classify_pcurve_domain(
    solid: &Solid,
    face: FaceId,
    pcurve: &TrimCurve2D,
    tolerance: f64,
) -> TrimDomainStatus {
    let Some(samples) = sample_pcurve_points(pcurve) else {
        return TrimDomainStatus::Unknown;
    };
    let mut saw_inside = false;
    let mut saw_boundary = false;
    let mut saw_outside = false;
    for sample in samples {
        match classify_face_uv(solid, face, sample, tolerance) {
            Some(LoopPointLocation::Inside) => saw_inside = true,
            Some(LoopPointLocation::Boundary) => saw_boundary = true,
            Some(LoopPointLocation::Outside) => saw_outside = true,
            None => return TrimDomainStatus::Unknown,
        }
    }

    match (saw_inside, saw_boundary, saw_outside) {
        (true, _, false) => TrimDomainStatus::Inside,
        (false, true, false) => TrimDomainStatus::Boundary,
        (false, false, true) => TrimDomainStatus::Outside,
        (false, false, false) => TrimDomainStatus::Unknown,
        _ => TrimDomainStatus::Crossing,
    }
}

fn classify_face_uv(
    solid: &Solid,
    face: FaceId,
    point: Vec2,
    tolerance: f64,
) -> Option<LoopPointLocation> {
    let face = solid.faces.get(face)?;
    let mut outer = None;
    let mut inside_hole = false;
    for trim_loop in &face.trim_loops {
        let polyline = trim_loop_polyline(trim_loop)?;
        let location = classify_loop_point(&polyline, point, tolerance);
        if location == LoopPointLocation::Boundary {
            return Some(LoopPointLocation::Boundary);
        }
        match trim_loop.kind {
            TrimLoopKind::Outer => outer = Some(location),
            TrimLoopKind::Inner => {
                if location == LoopPointLocation::Inside {
                    inside_hole = true;
                }
            }
        }
    }

    match outer? {
        LoopPointLocation::Inside if !inside_hole => Some(LoopPointLocation::Inside),
        LoopPointLocation::Boundary => Some(LoopPointLocation::Boundary),
        LoopPointLocation::Inside | LoopPointLocation::Outside => Some(LoopPointLocation::Outside),
    }
}

fn classify_loop_point(polyline: &[Vec2], point: Vec2, tolerance: f64) -> LoopPointLocation {
    if polyline.len() < 3 {
        return LoopPointLocation::Outside;
    }
    let mut inside = false;
    for i in 0..polyline.len() {
        let a = polyline[i];
        let b = polyline[(i + 1) % polyline.len()];
        if point_segment_distance(point, a, b) <= tolerance {
            return LoopPointLocation::Boundary;
        }
        let crosses_y = (a.y > point.y) != (b.y > point.y);
        if crosses_y {
            let x_at_y = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if point.x < x_at_y {
                inside = !inside;
            }
        }
    }
    if inside {
        LoopPointLocation::Inside
    } else {
        LoopPointLocation::Outside
    }
}

fn trim_loop_polyline(trim_loop: &crate::topology::TrimLoop) -> Option<Vec<Vec2>> {
    let mut points = Vec::new();
    for trim in &trim_loop.trims {
        append_trim_curve_points(&mut points, &trim.curve)?;
    }
    if points.len() >= 2 && vec2_distance(points[0], points[points.len() - 1]) <= 1.0e-12 {
        points.pop();
    }
    Some(points)
}

fn append_path_without_duplicate(target: &mut Vec<Vec2>, path: &[Vec2], tolerance: f64) {
    for point in path {
        if target
            .last()
            .is_none_or(|last| vec2_distance(*last, *point) > tolerance)
        {
            target.push(*point);
        }
    }
}

fn append_trim_curve_points(points: &mut Vec<Vec2>, curve: &TrimCurve2D) -> Option<()> {
    match curve {
        TrimCurve2D::LineSegment { start, end } => {
            push_unique_uv(points, *start);
            push_unique_uv(points, *end);
        }
        TrimCurve2D::CircularArc {
            center,
            radius,
            start_angle,
            end_angle,
        } => {
            let delta = *end_angle - *start_angle;
            let steps =
                ((delta.abs() / (core::f64::consts::PI / 8.0)).ceil() as usize).clamp(2, 64);
            for i in 0..=steps {
                let t = i as f64 / steps as f64;
                let angle = *start_angle + delta * t;
                push_unique_uv(
                    points,
                    *center + Vec2::new(angle.cos() * *radius, angle.sin() * *radius),
                );
            }
        }
        TrimCurve2D::Polyline {
            points: curve_points,
        } => {
            for point in curve_points {
                push_unique_uv(points, *point);
            }
        }
        TrimCurve2D::Nurbs(curve) => {
            let (u0, u1) = curve.domain();
            for i in 0..=32 {
                let u = u0 * (1.0 - i as f64 / 32.0) + u1 * (i as f64 / 32.0);
                let point = curve.evaluate(u);
                push_unique_uv(points, Vec2::new(point.x, point.y));
            }
        }
        TrimCurve2D::Unresolved => return None,
    }
    Some(())
}

fn cleanup_open_polyline(points: &mut Vec<Vec2>, tolerance: f64) {
    let mut cleaned = Vec::with_capacity(points.len());
    for point in points.iter().copied() {
        if cleaned
            .last()
            .is_none_or(|last| vec2_distance(*last, point) > tolerance)
        {
            cleaned.push(point);
        }
    }
    *points = cleaned;
}

fn cleanup_loop(points: &mut Vec<Vec2>, tolerance: f64) {
    cleanup_open_polyline(points, tolerance);
    if points.len() >= 2 && vec2_distance(points[0], points[points.len() - 1]) <= tolerance {
        points.pop();
    }
}

fn polygon_signed_area(points: &[Vec2]) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a.cross(b);
    }
    area * 0.5
}

fn sample_pcurve_points(curve: &TrimCurve2D) -> Option<Vec<Vec2>> {
    match curve {
        TrimCurve2D::LineSegment { start, end } => Some(vec![*start, (*start + *end) * 0.5, *end]),
        TrimCurve2D::CircularArc {
            center,
            radius,
            start_angle,
            end_angle,
        } => {
            let delta = *end_angle - *start_angle;
            let steps =
                ((delta.abs() / (core::f64::consts::PI / 8.0)).ceil() as usize).clamp(2, 64);
            let mut samples = Vec::with_capacity(steps + 1);
            for i in 0..=steps {
                let angle = *start_angle + delta * i as f64 / steps as f64;
                samples.push(*center + Vec2::new(angle.cos() * *radius, angle.sin() * *radius));
            }
            Some(samples)
        }
        TrimCurve2D::Polyline { points } => {
            if points.len() < 2 {
                return None;
            }
            let mut samples = Vec::with_capacity(points.len() * 2 - 1);
            samples.push(points[0]);
            for edge in points.windows(2) {
                samples.push((edge[0] + edge[1]) * 0.5);
                samples.push(edge[1]);
            }
            Some(samples)
        }
        TrimCurve2D::Nurbs(curve) => {
            let (u0, u1) = curve.domain();
            Some(
                (0..=32)
                    .map(|i| {
                        let t = i as f64 / 32.0;
                        let point = curve.evaluate(u0 * (1.0 - t) + u1 * t);
                        Vec2::new(point.x, point.y)
                    })
                    .collect(),
            )
        }
        TrimCurve2D::Unresolved => None,
    }
}

fn weld_mesh_vertex(vertices: &mut Vec<Point3>, point: Point3, tolerance: f64) -> usize {
    if let Some(index) = vertices
        .iter()
        .position(|existing| existing.distance(point) <= tolerance)
    {
        index
    } else {
        vertices.push(point);
        vertices.len() - 1
    }
}

fn push_oriented_triangle(
    points: &[Point3],
    triangles: &mut Vec<[usize; 3]>,
    tri: [usize; 3],
    desired_normal: Vec3,
) {
    let normal = (points[tri[1]] - points[tri[0]]).cross(points[tri[2]] - points[tri[0]]);
    if normal.dot(desired_normal) >= 0.0 {
        triangles.push(tri);
    } else {
        triangles.push([tri[0], tri[2], tri[1]]);
    }
}

fn representative_pcurve_sample(curve: &TrimCurve2D, tolerance: f64) -> Option<(Vec2, Vec2)> {
    match curve {
        TrimCurve2D::LineSegment { start, end } => {
            let tangent = *end - *start;
            if vec2_norm(tangent) <= tolerance {
                None
            } else {
                Some(((*start + *end) * 0.5, tangent))
            }
        }
        TrimCurve2D::CircularArc {
            center,
            radius,
            start_angle,
            end_angle,
        } => {
            let delta = *end_angle - *start_angle;
            if radius.abs() <= tolerance || delta.abs() <= tolerance {
                return None;
            }
            let angle = (*start_angle + *end_angle) * 0.5;
            Some((
                *center + Vec2::new(angle.cos() * *radius, angle.sin() * *radius),
                Vec2::new(
                    -angle.sin() * *radius * delta,
                    angle.cos() * *radius * delta,
                ),
            ))
        }
        TrimCurve2D::Polyline { points } => {
            let mut best = None;
            let mut best_len = tolerance;
            for edge in points.windows(2) {
                let tangent = edge[1] - edge[0];
                let length = vec2_norm(tangent);
                if length > best_len {
                    best_len = length;
                    best = Some(((edge[0] + edge[1]) * 0.5, tangent));
                }
            }
            best
        }
        TrimCurve2D::Nurbs(curve) => {
            let (u0, u1) = curve.domain();
            let u = (u0 + u1) * 0.5;
            let point = curve.evaluate(u);
            let tangent = curve.derivative(u);
            let tangent = Vec2::new(tangent.x, tangent.y);
            if vec2_norm(tangent) <= tolerance {
                None
            } else {
                Some((Vec2::new(point.x, point.y), tangent))
            }
        }
        TrimCurve2D::Unresolved => None,
    }
}

fn point_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f64 {
    let edge = b - a;
    let len2 = edge.dot(edge);
    if len2 <= f64::EPSILON {
        return vec2_distance(point, a);
    }
    let t = ((point - a).dot(edge) / len2).clamp(0.0, 1.0);
    vec2_distance(point, a + edge * t)
}

fn push_unique_uv(points: &mut Vec<Vec2>, point: Vec2) {
    if points
        .last()
        .is_none_or(|last| vec2_distance(*last, point) > 1.0e-12)
    {
        points.push(point);
    }
}

fn vec2_norm(value: Vec2) -> f64 {
    value.dot(value).sqrt()
}

fn vec2_distance(a: Vec2, b: Vec2) -> f64 {
    vec2_norm(a - b)
}
