//! Boolean operations on supported analytic solids.
//!
//! The module implements a production-style pipeline shape for one deliberately
//! scoped case: subtracting a vertical cylinder from a cube. It also includes a
//! staged classifier for split faces: SSI output installed as topology
//! [`crate::topology::SplitEdge`] records can be classified into Boolean
//! keep/discard side decisions before a later trim-healing pass rewrites the
//! shell topology.

use crate::geometry::Plane;
use crate::intersection::{
    intersect_nurbs_surfaces, intersect_plane_nurbs_surface, IntersectionPolyline,
    TrimReadyIntersectionCurve,
};
use crate::math::{Point3, Vec2, Vec3};
use crate::topology::{
    EdgeCurve3D, EdgeId, FaceId, FaceSurface, SewingReport, Solid, SplitEdgeId, TopologyError,
    TrimCurve2D, TrimLoopKind, VertexId,
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

/// One connected face-domain region after applying all active split curves on a face.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanClassifiedRegion {
    /// Source face for this region.
    pub face: FaceId,
    /// Boolean operand that owns the source face.
    pub operand: BooleanOperand,
    /// Face-local split indices considered while partitioning this face.
    pub source_splits: Vec<usize>,
    /// True when the source face had no active split curves.
    pub unsplit_face: bool,
    /// Inside/outside classification of this region relative to the opposite operand.
    pub side: BooleanRegionSide,
    /// Keep/discard action for this region under the requested Boolean operation.
    pub action: BooleanRegionAction,
    /// Region boundary in the source face parameter domain.
    pub uv_loop: Vec<Vec2>,
    /// Interior sample used for classification.
    pub sample_uv: Vec2,
    /// Model-space sample point when the source support surface can be evaluated.
    pub sample_point: Option<Point3>,
    /// True when the region should be emitted with reversed orientation.
    pub orientation_reversed: bool,
}

/// Report for classifying every face region, including unsplit faces.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanRegionClassificationReport {
    /// Boolean operation being classified.
    pub operation: BooleanOp,
    /// Number of faces considered.
    pub face_count: usize,
    /// Number of staged split edges considered.
    pub split_count: usize,
    /// Number of faces with at least one active split curve.
    pub affected_face_count: usize,
    /// Number of emitted regions.
    pub region_count: usize,
    /// Number of regions emitted from split faces.
    pub split_region_count: usize,
    /// Number of regions emitted from unsplit faces.
    pub unsplit_region_count: usize,
    /// Number of regions with a keep action.
    pub kept_region_count: usize,
    /// Number of regions with a discard action.
    pub discarded_region_count: usize,
    /// Number of regions whose action remains ambiguous.
    pub ambiguous_region_count: usize,
    /// Per-region classifications.
    pub regions: Vec<BooleanClassifiedRegion>,
    /// Underlying per-split side classification.
    pub split_classification: BooleanClassificationReport,
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

/// Report from tolerance-aware Boolean mesh healing.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanMeshHealingReport {
    /// Healing tolerance in model units.
    pub tolerance: f64,
    /// Number of input vertices.
    pub input_vertices: usize,
    /// Number of input triangles.
    pub input_triangles: usize,
    /// Number of vertices after gap-closing sewing.
    pub sewn_vertices: usize,
    /// Number of triangles after sewing collapsed degenerate faces.
    pub sewn_triangles: usize,
    /// Number of output vertices after compaction.
    pub output_vertices: usize,
    /// Number of output triangles after all filtering.
    pub output_triangles: usize,
    /// Number of input vertices merged by tolerance sewing.
    pub merged_vertices: usize,
    /// Triangles removed because sewing collapsed two or more corners.
    pub removed_degenerate_triangles: usize,
    /// Triangles removed because they contain an edge at or below tolerance.
    pub removed_tiny_edge_triangles: usize,
    /// Triangles removed because they are numerically skinny or near zero-area.
    pub removed_sliver_triangles: usize,
    /// Duplicate or reversed duplicate triangles removed after filtering.
    pub removed_duplicate_triangles: usize,
    /// True when the healed mesh validated as a closed half-edge solid.
    pub manifold: bool,
    /// Topology validation error when the healed mesh is still not a valid manifold.
    pub solid_error: Option<TopologyError>,
    /// Underlying vertex-clustering/sewing diagnostics.
    pub sewing_report: SewingReport,
}

/// Output from healing a raw Boolean triangle mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanMeshHealingOutput {
    /// Healed and compacted triangle mesh.
    pub mesh: HealedTriangleMesh,
    /// Valid closed half-edge solid when manifold validation succeeds.
    pub solid: Option<Solid>,
    /// Healing diagnostics.
    pub report: BooleanMeshHealingReport,
}

/// Generalized Boolean output assembled from classified kept regions.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedBooleanOutput {
    /// Boolean operation used for region classification.
    pub operation: BooleanOp,
    /// Kept classified regions used to assemble the output mesh.
    pub kept_regions: Vec<BooleanClassifiedRegion>,
    /// Healed and compacted output mesh.
    pub mesh: HealedTriangleMesh,
    /// Valid closed half-edge solid when the kept regions form a watertight shell.
    pub solid: Option<Solid>,
    /// Topology validation error when the kept regions produce only a partial or invalid shell.
    pub solid_error: Option<TopologyError>,
    /// Number of input classified regions considered.
    pub input_region_count: usize,
    /// Number of kept regions emitted into the output mesh.
    pub kept_region_count: usize,
    /// Number of classified regions discarded before meshing.
    pub discarded_region_count: usize,
    /// Number of ambiguous regions ignored before meshing.
    pub ambiguous_region_count: usize,
    /// Mesh repair diagnostics from the final healing pass.
    pub healing_report: BooleanMeshHealingReport,
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

/// Relationship found for one left/right face pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanFacePairStatus {
    /// The two finite faces do not intersect.
    Disjoint,
    /// The faces touch at a point or along a numerically tiny interval.
    Touching,
    /// The faces intersect in one or more curves.
    Intersecting,
    /// The faces are coplanar or coincident over an area.
    Coincident,
    /// The pair could not be intersected with the available representation.
    Unsupported,
}

/// One intersection curve/segment recorded in the face-pair graph.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanFaceIntersectionCurve {
    /// Model-space samples on the curve.
    pub points: Vec<Point3>,
    /// Model-space curve for downstream split installation.
    pub edge_curve: EdgeCurve3D,
    /// P-curve on the left face when available.
    pub left_pcurve: Option<TrimCurve2D>,
    /// P-curve on the right face when available.
    pub right_pcurve: Option<TrimCurve2D>,
    /// Trim-ready SSI curve when a supported analytic path produced one.
    pub trim_ready: Option<TrimReadyIntersectionCurve>,
    /// Maximum residual reported by the intersection routine.
    pub max_residual: f64,
}

/// Intersection record for one pair of faces, one from each operand.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanFacePairIntersection {
    /// Face index in the left operand.
    pub left_face: FaceId,
    /// Face index in the right operand.
    pub right_face: FaceId,
    /// Pair status.
    pub status: BooleanFacePairStatus,
    /// Curve or segment intersections for this pair.
    pub curves: Vec<BooleanFaceIntersectionCurve>,
    /// Point contacts for tangencies or zero-length overlaps.
    pub contact_points: Vec<Point3>,
    /// Largest curve residual for this pair.
    pub max_residual: f64,
}

/// Full face-pair intersection graph between two solids.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanIntersectionGraph {
    /// Intersection tolerance in model units.
    pub tolerance: f64,
    /// Number of faces in the left operand.
    pub left_face_count: usize,
    /// Number of faces in the right operand.
    pub right_face_count: usize,
    /// One record for every left/right face pair.
    pub face_pairs: Vec<BooleanFacePairIntersection>,
    /// Indices into `face_pairs` for non-empty pairs adjacent to each left face.
    pub left_adjacency: Vec<Vec<usize>>,
    /// Indices into `face_pairs` for non-empty pairs adjacent to each right face.
    pub right_adjacency: Vec<Vec<usize>>,
    /// Number of non-empty face pairs.
    pub active_pair_count: usize,
    /// Total number of intersection curves across active pairs.
    pub curve_count: usize,
}

/// Classification of a tolerance merge relation between two edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanEdgeMergeKind {
    /// Endpoints match in the same or reversed order.
    Coincident,
    /// Collinear edges overlap over a finite interval but do not have identical endpoints.
    Overlapping,
    /// The edges touch at a point within tolerance.
    TangentTouch,
    /// Endpoints are close enough to merge but not exactly coincident.
    NearlyCoincident,
}

/// Classification of a tolerance merge relation between two faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanFaceMergeKind {
    /// Coplanar finite faces overlap in area.
    CoplanarOverlap,
    /// Nearly coplanar finite faces overlap in area within tolerance.
    NearlyCoincident,
    /// Faces touch at a point or along a numerically tiny interval.
    TangentTouch,
    /// Faces overlap or intersect in a finite curve.
    Overlapping,
}

/// Cross-operand vertex merge candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanMergedVertex {
    /// Vertex in the left operand.
    pub left_vertex: VertexId,
    /// Vertex in the right operand.
    pub right_vertex: VertexId,
    /// Averaged representative point for later sewing.
    pub merged_point: Point3,
    /// Distance between the two source vertices.
    pub distance: f64,
}

/// Cross-operand edge merge candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanMergedEdge {
    /// Edge in the left operand.
    pub left_edge: EdgeId,
    /// Edge in the right operand.
    pub right_edge: EdgeId,
    /// Merge/contact classification.
    pub kind: BooleanEdgeMergeKind,
    /// True when coincident endpoints match in reversed order.
    pub reversed: bool,
    /// Parametric overlap interval on the left edge, normalized to `[0, 1]`.
    pub left_interval: (f64, f64),
    /// Parametric overlap interval on the right edge, normalized to `[0, 1]`.
    pub right_interval: (f64, f64),
    /// Maximum distance used to certify this relation.
    pub max_distance: f64,
}

/// Cross-operand face merge/contact candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanMergedFace {
    /// Face in the left operand.
    pub left_face: FaceId,
    /// Face in the right operand.
    pub right_face: FaceId,
    /// Merge/contact classification.
    pub kind: BooleanFaceMergeKind,
    /// Maximum sampled plane/triangle distance used for the classification.
    pub max_distance: f64,
    /// Absolute angle between support normals, in radians.
    pub normal_angle: f64,
}

/// Tolerance-aware merge analysis for Boolean operands.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanTopologyMergeReport {
    /// Tolerance used for all merge decisions.
    pub tolerance: f64,
    /// Vertex merge candidates.
    pub vertices: Vec<BooleanMergedVertex>,
    /// Edge merge/contact candidates.
    pub edges: Vec<BooleanMergedEdge>,
    /// Face merge/contact candidates.
    pub faces: Vec<BooleanMergedFace>,
    /// Number of vertex pairs within tolerance.
    pub merged_vertex_count: usize,
    /// Number of edge pairs within tolerance.
    pub merged_edge_count: usize,
    /// Number of face pairs within tolerance.
    pub merged_face_count: usize,
    /// Number of tangent point/zero-length contacts.
    pub tangent_pair_count: usize,
    /// Number of finite edge/face overlap relations.
    pub overlapping_pair_count: usize,
    /// Number of coplanar face overlap relations.
    pub coplanar_pair_count: usize,
    /// Number of nearly coincident edge/face relations.
    pub nearly_coincident_pair_count: usize,
}

/// Intersect every face pair across two solids and build a face-pair graph.
pub fn build_face_intersection_graph(
    left: &Solid,
    right: &Solid,
    tolerance: f64,
) -> Result<BooleanIntersectionGraph, BooleanError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }
    left.validate()?;
    right.validate()?;
    let tolerance = tolerance.max(1.0e-9);
    let mut face_pairs = Vec::with_capacity(left.faces.len() * right.faces.len());
    let mut left_adjacency = vec![Vec::<usize>::new(); left.faces.len()];
    let mut right_adjacency = vec![Vec::<usize>::new(); right.faces.len()];
    let mut active_pair_count = 0;
    let mut curve_count = 0;

    for (left_face, left_pairs) in left_adjacency.iter_mut().enumerate() {
        for (right_face, right_pairs) in right_adjacency.iter_mut().enumerate() {
            let record = intersect_face_pair(left, right, left_face, right_face, tolerance);
            let pair_index = face_pairs.len();
            if !matches!(
                record.status,
                BooleanFacePairStatus::Disjoint | BooleanFacePairStatus::Unsupported
            ) {
                active_pair_count += 1;
                left_pairs.push(pair_index);
                right_pairs.push(pair_index);
            }
            curve_count += record.curves.len();
            face_pairs.push(record);
        }
    }

    Ok(BooleanIntersectionGraph {
        tolerance,
        left_face_count: left.faces.len(),
        right_face_count: right.faces.len(),
        face_pairs,
        left_adjacency,
        right_adjacency,
        active_pair_count,
        curve_count,
    })
}

/// Analyze cross-operand coincident, near-coincident, overlapping, and tangent topology.
///
/// This pass does not mutate either solid. It creates the merge/contact records
/// needed by later Boolean shell rewriting: vertices to sew, edges to reuse or
/// split, and face pairs that require special overlap/tangent handling instead
/// of ordinary crossing SSI.
pub fn analyze_boolean_topology_merges(
    left: &Solid,
    right: &Solid,
    tolerance: f64,
) -> Result<BooleanTopologyMergeReport, BooleanError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }
    left.validate()?;
    right.validate()?;
    let tolerance = tolerance.max(1.0e-9);

    let vertices = analyze_vertex_merges(left, right, tolerance);
    let edges = analyze_edge_merges(left, right, tolerance);
    let faces = analyze_face_merges(left, right, tolerance);

    let tangent_pair_count = edges
        .iter()
        .filter(|edge| edge.kind == BooleanEdgeMergeKind::TangentTouch)
        .count()
        + faces
            .iter()
            .filter(|face| face.kind == BooleanFaceMergeKind::TangentTouch)
            .count();
    let overlapping_pair_count = edges
        .iter()
        .filter(|edge| edge.kind == BooleanEdgeMergeKind::Overlapping)
        .count()
        + faces
            .iter()
            .filter(|face| {
                matches!(
                    face.kind,
                    BooleanFaceMergeKind::Overlapping | BooleanFaceMergeKind::CoplanarOverlap
                )
            })
            .count();
    let coplanar_pair_count = faces
        .iter()
        .filter(|face| face.kind == BooleanFaceMergeKind::CoplanarOverlap)
        .count();
    let nearly_coincident_pair_count = edges
        .iter()
        .filter(|edge| edge.kind == BooleanEdgeMergeKind::NearlyCoincident)
        .count()
        + faces
            .iter()
            .filter(|face| face.kind == BooleanFaceMergeKind::NearlyCoincident)
            .count();

    Ok(BooleanTopologyMergeReport {
        tolerance,
        merged_vertex_count: vertices.len(),
        merged_edge_count: edges.len(),
        merged_face_count: faces.len(),
        vertices,
        edges,
        faces,
        tangent_pair_count,
        overlapping_pair_count,
        coplanar_pair_count,
        nearly_coincident_pair_count,
    })
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

/// Split all affected face domains by their active split curves and classify every region.
///
/// Split faces are partitioned in UV space by all active staged split p-curves on
/// that face. Faces with no active split curves still produce one region and are
/// classified against the opposite operand when that operand's labelled face
/// subset is closed. The output is a region graph for later trim-loop rewriting;
/// it does not mutate the shell topology.
pub fn classify_boolean_regions(
    solid: &Solid,
    face_operands: &[BooleanOperand],
    operation: BooleanOp,
    tolerance: f64,
) -> Result<BooleanRegionClassificationReport, BooleanError> {
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
    let split_classification = classify_split_faces(solid, face_operands, operation, tolerance)?;
    let mut active_by_face = vec![Vec::<BooleanFaceSplitClassification>::new(); solid.faces.len()];
    for split in &split_classification.classifications {
        if split.status == BooleanSplitStatus::Active {
            active_by_face[split.a.face].push(split.a.clone());
            active_by_face[split.b.face].push(split.b.clone());
        }
    }
    let operand_closed = [
        operand_subset_is_closed(solid, face_operands, BooleanOperand::Left),
        operand_subset_is_closed(solid, face_operands, BooleanOperand::Right),
    ];
    let context = RegionClassificationContext {
        solid,
        face_operands,
        operand_closed,
        operation,
        tolerance,
    };

    let mut regions = Vec::new();
    let mut affected_face_count = 0;
    for face in 0..solid.faces.len() {
        let outer = face_outer_loop_polyline(solid, face, tolerance)?;
        let operand = face_operands[face];
        let active_splits = &active_by_face[face];
        if active_splits.is_empty() {
            regions.push(classified_region_from_loop(
                &context,
                face,
                operand,
                &[],
                true,
                outer,
            )?);
            continue;
        }

        affected_face_count += 1;
        let cells =
            split_face_domain_by_active_curves(solid, face, &outer, active_splits, tolerance);
        for cell in cells {
            regions.push(classified_region_from_loop(
                &context,
                face,
                operand,
                active_splits,
                false,
                cell,
            )?);
        }
    }

    let region_count = regions.len();
    let split_region_count = regions.iter().filter(|region| !region.unsplit_face).count();
    let unsplit_region_count = regions.iter().filter(|region| region.unsplit_face).count();
    let kept_region_count = regions
        .iter()
        .filter(|region| action_keeps_region(region.action))
        .count();
    let discarded_region_count = regions
        .iter()
        .filter(|region| region.action == BooleanRegionAction::Discard)
        .count();
    let ambiguous_region_count = regions
        .iter()
        .filter(|region| region.action == BooleanRegionAction::Ambiguous)
        .count();

    Ok(BooleanRegionClassificationReport {
        operation,
        face_count: solid.faces.len(),
        split_count: solid.split_edges.len(),
        affected_face_count,
        region_count,
        split_region_count,
        unsplit_region_count,
        kept_region_count,
        discarded_region_count,
        ambiguous_region_count,
        regions,
        split_classification,
    })
}

/// Heal a raw Boolean triangle mesh by closing gaps, removing slivers/tiny edges,
/// and validating manifoldness.
///
/// The function is intentionally conservative. It performs tolerance sewing,
/// removes collapsed, tiny-edge, skinny, and duplicate triangles, compacts the
/// remaining mesh, then attempts to build a closed half-edge [`Solid`]. When
/// validation fails, the repaired mesh and the topology error are still returned
/// so callers can inspect why an arbitrary input could not be made manifold.
pub fn heal_boolean_triangle_mesh(
    vertices: Vec<Point3>,
    triangles: &[[usize; 3]],
    tolerance: f64,
) -> Result<BooleanMeshHealingOutput, BooleanError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }
    let tolerance = tolerance.max(1.0e-9);
    let input_vertices = vertices.len();
    let input_triangles = triangles.len();
    let sewn = Solid::sew_triangle_mesh(vertices, triangles, tolerance)?;
    let sewing_report = sewn.report.clone();
    let filter = filter_boolean_triangles(&sewn.vertices, &sewn.triangles, tolerance);
    let mesh = compact_triangle_mesh(sewn.vertices, &filter.triangles);
    let (solid, solid_error) = if mesh.triangles.is_empty() {
        (None, None)
    } else {
        match Solid::from_triangle_mesh(mesh.vertices.clone(), &mesh.triangles) {
            Ok(mut solid) => {
                for edge in &mut solid.edges {
                    edge.tolerance = tolerance;
                }
                (Some(solid), None)
            }
            Err(error) => (None, Some(error)),
        }
    };

    Ok(BooleanMeshHealingOutput {
        report: BooleanMeshHealingReport {
            tolerance,
            input_vertices,
            input_triangles,
            sewn_vertices: sewing_report.output_vertices,
            sewn_triangles: sewing_report.output_triangles,
            output_vertices: mesh.vertices.len(),
            output_triangles: mesh.triangles.len(),
            merged_vertices: sewing_report.merged_vertices,
            removed_degenerate_triangles: sewing_report.removed_degenerate_triangles,
            removed_tiny_edge_triangles: filter.removed_tiny_edge_triangles,
            removed_sliver_triangles: filter.removed_sliver_triangles,
            removed_duplicate_triangles: filter.removed_duplicate_triangles,
            manifold: solid.is_some(),
            solid_error: solid_error.clone(),
            sewing_report,
        },
        mesh,
        solid,
    })
}

/// Assemble classified Boolean regions into a healed closed output solid when possible.
///
/// This is the generalized output stage for the current pipeline. It consumes
/// the region graph from [`classify_boolean_regions`], triangulates every kept
/// region on its source face, runs the Boolean mesh-healing pass, and returns a
/// validated closed [`Solid`] when the kept regions form a watertight manifold.
/// If the regions are still partial, the repaired mesh and topology error are
/// returned for diagnostics.
pub fn build_closed_boolean_output(
    solid: &Solid,
    classification: &BooleanRegionClassificationReport,
    tolerance: f64,
) -> Result<ClosedBooleanOutput, BooleanError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BooleanError::InvalidInput("tolerance must be nonnegative"));
    }
    if classification.face_count != solid.faces.len() {
        return Err(BooleanError::InvalidInput(
            "classification must match the source solid face count",
        ));
    }
    solid.validate()?;
    let tolerance = tolerance.max(1.0e-9);
    let kept_regions: Vec<BooleanClassifiedRegion> = classification
        .regions
        .iter()
        .filter(|region| action_keeps_region(region.action))
        .cloned()
        .collect();
    let discarded_region_count = classification
        .regions
        .iter()
        .filter(|region| region.action == BooleanRegionAction::Discard)
        .count();
    let ambiguous_region_count = classification
        .regions
        .iter()
        .filter(|region| region.action == BooleanRegionAction::Ambiguous)
        .count();

    let raw_mesh = triangulate_classified_regions(solid, &kept_regions, tolerance)?;
    let healed = heal_boolean_triangle_mesh(raw_mesh.vertices, &raw_mesh.triangles, tolerance)?;
    let solid_error = healed.report.solid_error.clone();
    Ok(ClosedBooleanOutput {
        operation: classification.operation,
        input_region_count: classification.regions.len(),
        kept_region_count: kept_regions.len(),
        discarded_region_count,
        ambiguous_region_count,
        kept_regions,
        mesh: healed.mesh,
        solid: healed.solid,
        solid_error,
        healing_report: healed.report,
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
    let raw_mesh = triangulate_healed_regions(&regions, tolerance)?;
    let (mesh, solid, solid_error, sewing_report) = if raw_mesh.triangles.is_empty() {
        (raw_mesh, None, None, None)
    } else {
        let healed = heal_boolean_triangle_mesh(raw_mesh.vertices, &raw_mesh.triangles, tolerance)?;
        let solid_error = healed.report.solid_error.clone();
        let sewing_report = Some(healed.report.sewing_report.clone());
        (healed.mesh, healed.solid, solid_error, sewing_report)
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

#[derive(Clone, Copy, Debug)]
struct SegmentDistance {
    distance: f64,
    left_t: f64,
    right_t: f64,
}

#[derive(Clone, Debug, Default)]
struct TriangleFilterReport {
    triangles: Vec<[usize; 3]>,
    removed_tiny_edge_triangles: usize,
    removed_sliver_triangles: usize,
    removed_duplicate_triangles: usize,
}

fn filter_boolean_triangles(
    vertices: &[Point3],
    triangles: &[[usize; 3]],
    tolerance: f64,
) -> TriangleFilterReport {
    let mut report = TriangleFilterReport::default();
    let mut canonical_triangles = Vec::<[usize; 3]>::new();
    for tri in triangles.iter().copied() {
        let a = vertices[tri[0]];
        let b = vertices[tri[1]];
        let c = vertices[tri[2]];
        if triangle_has_tiny_edge(a, b, c, tolerance) {
            report.removed_tiny_edge_triangles += 1;
            continue;
        }
        if triangle_is_sliver(a, b, c, tolerance) {
            report.removed_sliver_triangles += 1;
            continue;
        }
        let canonical = canonical_triangle(tri);
        if canonical_triangles.contains(&canonical) {
            report.removed_duplicate_triangles += 1;
            continue;
        }
        canonical_triangles.push(canonical);
        report.triangles.push(tri);
    }
    report
}

fn triangle_has_tiny_edge(a: Point3, b: Point3, c: Point3, tolerance: f64) -> bool {
    a.distance(b) <= tolerance || b.distance(c) <= tolerance || c.distance(a) <= tolerance
}

fn triangle_is_sliver(a: Point3, b: Point3, c: Point3, tolerance: f64) -> bool {
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);
    let max_edge = ab.max(bc).max(ca);
    if max_edge <= tolerance {
        return true;
    }
    let area = (b - a).cross(c - a).norm() * 0.5;
    if area <= tolerance * max_edge {
        return true;
    }
    let quality = 4.0 * 3.0_f64.sqrt() * area / (ab * ab + bc * bc + ca * ca);
    quality <= 1.0e-6
}

fn canonical_triangle(mut triangle: [usize; 3]) -> [usize; 3] {
    triangle.sort_unstable();
    triangle
}

fn compact_triangle_mesh(vertices: Vec<Point3>, triangles: &[[usize; 3]]) -> HealedTriangleMesh {
    let mut remap = vec![usize::MAX; vertices.len()];
    let mut compact_vertices = Vec::<Point3>::new();
    let mut compact_triangles = Vec::<[usize; 3]>::with_capacity(triangles.len());
    for tri in triangles {
        let mut compact = [0; 3];
        for (corner, vertex) in tri.iter().copied().enumerate() {
            if remap[vertex] == usize::MAX {
                remap[vertex] = compact_vertices.len();
                compact_vertices.push(vertices[vertex]);
            }
            compact[corner] = remap[vertex];
        }
        compact_triangles.push(compact);
    }
    HealedTriangleMesh {
        vertices: compact_vertices,
        triangles: compact_triangles,
    }
}

fn analyze_vertex_merges(left: &Solid, right: &Solid, tolerance: f64) -> Vec<BooleanMergedVertex> {
    let mut vertices = Vec::new();
    for (left_vertex, left_record) in left.vertices.iter().enumerate() {
        for (right_vertex, right_record) in right.vertices.iter().enumerate() {
            let distance = left_record.point.distance(right_record.point);
            if distance <= tolerance {
                vertices.push(BooleanMergedVertex {
                    left_vertex,
                    right_vertex,
                    merged_point: (left_record.point + right_record.point) * 0.5,
                    distance,
                });
            }
        }
    }
    vertices
}

fn analyze_edge_merges(left: &Solid, right: &Solid, tolerance: f64) -> Vec<BooleanMergedEdge> {
    let mut edges = Vec::new();
    for left_edge in 0..left.edges.len() {
        let Some((left_start, left_end)) = left.edge_points(left_edge) else {
            continue;
        };
        for right_edge in 0..right.edges.len() {
            let Some((right_start, right_end)) = right.edge_points(right_edge) else {
                continue;
            };
            if let Some(record) = classify_edge_merge(
                left_edge,
                left_start,
                left_end,
                right_edge,
                right_start,
                right_end,
                tolerance,
            ) {
                edges.push(record);
            }
        }
    }
    edges
}

fn classify_edge_merge(
    left_edge: EdgeId,
    left_start: Point3,
    left_end: Point3,
    right_edge: EdgeId,
    right_start: Point3,
    right_end: Point3,
    tolerance: f64,
) -> Option<BooleanMergedEdge> {
    let exact_tolerance = exact_merge_tolerance(tolerance);
    let forward_distance = left_start
        .distance(right_start)
        .max(left_end.distance(right_end));
    let reverse_distance = left_start
        .distance(right_end)
        .max(left_end.distance(right_start));
    if forward_distance <= tolerance || reverse_distance <= tolerance {
        let reversed = reverse_distance < forward_distance;
        let max_distance = forward_distance.min(reverse_distance);
        return Some(BooleanMergedEdge {
            left_edge,
            right_edge,
            kind: if max_distance <= exact_tolerance {
                BooleanEdgeMergeKind::Coincident
            } else {
                BooleanEdgeMergeKind::NearlyCoincident
            },
            reversed,
            left_interval: (0.0, 1.0),
            right_interval: (0.0, 1.0),
            max_distance,
        });
    }

    if let Some(record) = classify_collinear_edge_overlap(
        left_edge,
        left_start,
        left_end,
        right_edge,
        right_start,
        right_end,
        tolerance,
    ) {
        return Some(record);
    }

    let distance = segment_segment_distance(left_start, left_end, right_start, right_end);
    if distance.distance <= tolerance {
        return Some(BooleanMergedEdge {
            left_edge,
            right_edge,
            kind: BooleanEdgeMergeKind::TangentTouch,
            reversed: false,
            left_interval: (distance.left_t, distance.left_t),
            right_interval: (distance.right_t, distance.right_t),
            max_distance: distance.distance,
        });
    }
    None
}

fn classify_collinear_edge_overlap(
    left_edge: EdgeId,
    left_start: Point3,
    left_end: Point3,
    right_edge: EdgeId,
    right_start: Point3,
    right_end: Point3,
    tolerance: f64,
) -> Option<BooleanMergedEdge> {
    let left_vec = left_end - left_start;
    let right_vec = right_end - right_start;
    let left_len = left_vec.norm();
    let right_len = right_vec.norm();
    if left_len <= tolerance || right_len <= tolerance {
        return None;
    }
    let left_axis = left_vec / left_len;
    let right_axis = right_vec / right_len;
    if left_axis.cross(right_axis).norm() > angular_merge_tolerance() {
        return None;
    }
    if point_line_distance(right_start, left_start, left_axis) > tolerance
        || point_line_distance(right_end, left_start, left_axis) > tolerance
    {
        return None;
    }

    let right_left_t0 = (right_start - left_start).dot(left_axis) / left_len;
    let right_left_t1 = (right_end - left_start).dot(left_axis) / left_len;
    let overlap_min = 0.0_f64.max(right_left_t0.min(right_left_t1));
    let overlap_max = 1.0_f64.min(right_left_t0.max(right_left_t1));
    if (overlap_max - overlap_min) * left_len <= tolerance {
        return None;
    }

    let left_overlap_start = left_start.lerp(left_end, overlap_min);
    let left_overlap_end = left_start.lerp(left_end, overlap_max);
    let right_interval_a =
        ((left_overlap_start - right_start).dot(right_axis) / right_len).clamp(0.0, 1.0);
    let right_interval_b =
        ((left_overlap_end - right_start).dot(right_axis) / right_len).clamp(0.0, 1.0);
    let max_distance = point_segment_distance3(left_overlap_start, right_start, right_end).max(
        point_segment_distance3(left_overlap_end, right_start, right_end),
    );
    Some(BooleanMergedEdge {
        left_edge,
        right_edge,
        kind: BooleanEdgeMergeKind::Overlapping,
        reversed: left_axis.dot(right_axis) < 0.0,
        left_interval: (overlap_min, overlap_max),
        right_interval: (
            right_interval_a.min(right_interval_b),
            right_interval_a.max(right_interval_b),
        ),
        max_distance,
    })
}

fn analyze_face_merges(left: &Solid, right: &Solid, tolerance: f64) -> Vec<BooleanMergedFace> {
    let mut faces = Vec::new();
    for left_face in 0..left.faces.len() {
        for right_face in 0..right.faces.len() {
            if let Some(record) = classify_face_merge(left, right, left_face, right_face, tolerance)
            {
                faces.push(record);
            }
        }
    }
    faces
}

fn classify_face_merge(
    left: &Solid,
    right: &Solid,
    left_face: FaceId,
    right_face: FaceId,
    tolerance: f64,
) -> Option<BooleanMergedFace> {
    let left_triangle = face_triangle(left, left_face)?;
    let right_triangle = face_triangle(right, right_face)?;
    if !triangle_bounds_overlap(&left_triangle, &right_triangle, tolerance) {
        return None;
    }
    let left_plane = Plane::from_points(left_triangle[0], left_triangle[1], left_triangle[2])?;
    let right_plane = Plane::from_points(right_triangle[0], right_triangle[1], right_triangle[2])?;
    let normal_angle = normal_angle(left_plane.normal, right_plane.normal);
    let left_to_right = triangle_signed_distances(&left_triangle, right_plane);
    let right_to_left = triangle_signed_distances(&right_triangle, left_plane);
    let max_plane_distance = left_to_right
        .into_iter()
        .chain(right_to_left)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let parallel = left_plane.normal.cross(right_plane.normal).norm() <= angular_merge_tolerance();

    if parallel
        && max_plane_distance <= tolerance
        && coplanar_triangles_overlap(
            &left_triangle,
            &right_triangle,
            left_plane.normal,
            tolerance,
        )
    {
        return Some(BooleanMergedFace {
            left_face,
            right_face,
            kind: if max_plane_distance <= exact_merge_tolerance(tolerance) {
                BooleanFaceMergeKind::CoplanarOverlap
            } else {
                BooleanFaceMergeKind::NearlyCoincident
            },
            max_distance: max_plane_distance,
            normal_angle,
        });
    }

    match intersect_triangle_faces(left, right, left_face, right_face, tolerance) {
        TriangleFaceIntersection::Touching(_) => Some(BooleanMergedFace {
            left_face,
            right_face,
            kind: BooleanFaceMergeKind::TangentTouch,
            max_distance: triangle_triangle_distance(&left_triangle, &right_triangle),
            normal_angle,
        }),
        TriangleFaceIntersection::Segment { max_residual, .. } => Some(BooleanMergedFace {
            left_face,
            right_face,
            kind: BooleanFaceMergeKind::Overlapping,
            max_distance: max_residual,
            normal_angle,
        }),
        TriangleFaceIntersection::Coincident => Some(BooleanMergedFace {
            left_face,
            right_face,
            kind: BooleanFaceMergeKind::CoplanarOverlap,
            max_distance: max_plane_distance,
            normal_angle,
        }),
        TriangleFaceIntersection::Disjoint | TriangleFaceIntersection::Unsupported => {
            let distance = triangle_triangle_distance(&left_triangle, &right_triangle);
            if distance <= tolerance {
                Some(BooleanMergedFace {
                    left_face,
                    right_face,
                    kind: BooleanFaceMergeKind::TangentTouch,
                    max_distance: distance,
                    normal_angle,
                })
            } else {
                None
            }
        }
    }
}

fn segment_segment_distance(a0: Point3, a1: Point3, b0: Point3, b1: Point3) -> SegmentDistance {
    let u = a1 - a0;
    let v = b1 - b0;
    let w = a0 - b0;
    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w);
    let e = v.dot(w);
    let denom = a * c - b * b;
    let mut s_numer;
    let mut s_denom = denom;
    let mut t_numer;
    let mut t_denom = denom;
    const EPS: f64 = 1.0e-12;

    if denom < EPS {
        s_numer = 0.0;
        s_denom = 1.0;
        t_numer = e;
        t_denom = c;
    } else {
        s_numer = b * e - c * d;
        t_numer = a * e - b * d;
        if s_numer < 0.0 {
            s_numer = 0.0;
            t_numer = e;
            t_denom = c;
        } else if s_numer > s_denom {
            s_numer = s_denom;
            t_numer = e + b;
            t_denom = c;
        }
    }

    if t_numer < 0.0 {
        t_numer = 0.0;
        if -d < 0.0 {
            s_numer = 0.0;
        } else if -d > a {
            s_numer = s_denom;
        } else {
            s_numer = -d;
            s_denom = a;
        }
    } else if t_numer > t_denom {
        t_numer = t_denom;
        if -d + b < 0.0 {
            s_numer = 0.0;
        } else if -d + b > a {
            s_numer = s_denom;
        } else {
            s_numer = -d + b;
            s_denom = a;
        }
    }

    let left_t = if s_numer.abs() < EPS {
        0.0
    } else {
        s_numer / s_denom
    }
    .clamp(0.0, 1.0);
    let right_t = if t_numer.abs() < EPS {
        0.0
    } else {
        t_numer / t_denom
    }
    .clamp(0.0, 1.0);
    SegmentDistance {
        distance: (w + u * left_t - v * right_t).norm(),
        left_t,
        right_t,
    }
}

fn triangle_triangle_distance(a: &[Point3; 3], b: &[Point3; 3]) -> f64 {
    let mut distance = f64::INFINITY;
    for point in a {
        distance = distance.min(point_triangle_distance(*point, b));
    }
    for point in b {
        distance = distance.min(point_triangle_distance(*point, a));
    }
    for i in 0..3 {
        let a0 = a[i];
        let a1 = a[(i + 1) % 3];
        for j in 0..3 {
            let b0 = b[j];
            let b1 = b[(j + 1) % 3];
            distance = distance.min(segment_segment_distance(a0, a1, b0, b1).distance);
        }
    }
    distance
}

fn point_line_distance(point: Point3, line_origin: Point3, line_axis: Vec3) -> f64 {
    (point - line_origin).cross(line_axis).norm()
}

fn normal_angle(a: Vec3, b: Vec3) -> f64 {
    a.normalized()
        .dot(b.normalized())
        .abs()
        .clamp(-1.0, 1.0)
        .acos()
}

fn exact_merge_tolerance(tolerance: f64) -> f64 {
    tolerance.clamp(1.0e-12, 1.0e-9)
}

fn angular_merge_tolerance() -> f64 {
    1.0e-7
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TriangleFaceIntersection {
    Disjoint,
    Touching(Point3),
    Segment {
        start: Point3,
        end: Point3,
        max_residual: f64,
    },
    Coincident,
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TrianglePlaneCut {
    Point(Point3),
    Segment { start: Point3, end: Point3 },
}

fn intersect_face_pair(
    left: &Solid,
    right: &Solid,
    left_face: FaceId,
    right_face: FaceId,
    tolerance: f64,
) -> BooleanFacePairIntersection {
    let mut curves = analytic_face_pair_curves(left, right, left_face, right_face, tolerance);
    let mut contact_points = Vec::new();
    let mut status = if curves.is_empty() {
        BooleanFacePairStatus::Disjoint
    } else {
        BooleanFacePairStatus::Intersecting
    };

    if curves.is_empty() {
        match intersect_triangle_faces(left, right, left_face, right_face, tolerance) {
            TriangleFaceIntersection::Disjoint => status = BooleanFacePairStatus::Disjoint,
            TriangleFaceIntersection::Touching(point) => {
                status = BooleanFacePairStatus::Touching;
                push_unique_point3(&mut contact_points, point, tolerance);
            }
            TriangleFaceIntersection::Segment {
                start,
                end,
                max_residual,
            } => {
                status = BooleanFacePairStatus::Intersecting;
                if let Some(curve) = graph_curve_from_points(
                    vec![start, end],
                    &left.faces[left_face].surface,
                    &right.faces[right_face].surface,
                    tolerance,
                    max_residual,
                ) {
                    curves.push(curve);
                }
            }
            TriangleFaceIntersection::Coincident => status = BooleanFacePairStatus::Coincident,
            TriangleFaceIntersection::Unsupported => status = BooleanFacePairStatus::Unsupported,
        }
    }

    let max_residual = curves
        .iter()
        .map(|curve| curve.max_residual)
        .fold(0.0_f64, f64::max);
    BooleanFacePairIntersection {
        left_face,
        right_face,
        status,
        curves,
        contact_points,
        max_residual,
    }
}

fn analytic_face_pair_curves(
    left: &Solid,
    right: &Solid,
    left_face: FaceId,
    right_face: FaceId,
    tolerance: f64,
) -> Vec<BooleanFaceIntersectionCurve> {
    let left_surface = &left.faces[left_face].surface;
    let right_surface = &right.faces[right_face].surface;
    let mut curves = match (left_surface, right_surface) {
        (FaceSurface::Nurbs(left_nurbs), FaceSurface::Nurbs(right_nurbs)) => {
            intersect_nurbs_surfaces(left_nurbs, right_nurbs, 24, 24, tolerance)
                .into_iter()
                .map(trim_ready_curve_to_graph_curve)
                .collect()
        }
        (FaceSurface::Plane(plane), FaceSurface::Nurbs(nurbs)) => {
            intersect_plane_nurbs_surface(*plane, nurbs, 32, 32, tolerance)
                .into_iter()
                .filter_map(|polyline| {
                    graph_curve_from_polyline(polyline, left_surface, right_surface, tolerance)
                })
                .collect()
        }
        (FaceSurface::Nurbs(nurbs), FaceSurface::Plane(plane)) => {
            intersect_plane_nurbs_surface(*plane, nurbs, 32, 32, tolerance)
                .into_iter()
                .filter_map(|polyline| {
                    graph_curve_from_polyline(polyline, left_surface, right_surface, tolerance)
                })
                .collect()
        }
        _ => Vec::new(),
    };
    curves.retain(|curve| {
        graph_curve_intersects_face_domains(left, right, left_face, right_face, curve, tolerance)
    });
    curves
}

fn trim_ready_curve_to_graph_curve(
    curve: TrimReadyIntersectionCurve,
) -> BooleanFaceIntersectionCurve {
    let points = curve
        .points
        .iter()
        .map(|sample| sample.point)
        .collect::<Vec<_>>();
    BooleanFaceIntersectionCurve {
        points,
        edge_curve: curve.edge_curve.clone(),
        left_pcurve: Some(curve.a_pcurve.clone()),
        right_pcurve: Some(curve.b_pcurve.clone()),
        max_residual: curve.max_residual,
        trim_ready: Some(curve),
    }
}

fn graph_curve_from_polyline(
    polyline: IntersectionPolyline,
    left_surface: &FaceSurface,
    right_surface: &FaceSurface,
    tolerance: f64,
) -> Option<BooleanFaceIntersectionCurve> {
    graph_curve_from_points(
        polyline.points,
        left_surface,
        right_surface,
        tolerance,
        polyline.max_residual,
    )
}

fn graph_curve_from_points(
    points: Vec<Point3>,
    left_surface: &FaceSurface,
    right_surface: &FaceSurface,
    tolerance: f64,
    max_residual: f64,
) -> Option<BooleanFaceIntersectionCurve> {
    if points.len() < 2 {
        return None;
    }
    let edge_curve = edge_curve_from_points(&points, tolerance)?;
    let left_pcurve = pcurve_from_model_points(left_surface, &points, tolerance);
    let right_pcurve = pcurve_from_model_points(right_surface, &points, tolerance);
    Some(BooleanFaceIntersectionCurve {
        points,
        edge_curve,
        left_pcurve,
        right_pcurve,
        trim_ready: None,
        max_residual,
    })
}

fn edge_curve_from_points(points: &[Point3], tolerance: f64) -> Option<EdgeCurve3D> {
    let start = *points.first()?;
    let end = *points.last()?;
    if start.distance(end) <= tolerance {
        return None;
    }
    if points.len() == 2 {
        Some(EdgeCurve3D::line_segment(start, end))
    } else {
        Some(EdgeCurve3D::Polyline {
            points: points.to_vec(),
        })
    }
}

fn pcurve_from_model_points(
    surface: &FaceSurface,
    points: &[Point3],
    tolerance: f64,
) -> Option<TrimCurve2D> {
    let mut uv_points = Vec::with_capacity(points.len());
    let mut seed = None;
    for point in points {
        let uv = surface.project_point_near(*point, seed, tolerance.max(1.0e-7))?;
        seed = Some(uv);
        if uv_points
            .last()
            .is_none_or(|last| vec2_distance(*last, uv) > tolerance)
        {
            uv_points.push(uv);
        }
    }
    if uv_points.len() < 2 {
        return None;
    }
    if uv_points.len() == 2 {
        Some(TrimCurve2D::LineSegment {
            start: uv_points[0],
            end: uv_points[1],
        })
    } else {
        Some(TrimCurve2D::Polyline { points: uv_points })
    }
}

fn graph_curve_intersects_face_domains(
    left: &Solid,
    right: &Solid,
    left_face: FaceId,
    right_face: FaceId,
    curve: &BooleanFaceIntersectionCurve,
    tolerance: f64,
) -> bool {
    let left_ok = curve
        .left_pcurve
        .as_ref()
        .is_none_or(|pcurve| pcurve_samples_visible_on_face(left, left_face, pcurve, tolerance));
    let right_ok = curve
        .right_pcurve
        .as_ref()
        .is_none_or(|pcurve| pcurve_samples_visible_on_face(right, right_face, pcurve, tolerance));
    left_ok && right_ok
}

fn pcurve_samples_visible_on_face(
    solid: &Solid,
    face: FaceId,
    pcurve: &TrimCurve2D,
    tolerance: f64,
) -> bool {
    let Some(samples) = sample_pcurve_points(pcurve) else {
        return true;
    };
    samples.into_iter().any(|uv| {
        matches!(
            classify_face_uv(solid, face, uv, tolerance),
            Some(LoopPointLocation::Inside | LoopPointLocation::Boundary)
        )
    })
}

fn intersect_triangle_faces(
    left: &Solid,
    right: &Solid,
    left_face: FaceId,
    right_face: FaceId,
    tolerance: f64,
) -> TriangleFaceIntersection {
    let Some(left_triangle) = face_triangle(left, left_face) else {
        return TriangleFaceIntersection::Unsupported;
    };
    let Some(right_triangle) = face_triangle(right, right_face) else {
        return TriangleFaceIntersection::Unsupported;
    };
    if !triangle_bounds_overlap(&left_triangle, &right_triangle, tolerance) {
        return TriangleFaceIntersection::Disjoint;
    }

    let Some(left_plane) = Plane::from_points(left_triangle[0], left_triangle[1], left_triangle[2])
    else {
        return TriangleFaceIntersection::Unsupported;
    };
    let Some(right_plane) =
        Plane::from_points(right_triangle[0], right_triangle[1], right_triangle[2])
    else {
        return TriangleFaceIntersection::Unsupported;
    };
    let eps = tolerance.max(1.0e-12);
    let left_to_right = triangle_signed_distances(&left_triangle, right_plane);
    if distances_are_strictly_one_sided(left_to_right, eps) {
        return TriangleFaceIntersection::Disjoint;
    }
    let right_to_left = triangle_signed_distances(&right_triangle, left_plane);
    if distances_are_strictly_one_sided(right_to_left, eps) {
        return TriangleFaceIntersection::Disjoint;
    }

    let line_direction = left_plane.normal.cross(right_plane.normal);
    if line_direction.norm() <= eps {
        let coplanar = left_to_right.iter().all(|distance| distance.abs() <= eps)
            && right_to_left.iter().all(|distance| distance.abs() <= eps);
        return if coplanar
            && coplanar_triangles_overlap(&left_triangle, &right_triangle, left_plane.normal, eps)
        {
            TriangleFaceIntersection::Coincident
        } else {
            TriangleFaceIntersection::Disjoint
        };
    }

    let Some(left_cut) = cut_triangle_with_plane(&left_triangle, left_to_right, eps) else {
        return TriangleFaceIntersection::Disjoint;
    };
    let Some(right_cut) = cut_triangle_with_plane(&right_triangle, right_to_left, eps) else {
        return TriangleFaceIntersection::Disjoint;
    };
    let axis = line_direction.normalized();
    let (left_min, left_max) = cut_interval(left_cut, axis);
    let (right_min, right_max) = cut_interval(right_cut, axis);
    let overlap_min = left_min.max(right_min);
    let overlap_max = left_max.min(right_max);
    if overlap_max < overlap_min - eps {
        return TriangleFaceIntersection::Disjoint;
    }
    if (overlap_max - overlap_min).abs() <= eps {
        let point = point_on_cut_at(left_cut, axis, (overlap_min + overlap_max) * 0.5);
        return TriangleFaceIntersection::Touching(point);
    }

    let start = point_on_cut_at(left_cut, axis, overlap_min);
    let end = point_on_cut_at(left_cut, axis, overlap_max);
    if start.distance(end) <= eps {
        TriangleFaceIntersection::Touching((start + end) * 0.5)
    } else {
        TriangleFaceIntersection::Segment {
            start,
            end,
            max_residual: max_triangle_residual(
                start,
                end,
                left_plane,
                right_plane,
                left_triangle,
                right_triangle,
            ),
        }
    }
}

fn face_triangle(solid: &Solid, face: FaceId) -> Option<[Point3; 3]> {
    if face >= solid.faces.len() {
        return None;
    }
    let vertices = solid.face_vertices(face);
    Some([
        solid.vertices.get(vertices[0])?.point,
        solid.vertices.get(vertices[1])?.point,
        solid.vertices.get(vertices[2])?.point,
    ])
}

fn triangle_signed_distances(triangle: &[Point3; 3], plane: Plane) -> [f64; 3] {
    [
        plane.signed_distance(triangle[0]),
        plane.signed_distance(triangle[1]),
        plane.signed_distance(triangle[2]),
    ]
}

fn distances_are_strictly_one_sided(distances: [f64; 3], tolerance: f64) -> bool {
    distances.iter().all(|distance| *distance > tolerance)
        || distances.iter().all(|distance| *distance < -tolerance)
}

fn cut_triangle_with_plane(
    triangle: &[Point3; 3],
    distances: [f64; 3],
    tolerance: f64,
) -> Option<TrianglePlaneCut> {
    let mut points = Vec::<Point3>::new();
    for index in 0..3 {
        if distances[index].abs() <= tolerance {
            push_unique_point3(&mut points, triangle[index], tolerance);
        }
        let next = (index + 1) % 3;
        let a = distances[index];
        let b = distances[next];
        if (a > tolerance && b < -tolerance) || (a < -tolerance && b > tolerance) {
            let t = a / (a - b);
            push_unique_point3(
                &mut points,
                triangle[index].lerp(triangle[next], t),
                tolerance,
            );
        }
    }
    match points.len() {
        0 => None,
        1 => Some(TrianglePlaneCut::Point(points[0])),
        _ => {
            let (start, end) = furthest_point_pair(&points);
            if start.distance(end) <= tolerance {
                Some(TrianglePlaneCut::Point(start))
            } else {
                Some(TrianglePlaneCut::Segment { start, end })
            }
        }
    }
}

fn cut_interval(cut: TrianglePlaneCut, axis: Vec3) -> (f64, f64) {
    match cut {
        TrianglePlaneCut::Point(point) => {
            let t = point.dot(axis);
            (t, t)
        }
        TrianglePlaneCut::Segment { start, end } => {
            let a = start.dot(axis);
            let b = end.dot(axis);
            (a.min(b), a.max(b))
        }
    }
}

fn point_on_cut_at(cut: TrianglePlaneCut, axis: Vec3, target: f64) -> Point3 {
    match cut {
        TrianglePlaneCut::Point(point) => point,
        TrianglePlaneCut::Segment { start, end } => {
            let a = start.dot(axis);
            let b = end.dot(axis);
            let denom = b - a;
            if denom.abs() <= f64::EPSILON {
                (start + end) * 0.5
            } else {
                start.lerp(end, ((target - a) / denom).clamp(0.0, 1.0))
            }
        }
    }
}

fn max_triangle_residual(
    start: Point3,
    end: Point3,
    left_plane: Plane,
    right_plane: Plane,
    left_triangle: [Point3; 3],
    right_triangle: [Point3; 3],
) -> f64 {
    [start, end]
        .into_iter()
        .map(|point| {
            left_plane
                .signed_distance(point)
                .abs()
                .max(right_plane.signed_distance(point).abs())
                .max(point_triangle_distance(point, &left_triangle))
                .max(point_triangle_distance(point, &right_triangle))
        })
        .fold(0.0_f64, f64::max)
}

fn triangle_bounds_overlap(a: &[Point3; 3], b: &[Point3; 3], tolerance: f64) -> bool {
    let (a_min, a_max) = triangle_bounds(a);
    let (b_min, b_max) = triangle_bounds(b);
    a_min.x <= b_max.x + tolerance
        && a_max.x + tolerance >= b_min.x
        && a_min.y <= b_max.y + tolerance
        && a_max.y + tolerance >= b_min.y
        && a_min.z <= b_max.z + tolerance
        && a_max.z + tolerance >= b_min.z
}

fn triangle_bounds(triangle: &[Point3; 3]) -> (Point3, Point3) {
    let mut min = triangle[0];
    let mut max = triangle[0];
    for point in triangle.iter().skip(1) {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        min.z = min.z.min(point.z);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
        max.z = max.z.max(point.z);
    }
    (min, max)
}

fn coplanar_triangles_overlap(
    a: &[Point3; 3],
    b: &[Point3; 3],
    normal: Vec3,
    tolerance: f64,
) -> bool {
    let a2 = [
        project_coplanar_point(a[0], normal),
        project_coplanar_point(a[1], normal),
        project_coplanar_point(a[2], normal),
    ];
    let b2 = [
        project_coplanar_point(b[0], normal),
        project_coplanar_point(b[1], normal),
        project_coplanar_point(b[2], normal),
    ];
    !has_separating_axis_2d(&a2, &b2, tolerance) && !has_separating_axis_2d(&b2, &a2, tolerance)
}

fn has_separating_axis_2d(a: &[Vec2; 3], b: &[Vec2; 3], tolerance: f64) -> bool {
    for index in 0..3 {
        let edge = a[(index + 1) % 3] - a[index];
        let axis = Vec2::new(-edge.y, edge.x);
        let (a_min, a_max) = projected_interval_2d(a, axis);
        let (b_min, b_max) = projected_interval_2d(b, axis);
        if a_max < b_min - tolerance || b_max < a_min - tolerance {
            return true;
        }
    }
    false
}

fn projected_interval_2d(points: &[Vec2; 3], axis: Vec2) -> (f64, f64) {
    let mut min = points[0].dot(axis);
    let mut max = min;
    for point in points.iter().skip(1) {
        let t = point.dot(axis);
        min = min.min(t);
        max = max.max(t);
    }
    (min, max)
}

fn project_coplanar_point(point: Point3, normal: Vec3) -> Vec2 {
    let ax = normal.x.abs();
    let ay = normal.y.abs();
    let az = normal.z.abs();
    if ax >= ay && ax >= az {
        Vec2::new(point.y, point.z)
    } else if ay >= az {
        Vec2::new(point.x, point.z)
    } else {
        Vec2::new(point.x, point.y)
    }
}

fn point_triangle_distance(point: Point3, triangle: &[Point3; 3]) -> f64 {
    let Some(plane) = Plane::from_points(triangle[0], triangle[1], triangle[2]) else {
        return f64::INFINITY;
    };
    let projected = plane.project(point);
    if point_in_triangle_3d(projected, triangle, plane.normal) {
        point.distance(projected)
    } else {
        point_segment_distance3(point, triangle[0], triangle[1])
            .min(point_segment_distance3(point, triangle[1], triangle[2]))
            .min(point_segment_distance3(point, triangle[2], triangle[0]))
    }
}

fn point_in_triangle_3d(point: Point3, triangle: &[Point3; 3], normal: Vec3) -> bool {
    let a = triangle[0];
    let b = triangle[1];
    let c = triangle[2];
    normal.dot((b - a).cross(point - a)) >= -1.0e-9
        && normal.dot((c - b).cross(point - b)) >= -1.0e-9
        && normal.dot((a - c).cross(point - c)) >= -1.0e-9
}

fn point_segment_distance3(point: Point3, a: Point3, b: Point3) -> f64 {
    let edge = b - a;
    let len2 = edge.dot(edge);
    if len2 <= f64::EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(edge) / len2).clamp(0.0, 1.0);
    point.distance(a + edge * t)
}

fn furthest_point_pair(points: &[Point3]) -> (Point3, Point3) {
    let mut best = (points[0], points[0]);
    let mut best_distance = 0.0;
    for (i, a) in points.iter().enumerate() {
        for b in points.iter().skip(i + 1) {
            let distance = a.distance(*b);
            if distance > best_distance {
                best_distance = distance;
                best = (*a, *b);
            }
        }
    }
    best
}

fn push_unique_point3(points: &mut Vec<Point3>, point: Point3, tolerance: f64) {
    if points
        .iter()
        .all(|existing| existing.distance(point) > tolerance)
    {
        points.push(point);
    }
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

fn triangulate_classified_regions(
    solid: &Solid,
    regions: &[BooleanClassifiedRegion],
    tolerance: f64,
) -> Result<HealedTriangleMesh, BooleanError> {
    let mut mesh = HealedTriangleMesh::default();
    for region in regions {
        if region.uv_loop.len() < 3 {
            return Err(BooleanError::Unsupported);
        }
        let face = solid
            .faces
            .get(region.face)
            .ok_or(BooleanError::Unsupported)?;
        let mut flat = Vec::with_capacity(region.uv_loop.len() * 2);
        let mut vertices = Vec::with_capacity(region.uv_loop.len());
        for uv in &region.uv_loop {
            flat.push(uv.x);
            flat.push(uv.y);
            vertices.push(
                face.surface
                    .evaluate(*uv)
                    .ok_or(BooleanError::Unsupported)?,
            );
        }
        let local_triangles =
            earcutr::earcut(&flat, &[], 2).map_err(|_| BooleanError::Unsupported)?;
        if local_triangles.len() % 3 != 0 {
            return Err(BooleanError::Unsupported);
        }
        let indices: Vec<usize> = vertices
            .iter()
            .map(|vertex| weld_mesh_vertex(&mut mesh.vertices, *vertex, tolerance))
            .collect();
        let normal = face
            .surface
            .normal_at(region.uv_loop[0])
            .ok_or(BooleanError::Unsupported)?;
        let desired = if region.orientation_reversed {
            -normal
        } else {
            normal
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

struct RegionClassificationContext<'a> {
    solid: &'a Solid,
    face_operands: &'a [BooleanOperand],
    operand_closed: [bool; 2],
    operation: BooleanOp,
    tolerance: f64,
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

fn classified_region_from_loop(
    context: &RegionClassificationContext<'_>,
    face: FaceId,
    operand: BooleanOperand,
    active_splits: &[BooleanFaceSplitClassification],
    unsplit_face: bool,
    mut uv_loop: Vec<Vec2>,
) -> Result<BooleanClassifiedRegion, BooleanError> {
    let tolerance = context.tolerance;
    cleanup_loop(&mut uv_loop, tolerance);
    if !polygon_is_usable(&uv_loop, tolerance) {
        return Err(BooleanError::Unsupported);
    }
    if polygon_signed_area(&uv_loop) < 0.0 {
        uv_loop.reverse();
    }
    let sample_uv = polygon_centroid(&uv_loop);
    let sample_point = context.solid.faces[face].surface.evaluate(sample_uv);
    let side = classify_region_sample_side(
        context,
        face,
        operand,
        active_splits,
        sample_uv,
        sample_point,
    );
    let action = action_for_side(context.operation, operand, other_operand(operand), side);
    Ok(BooleanClassifiedRegion {
        face,
        operand,
        source_splits: active_splits
            .iter()
            .map(|split| split.split_index)
            .collect(),
        unsplit_face,
        side,
        action,
        uv_loop,
        sample_uv,
        sample_point,
        orientation_reversed: action == BooleanRegionAction::KeepReversed,
    })
}

fn classify_region_sample_side(
    context: &RegionClassificationContext<'_>,
    face: FaceId,
    operand: BooleanOperand,
    active_splits: &[BooleanFaceSplitClassification],
    sample_uv: Vec2,
    sample_point: Option<Point3>,
) -> BooleanRegionSide {
    let tolerance = context.tolerance;
    let other = other_operand(operand);
    if !operand_has_faces(context.face_operands, other) {
        return BooleanRegionSide::OutsideOther;
    }
    if context.operand_closed[operand_index(other)] {
        if let Some(point) = sample_point {
            let side = classify_point_against_operand(
                context.solid,
                context.face_operands,
                other,
                point,
                tolerance,
            );
            if side != BooleanRegionSide::Ambiguous {
                return side;
            }
        }
    }
    classify_region_sample_from_splits(context.solid, face, active_splits, sample_uv, tolerance)
        .unwrap_or(BooleanRegionSide::Ambiguous)
}

fn classify_region_sample_from_splits(
    solid: &Solid,
    face: FaceId,
    active_splits: &[BooleanFaceSplitClassification],
    sample_uv: Vec2,
    tolerance: f64,
) -> Option<BooleanRegionSide> {
    let mut saw_inside = false;
    let mut saw_outside = false;
    for split in active_splits {
        let pcurve = &solid.faces[face].split_curves[split.split_index].pcurve;
        let Some((uv, tangent)) = representative_pcurve_sample(pcurve, tolerance) else {
            continue;
        };
        let signed = tangent.cross(sample_uv - uv);
        if signed.abs() <= tolerance {
            continue;
        }
        match if signed > 0.0 {
            split.left_of_curve
        } else {
            split.right_of_curve
        } {
            BooleanRegionSide::InsideOther => saw_inside = true,
            BooleanRegionSide::OutsideOther => saw_outside = true,
            BooleanRegionSide::Ambiguous => {}
        }
    }
    if saw_outside {
        Some(BooleanRegionSide::OutsideOther)
    } else if saw_inside {
        Some(BooleanRegionSide::InsideOther)
    } else {
        None
    }
}

fn split_face_domain_by_active_curves(
    solid: &Solid,
    face: FaceId,
    outer: &[Vec2],
    active_splits: &[BooleanFaceSplitClassification],
    tolerance: f64,
) -> Vec<Vec<Vec2>> {
    let mut cells = vec![outer.to_vec()];
    for split in active_splits {
        let pcurve = &solid.faces[face].split_curves[split.split_index].pcurve;
        let Some((point, tangent)) = representative_pcurve_sample(pcurve, tolerance) else {
            continue;
        };
        if vec2_norm(tangent) <= tolerance {
            continue;
        }

        let mut next_cells = Vec::<Vec<Vec2>>::new();
        for cell in &cells {
            let left = clip_polygon_by_oriented_line(cell, point, tangent, true, tolerance);
            let right = clip_polygon_by_oriented_line(cell, point, tangent, false, tolerance);
            if polygon_is_usable(&left, tolerance) {
                push_unique_polygon(&mut next_cells, left, tolerance);
            }
            if polygon_is_usable(&right, tolerance) {
                push_unique_polygon(&mut next_cells, right, tolerance);
            }
        }
        if !next_cells.is_empty() {
            cells = next_cells;
        } else {
            break;
        }
    }
    cells
}

fn clip_polygon_by_oriented_line(
    polygon: &[Vec2],
    point: Vec2,
    tangent: Vec2,
    keep_left: bool,
    tolerance: f64,
) -> Vec<Vec2> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let mut output = Vec::new();
    for index in 0..polygon.len() {
        let current = polygon[index];
        let next = polygon[(index + 1) % polygon.len()];
        let current_signed = tangent.cross(current - point);
        let next_signed = tangent.cross(next - point);
        let current_inside = if keep_left {
            current_signed >= -tolerance
        } else {
            current_signed <= tolerance
        };
        let next_inside = if keep_left {
            next_signed >= -tolerance
        } else {
            next_signed <= tolerance
        };

        if current_inside && next_inside {
            push_unique_uv_with_tolerance(&mut output, next, tolerance);
        } else if current_inside && !next_inside {
            push_unique_uv_with_tolerance(
                &mut output,
                line_edge_intersection(current, next, current_signed, next_signed),
                tolerance,
            );
        } else if !current_inside && next_inside {
            push_unique_uv_with_tolerance(
                &mut output,
                line_edge_intersection(current, next, current_signed, next_signed),
                tolerance,
            );
            push_unique_uv_with_tolerance(&mut output, next, tolerance);
        }
    }
    cleanup_loop(&mut output, tolerance);
    output
}

fn line_edge_intersection(start: Vec2, end: Vec2, start_signed: f64, end_signed: f64) -> Vec2 {
    let denom = start_signed - end_signed;
    if denom.abs() <= f64::EPSILON {
        (start + end) * 0.5
    } else {
        start + (end - start) * (start_signed / denom).clamp(0.0, 1.0)
    }
}

fn face_outer_loop_polyline(
    solid: &Solid,
    face: FaceId,
    tolerance: f64,
) -> Result<Vec<Vec2>, BooleanError> {
    let outer_loop = solid.faces[face]
        .trim_loops
        .iter()
        .find(|trim_loop| trim_loop.kind == TrimLoopKind::Outer)
        .ok_or(BooleanError::Unsupported)?;
    let mut outer = trim_loop_polyline(outer_loop).ok_or(BooleanError::Unsupported)?;
    cleanup_loop(&mut outer, tolerance);
    if !polygon_is_usable(&outer, tolerance) {
        return Err(BooleanError::Unsupported);
    }
    if polygon_signed_area(&outer) < 0.0 {
        outer.reverse();
    }
    Ok(outer)
}

fn classify_point_against_operand(
    solid: &Solid,
    face_operands: &[BooleanOperand],
    operand: BooleanOperand,
    point: Point3,
    tolerance: f64,
) -> BooleanRegionSide {
    let ray = Vec3::new(0.873_217, 0.371_391, 0.317_173).normalized();
    let mut hits = Vec::<f64>::new();
    for (face, face_operand) in face_operands.iter().copied().enumerate() {
        if face_operand != operand {
            continue;
        }
        let Some(triangle) = face_triangle(solid, face) else {
            return BooleanRegionSide::Ambiguous;
        };
        if point_triangle_distance(point, &triangle) <= tolerance {
            return BooleanRegionSide::Ambiguous;
        }
        if let Some(t) = ray_triangle_intersection(point, ray, &triangle, tolerance) {
            push_unique_scalar(&mut hits, t, tolerance.max(1.0e-8));
        }
    }
    if hits.len() % 2 == 1 {
        BooleanRegionSide::InsideOther
    } else {
        BooleanRegionSide::OutsideOther
    }
}

fn ray_triangle_intersection(
    origin: Point3,
    direction: Vec3,
    triangle: &[Point3; 3],
    tolerance: f64,
) -> Option<f64> {
    let edge1 = triangle[1] - triangle[0];
    let edge2 = triangle[2] - triangle[0];
    let h = direction.cross(edge2);
    let det = edge1.dot(h);
    if det.abs() <= tolerance {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = origin - triangle[0];
    let u = inv_det * s.dot(h);
    if u < -tolerance || u > 1.0 + tolerance {
        return None;
    }
    let q = s.cross(edge1);
    let v = inv_det * direction.dot(q);
    if v < -tolerance || u + v > 1.0 + tolerance {
        return None;
    }
    let t = inv_det * edge2.dot(q);
    if t > tolerance {
        Some(t)
    } else {
        None
    }
}

fn operand_subset_is_closed(
    solid: &Solid,
    face_operands: &[BooleanOperand],
    operand: BooleanOperand,
) -> bool {
    let mut has_face = false;
    for (face, face_operand) in face_operands.iter().copied().enumerate() {
        if face_operand != operand {
            continue;
        }
        has_face = true;
        let start = solid.faces[face].halfedge;
        let mut halfedge = start;
        let mut guard = 0;
        loop {
            let Some(twin) = solid.halfedges[halfedge].twin else {
                return false;
            };
            if face_operands[solid.halfedges[twin].face] != operand {
                return false;
            }
            halfedge = solid.halfedges[halfedge].next;
            guard += 1;
            if halfedge == start {
                break;
            }
            if guard > solid.halfedges.len() {
                return false;
            }
        }
    }
    has_face
}

fn operand_has_faces(face_operands: &[BooleanOperand], operand: BooleanOperand) -> bool {
    face_operands.contains(&operand)
}

fn other_operand(operand: BooleanOperand) -> BooleanOperand {
    match operand {
        BooleanOperand::Left => BooleanOperand::Right,
        BooleanOperand::Right => BooleanOperand::Left,
    }
}

fn operand_index(operand: BooleanOperand) -> usize {
    match operand {
        BooleanOperand::Left => 0,
        BooleanOperand::Right => 1,
    }
}

fn polygon_is_usable(points: &[Vec2], tolerance: f64) -> bool {
    points.len() >= 3 && polygon_signed_area(points).abs() > tolerance * tolerance
}

fn polygon_centroid(points: &[Vec2]) -> Vec2 {
    let mut twice_area = 0.0;
    let mut centroid = Vec2::new(0.0, 0.0);
    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        let cross = a.cross(b);
        twice_area += cross;
        centroid = centroid + (a + b) * cross;
    }
    if twice_area.abs() <= f64::EPSILON {
        let sum = points
            .iter()
            .copied()
            .fold(Vec2::new(0.0, 0.0), |acc, point| acc + point);
        sum * (1.0 / points.len() as f64)
    } else {
        centroid * (1.0 / (3.0 * twice_area))
    }
}

fn push_unique_polygon(polygons: &mut Vec<Vec<Vec2>>, mut polygon: Vec<Vec2>, tolerance: f64) {
    cleanup_loop(&mut polygon, tolerance);
    if !polygon_is_usable(&polygon, tolerance) {
        return;
    }
    if polygons
        .iter()
        .any(|existing| polygons_are_equivalent(existing, &polygon, tolerance))
    {
        return;
    }
    polygons.push(polygon);
}

fn polygons_are_equivalent(a: &[Vec2], b: &[Vec2], tolerance: f64) -> bool {
    (polygon_signed_area(a).abs() - polygon_signed_area(b).abs()).abs() <= tolerance * tolerance
        && vec2_distance(polygon_centroid(a), polygon_centroid(b)) <= tolerance
}

fn push_unique_scalar(values: &mut Vec<f64>, value: f64, tolerance: f64) {
    if values
        .iter()
        .all(|existing| (*existing - value).abs() > tolerance)
    {
        values.push(value);
    }
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

fn push_unique_uv_with_tolerance(points: &mut Vec<Vec2>, point: Vec2, tolerance: f64) {
    if points
        .last()
        .is_none_or(|last| vec2_distance(*last, point) > tolerance)
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
