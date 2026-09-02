//! Coordinate distance and geometry routines.
//!
//! The API in this module mirrors the array-oriented part of
//! `MDAnalysis.lib.distances` while keeping the coordinate representation
//! lightweight: one point is an `[f64; 3]`, and a collection is a slice of
//! those points.  Angles and dihedrals are returned in radians.  Pairwise
//! routines support orthorhombic boxes represented by their three positive
//! lengths; the explicit triclinic helpers accept the usual six box
//! dimensions `[a, b, c, alpha, beta, gamma]`.

use std::fmt;

use crate::geometry::{Matrix3, Vec3};

/// A point in Cartesian space.
pub type Coordinate = [f64; 3];

/// Errors returned by the distance routines.
#[derive(Clone, Debug, PartialEq)]
pub enum DistanceError {
    /// Coordinate collections that should describe corresponding atoms have
    /// different lengths.
    LengthMismatch {
        /// Name of the operation that detected the mismatch.
        operation: &'static str,
        /// Number of points in the first collection.
        expected: usize,
        /// Number of points in the collection that differed.
        found: usize,
    },
    /// A periodic box must contain finite, strictly positive lengths.
    InvalidBox(Coordinate),
    /// A triclinic box must contain finite positive lengths and angles.
    InvalidTriclinicBox([f64; 6]),
    /// A cutoff is not finite or is negative.
    InvalidCutoff(f64),
    /// The lower cutoff is larger than the upper cutoff.
    CutoffOrder { min: f64, max: f64 },
}

impl fmt::Display for DistanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                operation,
                expected,
                found,
            } => write!(
                formatter,
                "{operation} requires equal coordinate lengths (expected {expected}, found {found})"
            ),
            Self::InvalidBox(lengths) => write!(
                formatter,
                "orthorhombic box lengths must be finite and positive (got {lengths:?})"
            ),
            Self::InvalidTriclinicBox(dimensions) => write!(
                formatter,
                "triclinic box dimensions are invalid (got {dimensions:?})"
            ),
            Self::InvalidCutoff(cutoff) => {
                write!(
                    formatter,
                    "cutoff must be finite and non-negative (got {cutoff})"
                )
            }
            Self::CutoffOrder { min, max } => {
                write!(
                    formatter,
                    "minimum cutoff {min} exceeds maximum cutoff {max}"
                )
            }
        }
    }
}

impl std::error::Error for DistanceError {}

/// Result type used by this module.
pub type Result<T> = std::result::Result<T, DistanceError>;

/// Index pairs and their corresponding distances returned by capped queries.
#[derive(Clone, Debug, PartialEq)]
pub struct PairDistances {
    pub pairs: Vec<(usize, usize)>,
    pub distances: Vec<f64>,
}

/// Calculate distances for corresponding pairs of points.
pub fn calc_bonds(
    first: &[[f64; 3]],
    second: &[[f64; 3]],
    box_lengths: Option<Coordinate>,
) -> Result<Vec<f64>> {
    ensure_same_len("calc_bonds", first.len(), second.len())?;
    let box_lengths = valid_box(box_lengths)?;
    Ok(first
        .iter()
        .zip(second)
        .map(|(&a, &b)| norm(minimum_image(sub(a, b), box_lengths)))
        .collect())
}

/// Calculate one bond distance.
pub fn calc_bond(
    first: Coordinate,
    second: Coordinate,
    box_lengths: Option<Coordinate>,
) -> Result<f64> {
    Ok(calc_bonds(&[first], &[second], box_lengths)?[0])
}

/// Calculate angles for corresponding point triplets.
pub fn calc_angles(
    first: &[[f64; 3]],
    centers: &[[f64; 3]],
    last: &[[f64; 3]],
    box_lengths: Option<Coordinate>,
) -> Result<Vec<f64>> {
    ensure_same_len("calc_angles", first.len(), centers.len())?;
    ensure_same_len("calc_angles", first.len(), last.len())?;
    let box_lengths = valid_box(box_lengths)?;
    Ok(first
        .iter()
        .zip(centers)
        .zip(last)
        .map(|((&a, &b), &c)| {
            let first_bond = minimum_image(sub(a, b), box_lengths);
            let last_bond = minimum_image(sub(c, b), box_lengths);
            angle_between(first_bond, last_bond)
        })
        .collect())
}

/// Calculate one angle in radians.
pub fn calc_angle(
    first: Coordinate,
    center: Coordinate,
    last: Coordinate,
    box_lengths: Option<Coordinate>,
) -> Result<f64> {
    Ok(calc_angles(&[first], &[center], &[last], box_lengths)?[0])
}

/// Calculate dihedral angles for corresponding point quadruplets.
pub fn calc_dihedrals(
    first: &[[f64; 3]],
    second: &[[f64; 3]],
    third: &[[f64; 3]],
    fourth: &[[f64; 3]],
    box_lengths: Option<Coordinate>,
) -> Result<Vec<f64>> {
    ensure_same_len("calc_dihedrals", first.len(), second.len())?;
    ensure_same_len("calc_dihedrals", first.len(), third.len())?;
    ensure_same_len("calc_dihedrals", first.len(), fourth.len())?;
    let box_lengths = valid_box(box_lengths)?;
    Ok(first
        .iter()
        .zip(second)
        .zip(third)
        .zip(fourth)
        .map(|(((&a, &b), &c), &d)| {
            let ab = minimum_image(sub(b, a), box_lengths);
            let bc = minimum_image(sub(c, b), box_lengths);
            let cd = minimum_image(sub(d, c), box_lengths);
            dihedral_from_bonds(ab, bc, cd)
        })
        .collect())
}

/// Calculate one dihedral angle in radians.
pub fn calc_dihedral(
    first: Coordinate,
    second: Coordinate,
    third: Coordinate,
    fourth: Coordinate,
    box_lengths: Option<Coordinate>,
) -> Result<f64> {
    Ok(calc_dihedrals(&[first], &[second], &[third], &[fourth], box_lengths)?[0])
}

/// Return all pairwise distances between two collections.
///
/// The outer dimension is `reference.len()` and the inner dimension is
/// `configuration.len()`, matching the shape of MDAnalysis' result array.
pub fn distance_array(
    reference: &[[f64; 3]],
    configuration: &[[f64; 3]],
    box_lengths: Option<Coordinate>,
) -> Result<Vec<Vec<f64>>> {
    let box_lengths = valid_box(box_lengths)?;
    Ok(reference
        .iter()
        .map(|&a| {
            configuration
                .iter()
                .map(|&b| norm(minimum_image(sub(a, b), box_lengths)))
                .collect()
        })
        .collect())
}

/// Return distances for all unique pairs in one collection.
///
/// Pairs are ordered `(0, 1), (0, 2), ..., (1, 2), ...`, as in
/// `MDAnalysis.lib.distances.self_distance_array`.
pub fn self_distance_array(
    reference: &[[f64; 3]],
    box_lengths: Option<Coordinate>,
) -> Result<Vec<f64>> {
    let box_lengths = valid_box(box_lengths)?;
    let pair_count = reference
        .len()
        .saturating_mul(reference.len().saturating_sub(1))
        / 2;
    let mut distances = Vec::with_capacity(pair_count);
    for i in 0..reference.len() {
        for j in (i + 1)..reference.len() {
            distances.push(norm(minimum_image(
                sub(reference[i], reference[j]),
                box_lengths,
            )));
        }
    }
    Ok(distances)
}

/// Return all pairs whose distance is inside a cutoff interval.
///
/// The maximum cutoff is inclusive.  When `min_cutoff` is supplied, the
/// minimum is exclusive, matching MDAnalysis' `(min, max]` convention. Returned pairs are in row-major order and
/// have matching entries in the returned distance vector.
pub fn capped_distance(
    reference: &[[f64; 3]],
    configuration: &[[f64; 3]],
    max_cutoff: f64,
    min_cutoff: Option<f64>,
    box_lengths: Option<Coordinate>,
) -> Result<PairDistances> {
    let (min_cutoff, box_lengths) = validate_cutoffs(max_cutoff, min_cutoff, box_lengths)?;
    let mut pairs = Vec::new();
    let mut distances = Vec::new();
    for (i, &a) in reference.iter().enumerate() {
        for (j, &b) in configuration.iter().enumerate() {
            let distance = norm(minimum_image(sub(a, b), box_lengths));
            if distance <= max_cutoff && min_cutoff.is_none_or(|minimum| distance > minimum) {
                pairs.push((i, j));
                distances.push(distance);
            }
        }
    }
    Ok(PairDistances { pairs, distances })
}

/// Return all unique pairs in one collection whose distance is inside a
/// cutoff interval.  Self-pairs are never included.
pub fn self_capped_distance(
    reference: &[[f64; 3]],
    max_cutoff: f64,
    min_cutoff: Option<f64>,
    box_lengths: Option<Coordinate>,
) -> Result<PairDistances> {
    let (min_cutoff, box_lengths) = validate_cutoffs(max_cutoff, min_cutoff, box_lengths)?;
    let mut pairs = Vec::new();
    let mut distances = Vec::new();
    for i in 0..reference.len() {
        for j in (i + 1)..reference.len() {
            let distance = norm(minimum_image(sub(reference[i], reference[j]), box_lengths));
            if distance <= max_cutoff && min_cutoff.is_none_or(|minimum| distance > minimum) {
                pairs.push((i, j));
                distances.push(distance);
            }
        }
    }
    Ok(PairDistances { pairs, distances })
}

/// Convert Cartesian positions to fractional unit-cell coordinates.
pub fn transform_r_to_s(coordinates: &[[f64; 3]], dimensions: [f64; 6]) -> Result<Vec<Coordinate>> {
    let matrix = triclinic_matrix(dimensions)?;
    let inverse = matrix
        .inverse()
        .ok_or(DistanceError::InvalidTriclinicBox(dimensions))?;
    Ok(coordinates
        .iter()
        .map(|&coordinate| (inverse * Vec3::from(coordinate)).to_array())
        .collect())
}

/// Convert fractional unit-cell coordinates to Cartesian positions.
pub fn transform_s_to_r(coordinates: &[[f64; 3]], dimensions: [f64; 6]) -> Result<Vec<Coordinate>> {
    let matrix = triclinic_matrix(dimensions)?;
    Ok(coordinates
        .iter()
        .map(|&coordinate| (matrix * Vec3::from(coordinate)).to_array())
        .collect())
}

/// Move Cartesian positions into the primary triclinic unit cell.
pub fn apply_pbc(coordinates: &[[f64; 3]], dimensions: [f64; 6]) -> Result<Vec<Coordinate>> {
    let fractional = transform_r_to_s(coordinates, dimensions)?;
    let wrapped: Vec<Coordinate> = fractional
        .into_iter()
        .map(|coordinate| {
            [
                coordinate[0].rem_euclid(1.0),
                coordinate[1].rem_euclid(1.0),
                coordinate[2].rem_euclid(1.0),
            ]
        })
        .collect();
    transform_s_to_r(&wrapped, dimensions)
}

/// Apply the triclinic minimum-image convention to a collection of vectors.
pub fn minimize_vectors(vectors: &[[f64; 3]], dimensions: [f64; 6]) -> Result<Vec<Coordinate>> {
    let matrix = triclinic_matrix(dimensions)?;
    let inverse = matrix
        .inverse()
        .ok_or(DistanceError::InvalidTriclinicBox(dimensions))?;
    Ok(vectors
        .iter()
        .map(|&vector| minimize_triclinic_vector(vector, matrix, inverse))
        .collect())
}

/// Apply the triclinic minimum-image convention to one vector.
pub fn minimum_image_triclinic(vector: Coordinate, dimensions: [f64; 6]) -> Result<Coordinate> {
    Ok(minimize_vectors(&[vector], dimensions)?[0])
}

fn ensure_same_len(operation: &'static str, expected: usize, found: usize) -> Result<()> {
    (expected == found)
        .then_some(())
        .ok_or(DistanceError::LengthMismatch {
            operation,
            expected,
            found,
        })
}

fn valid_box(box_lengths: Option<Coordinate>) -> Result<Option<Coordinate>> {
    box_lengths
        .map(|lengths| {
            if lengths
                .iter()
                .all(|length| length.is_finite() && *length > 0.0)
            {
                Ok(lengths)
            } else {
                Err(DistanceError::InvalidBox(lengths))
            }
        })
        .transpose()
}

fn triclinic_matrix(dimensions: [f64; 6]) -> Result<Matrix3> {
    let [a, b, c, alpha, beta, gamma] = dimensions;
    if [a, b, c, alpha, beta, gamma]
        .iter()
        .any(|value| !value.is_finite())
        || a <= 0.0
        || b <= 0.0
        || c <= 0.0
        || !(0.0 < alpha
            && alpha < 180.0
            && 0.0 < beta
            && beta < 180.0
            && 0.0 < gamma
            && gamma < 180.0)
    {
        return Err(DistanceError::InvalidTriclinicBox(dimensions));
    }
    let vectors = crate::mdamath::triclinic_vectors(dimensions);
    let matrix = Matrix3::from_cols([
        Vec3::from(vectors[0]),
        Vec3::from(vectors[1]),
        Vec3::from(vectors[2]),
    ]);
    matrix
        .inverse()
        .map(|_| matrix)
        .ok_or(DistanceError::InvalidTriclinicBox(dimensions))
}

/// Find the shortest image of one Cartesian vector in an upper-triangular
/// triclinic lattice.  Rounding each fractional component independently is a
/// useful starting point, but is not exact for skewed cells because lattice
/// vectors are not orthogonal.  The recursive search below enumerates only
/// integer candidates that can improve that starting distance, using the
/// triangular form to prune the search from z to x.
fn minimize_triclinic_vector(vector: Coordinate, matrix: Matrix3, inverse: Matrix3) -> Coordinate {
    let vector = Vec3::from(vector);
    if !vector.x.is_finite() || !vector.y.is_finite() || !vector.z.is_finite() {
        return vector.to_array();
    }
    let fractional = inverse * vector;
    let rounded = Vec3::new(
        fractional.x.round(),
        fractional.y.round(),
        fractional.z.round(),
    );
    let mut best = vector - matrix * rounded;
    let mut best_squared = best.norm_squared();
    if !best_squared.is_finite() {
        return best.to_array();
    }

    let ax = matrix.m[0][0];
    let bx = matrix.m[0][1];
    let cx = matrix.m[0][2];
    let by = matrix.m[1][1];
    let cy = matrix.m[1][2];
    let cz = matrix.m[2][2];
    // A valid triclinic matrix has positive diagonal entries.  The checks in
    // `triclinic_matrix` also reject singular matrices, so these divisions
    // are safe here.
    let tolerance = 1.0e-12 * (1.0 + best_squared);
    let z_min = ((vector.z - (best_squared + tolerance).sqrt()) / cz).ceil() as i64;
    let z_max = ((vector.z + (best_squared + tolerance).sqrt()) / cz).floor() as i64;
    for z_index in z_min..=z_max {
        let z = z_index as f64;
        let residual_z = vector.z - cz * z;
        let remaining_y = best_squared - residual_z * residual_z;
        if remaining_y < -tolerance {
            continue;
        }
        let y_radius = remaining_y.max(0.0).sqrt();
        let y_center = (vector.y - cy * z) / by;
        let y_min = ((y_center * by - y_radius) / by).ceil() as i64;
        let y_max = ((y_center * by + y_radius) / by).floor() as i64;
        for y_index in y_min..=y_max {
            let y = y_index as f64;
            let residual_y = vector.y - cy * z - by * y;
            let remaining_x = remaining_y - residual_y * residual_y;
            if remaining_x < -tolerance {
                continue;
            }
            let x_radius = remaining_x.max(0.0).sqrt();
            let x_center = (vector.x - cx * z - bx * y) / ax;
            let x_min = ((x_center * ax - x_radius) / ax).ceil() as i64;
            let x_max = ((x_center * ax + x_radius) / ax).floor() as i64;
            for x_index in x_min..=x_max {
                let x = x_index as f64;
                let candidate =
                    Vec3::new(vector.x - ax * x - bx * y - cx * z, residual_y, residual_z);
                let candidate_squared = candidate.norm_squared();
                if candidate_squared + tolerance < best_squared {
                    best_squared = candidate_squared;
                    best = candidate;
                }
            }
        }
    }
    best.to_array()
}

fn validate_cutoffs(
    max_cutoff: f64,
    min_cutoff: Option<f64>,
    box_lengths: Option<Coordinate>,
) -> Result<(Option<f64>, Option<Coordinate>)> {
    if !max_cutoff.is_finite() || max_cutoff < 0.0 {
        return Err(DistanceError::InvalidCutoff(max_cutoff));
    }
    if let Some(minimum) = min_cutoff {
        if !minimum.is_finite() || minimum < 0.0 {
            return Err(DistanceError::InvalidCutoff(minimum));
        }
        if minimum > max_cutoff {
            return Err(DistanceError::CutoffOrder {
                min: minimum,
                max: max_cutoff,
            });
        }
    }
    Ok((min_cutoff, valid_box(box_lengths)?))
}

fn sub(first: Coordinate, second: Coordinate) -> Coordinate {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn dot(first: Coordinate, second: Coordinate) -> f64 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn cross(first: Coordinate, second: Coordinate) -> Coordinate {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn norm(vector: Coordinate) -> f64 {
    dot(vector, vector).sqrt()
}

fn minimum_image(mut displacement: Coordinate, box_lengths: Option<Coordinate>) -> Coordinate {
    if let Some(lengths) = box_lengths {
        for axis in 0..3 {
            displacement[axis] -= lengths[axis] * (displacement[axis] / lengths[axis]).round();
        }
    }
    displacement
}

fn angle_between(first: Coordinate, second: Coordinate) -> f64 {
    let denominator = norm(first) * norm(second);
    if denominator == 0.0 {
        return 0.0;
    }
    (dot(first, second) / denominator).clamp(-1.0, 1.0).acos()
}

fn dihedral_from_bonds(ab: Coordinate, bc: Coordinate, cd: Coordinate) -> f64 {
    let bc_length = norm(bc);
    if bc_length == 0.0 {
        return f64::NAN;
    }
    let bc_unit = [bc[0] / bc_length, bc[1] / bc_length, bc[2] / bc_length];
    let ab_parallel = dot(ab, bc_unit);
    let cd_parallel = dot(cd, bc_unit);
    let first_normal = [
        ab[0] - ab_parallel * bc_unit[0],
        ab[1] - ab_parallel * bc_unit[1],
        ab[2] - ab_parallel * bc_unit[2],
    ];
    let second_normal = [
        cd[0] - cd_parallel * bc_unit[0],
        cd[1] - cd_parallel * bc_unit[1],
        cd[2] - cd_parallel * bc_unit[2],
    ];
    let first_length = norm(first_normal);
    let second_length = norm(second_normal);
    if first_length == 0.0 || second_length == 0.0 {
        return f64::NAN;
    }
    let sine = dot(bc_unit, cross(first_normal, second_normal)) / (first_length * second_length);
    let cosine = dot(first_normal, second_normal) / (first_length * second_length);
    sine.atan2(cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-12, "{left} != {right}");
    }

    #[test]
    fn scalar_and_batch_bonds_use_minimum_image() {
        close(
            calc_bond([9.8, 0.0, 0.0], [0.2, 0.0, 0.0], Some([10.0; 3])).unwrap(),
            0.4,
        );
        assert_eq!(
            calc_bonds(
                &[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                &[[3.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
                None,
            )
            .unwrap(),
            vec![3.0, 5.0]
        );
    }

    #[test]
    fn angles_and_dihedrals_are_in_radians() {
        close(
            calc_angle(
                [9.8, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 1.0, 0.0],
                Some([10.0; 3]),
            )
            .unwrap(),
            std::f64::consts::FRAC_PI_2,
        );
        close(
            calc_dihedral(
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                None,
            )
            .unwrap(),
            std::f64::consts::FRAC_PI_2,
        );
        assert_eq!(
            calc_angle([0.0; 3], [0.0; 3], [1.0, 0.0, 0.0], None).unwrap(),
            0.0
        );
        assert!(
            calc_dihedral([0.0; 3], [0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], None)
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn distance_wrappers_preserve_shapes_and_order() {
        let reference = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let configuration = [[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]];
        let matrix = distance_array(&reference, &configuration, None).unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![2.0, 3.0]);
        close(matrix[1][0], 5.0_f64.sqrt());
        assert_eq!(self_distance_array(&reference, None).unwrap(), vec![1.0]);
        assert!(
            distance_array(&[], &configuration, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn capped_distances_filter_and_exclude_self_pairs() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let result = capped_distance(&points[..2], &points[1..], 2.0, None, None).unwrap();
        assert_eq!(result.pairs, vec![(0, 0), (1, 0), (1, 1)]);
        assert_eq!(result.distances, vec![1.0, 0.0, 2.0]);
        let result = self_capped_distance(&points, 2.0, Some(1.0), None).unwrap();
        assert_eq!(result.pairs, vec![(1, 2)]);
        assert_eq!(result.distances, vec![2.0]);
    }

    #[test]
    fn invalid_inputs_return_errors() {
        assert!(matches!(
            calc_bonds(&[[0.0; 3]], &[], None),
            Err(DistanceError::LengthMismatch { .. })
        ));
        assert_eq!(
            calc_bond([0.0; 3], [1.0, 0.0, 0.0], Some([0.0, 1.0, 1.0])),
            Err(DistanceError::InvalidBox([0.0, 1.0, 1.0]))
        );
        assert_eq!(
            self_capped_distance(&[], -1.0, None, None),
            Err(DistanceError::InvalidCutoff(-1.0))
        );
        assert_eq!(
            self_capped_distance(&[], 1.0, Some(2.0), None),
            Err(DistanceError::CutoffOrder { min: 2.0, max: 1.0 })
        );
        assert!(matches!(
            transform_r_to_s(&[], [1.0, 1.0, 1.0, 0.0, 90.0, 90.0]),
            Err(DistanceError::InvalidTriclinicBox(_))
        ));
    }

    #[test]
    fn triclinic_coordinate_transforms_and_minimum_image() {
        let dimensions = [10.0, 11.0, 12.0, 90.0, 80.0, 70.0];
        let fractional = [[0.2, 1.25, -0.4], [0.75, 0.5, 0.1]];
        let cartesian = transform_s_to_r(&fractional, dimensions).unwrap();
        let round_trip = transform_r_to_s(&cartesian, dimensions).unwrap();
        for (actual, expected) in round_trip.iter().zip(fractional) {
            for axis in 0..3 {
                assert!((actual[axis] - expected[axis]).abs() < 1.0e-10);
            }
        }
        let wrapped = apply_pbc(&cartesian, dimensions).unwrap();
        let wrapped_fractional = transform_r_to_s(&wrapped, dimensions).unwrap();
        for coordinate in wrapped_fractional {
            assert!(coordinate.iter().all(|value| (0.0..1.0).contains(value)));
        }
        let image = minimum_image_triclinic(cartesian[0], dimensions).unwrap();
        let image_fractional = transform_r_to_s(&[image], dimensions).unwrap()[0];
        assert!(
            image_fractional
                .iter()
                .all(|value| (-0.5..=0.5).contains(value))
        );
    }

    #[test]
    fn triclinic_minimum_image_handles_skewed_lattice() {
        let dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 5.0];
        let vector = transform_s_to_r(&[[0.49, 0.49, 0.0]], dimensions).unwrap()[0];
        let minimized = minimum_image_triclinic(vector, dimensions).unwrap();
        // Independent fractional rounding would retain this vector's nearly
        // full cell length.  The nearest image subtracts the a vector.
        assert!(norm(minimized) < 1.0);
        let expected = transform_s_to_r(&[[-0.51, 0.49, 0.0]], dimensions).unwrap()[0];
        for axis in 0..3 {
            assert!((minimized[axis] - expected[axis]).abs() < 1.0e-10);
        }
    }
}
