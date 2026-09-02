//! Coordinate distance and geometry routines.
//!
//! The API in this module mirrors the array-oriented part of
//! `MDAnalysis.lib.distances` while keeping the coordinate representation
//! lightweight: one point is an `[f64; 3]`, and a collection is a slice of
//! those points.  Angles and dihedrals are returned in radians.  Periodic
//! boundary conditions currently support orthorhombic boxes represented by
//! their three positive lengths.

use std::fmt;

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
    }
}
