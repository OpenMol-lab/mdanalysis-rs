//! Coordinate transformations used by molecular dynamics workflows.
//!
//! The routines in this module operate on [`crate::geometry::Vec3`] values and
//! deliberately do not depend on an atom or trajectory container.  This makes
//! them useful both for one frame and for callers that keep coordinates in a
//! custom data structure.

use crate::geometry::{Matrix3, Vec3, center_of_mass};

/// Return the 3x3 Rodrigues rotation matrix for an axis and angle.
///
/// `angle` is in radians.  A zero-length axis has no well-defined direction;
/// in that case the identity matrix is returned.  The matrix acts on column
/// vectors, i.e. `rotation_matrix(a, axis) * position`.
#[must_use]
pub fn rotation_matrix(angle: f64, axis: Vec3) -> Matrix3 {
    let axis = axis.normalized();
    if axis == Vec3::ZERO {
        return Matrix3::identity();
    }
    let (sin, cos) = angle.sin_cos();
    let one_minus_cos = 1.0 - cos;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    Matrix3::new([
        [
            cos + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin,
            x * z * one_minus_cos + y * sin,
        ],
        [
            y * x * one_minus_cos + z * sin,
            cos + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin,
        ],
        [
            z * x * one_minus_cos - y * sin,
            z * y * one_minus_cos + x * sin,
            cos + z * z * one_minus_cos,
        ],
    ])
}

/// Rotate one point around an axis through the origin.
#[must_use]
pub fn rotate_axis(position: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    rotation_matrix(angle, axis) * position
}

/// Rotate one point around an axis passing through `origin`.
#[must_use]
pub fn rotate_about(position: Vec3, axis: Vec3, angle: f64, origin: Vec3) -> Vec3 {
    origin + rotate_axis(position - origin, axis, angle)
}

/// Translate every position in place by `offset`.
pub fn translate(coordinates: &mut [Vec3], offset: Vec3) {
    for coordinate in coordinates {
        *coordinate += offset;
    }
}

/// Return translated copies of a coordinate collection.
#[must_use]
pub fn translated(coordinates: &[Vec3], offset: Vec3) -> Vec<Vec3> {
    coordinates
        .iter()
        .map(|&position| position + offset)
        .collect()
}

/// Rotate every position in place around an axis through `origin`.
pub fn rotate_positions(coordinates: &mut [Vec3], angle: f64, axis: Vec3, origin: Vec3) {
    let rotation = rotation_matrix(angle, axis);
    for coordinate in coordinates {
        *coordinate = origin + rotation * (*coordinate - origin);
    }
}

/// Result of a rigid-body fit of a coordinate set onto a reference set.
///
/// The transform maps a source position `x` to `rotation * x + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitResult {
    /// Optimal proper rotation (determinant +1).
    pub rotation: Matrix3,
    /// Translation that maps the source centroid to the reference centroid.
    pub translation: Vec3,
    /// RMSD after applying the transform.
    pub rmsd: f64,
}

impl FitResult {
    /// Apply this transform to one position.
    #[must_use]
    pub fn apply(self, position: Vec3) -> Vec3 {
        self.rotation * position + self.translation
    }

    /// Apply this transform to a coordinate collection in place.
    pub fn apply_in_place(self, coordinates: &mut [Vec3]) {
        for coordinate in coordinates {
            *coordinate = self.apply(*coordinate);
        }
    }
}

/// Compute the optimal weighted rigid transform mapping `coordinates` onto
/// `reference` (the Kabsch/Horn algorithm).
///
/// The two collections must have equal non-zero lengths.  `weights`, when
/// supplied, must have one non-negative value per point.  If all weights are
/// zero, the identity transform and zero RMSD are returned.
#[must_use]
pub fn fit_to_reference(
    coordinates: &[Vec3],
    reference: &[Vec3],
    weights: Option<&[f64]>,
) -> FitResult {
    assert_eq!(
        coordinates.len(),
        reference.len(),
        "coordinates and reference must have equal lengths"
    );
    if let Some(weights) = weights {
        assert_eq!(
            coordinates.len(),
            weights.len(),
            "coordinates and weights must have equal lengths"
        );
        assert!(
            weights
                .iter()
                .all(|weight| *weight >= 0.0 && weight.is_finite()),
            "weights must be finite and non-negative"
        );
    }
    if coordinates.is_empty() {
        return FitResult {
            rotation: Matrix3::identity(),
            translation: Vec3::ZERO,
            rmsd: 0.0,
        };
    }

    let unit_weights;
    let weights = match weights {
        Some(weights) => weights,
        None => {
            unit_weights = vec![1.0; coordinates.len()];
            &unit_weights
        }
    };
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return FitResult {
            rotation: Matrix3::identity(),
            translation: Vec3::ZERO,
            rmsd: 0.0,
        };
    }

    let source_center = center_of_mass(coordinates, weights);
    let reference_center = center_of_mass(reference, weights);

    // Horn's quaternion formulation uses M = sum(target * source^T).  The
    // principal eigenvector of the resulting 4x4 symmetric matrix is the
    // quaternion of the least-squares proper rotation.
    let mut covariance = [[0.0; 3]; 3];
    for ((&source, &target), &weight) in coordinates.iter().zip(reference).zip(weights) {
        let source = source - source_center;
        let target = target - reference_center;
        covariance[0][0] += weight * target.x * source.x;
        covariance[0][1] += weight * target.x * source.y;
        covariance[0][2] += weight * target.x * source.z;
        covariance[1][0] += weight * target.y * source.x;
        covariance[1][1] += weight * target.y * source.y;
        covariance[1][2] += weight * target.y * source.z;
        covariance[2][0] += weight * target.z * source.x;
        covariance[2][1] += weight * target.z * source.y;
        covariance[2][2] += weight * target.z * source.z;
    }

    let [sxx, sxy, sxz] = covariance[0];
    let [syx, syy, syz] = covariance[1];
    let [szx, szy, szz] = covariance[2];
    let trace = sxx + syy + szz;
    let mut n = [[0.0; 4]; 4];
    n[0] = [trace, syz - szy, szx - sxz, sxy - syx];
    n[1] = [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz];
    n[2] = [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy];
    n[3] = [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz];
    let quaternion = principal_eigenvector(n);
    // The eigensystem above is formed with target * source^T.  The quaternion
    // convention used by `quaternion_matrix` is the equivalent row-vector
    // form, so transpose it for our column-vector `Matrix3` convention.
    let rotation = quaternion_matrix(quaternion).transpose();
    let translation = reference_center - rotation * source_center;

    let transformed: Vec<Vec3> = coordinates
        .iter()
        .map(|&position| rotation * (position - source_center) + reference_center)
        .collect();
    let squared_error: f64 = transformed
        .iter()
        .zip(reference)
        .zip(weights)
        .map(|((&actual, &target), &weight)| weight * (actual - target).norm_squared())
        .sum();
    FitResult {
        rotation,
        translation,
        rmsd: (squared_error / total_weight).sqrt(),
    }
}

/// Fit and transform `coordinates` onto `reference` in place.
#[must_use]
pub fn fit_to_reference_in_place(
    coordinates: &mut [Vec3],
    reference: &[Vec3],
    weights: Option<&[f64]>,
) -> FitResult {
    let result = fit_to_reference(coordinates, reference, weights);
    result.apply_in_place(coordinates);
    result
}

/// Wrap positions into an orthorhombic periodic box `[0, box_lengths)`.
///
/// A non-positive or non-finite box length leaves that coordinate unchanged;
/// this is useful for slabs with a non-periodic dimension.
pub fn wrap_positions(coordinates: &mut [Vec3], box_lengths: Vec3) {
    for coordinate in coordinates {
        coordinate.x = wrap_component(coordinate.x, box_lengths.x);
        coordinate.y = wrap_component(coordinate.y, box_lengths.y);
        coordinate.z = wrap_component(coordinate.z, box_lengths.z);
    }
}

/// Return wrapped copies of a coordinate collection.
#[must_use]
pub fn wrapped_positions(coordinates: &[Vec3], box_lengths: Vec3) -> Vec<Vec3> {
    let mut wrapped = coordinates.to_vec();
    wrap_positions(&mut wrapped, box_lengths);
    wrapped
}

/// Wrap positions using a general (possibly triclinic) box matrix.
///
/// The rows of `box_vectors` are the three lattice vectors.  Coordinates are
/// converted to fractional coordinates, wrapped modulo one, then converted
/// back to Cartesian coordinates.  Singular box matrices are ignored.
pub fn wrap_positions_triclinic(coordinates: &mut [Vec3], box_vectors: Matrix3) {
    let Some(inverse) = box_vectors.inverse() else {
        return;
    };
    for coordinate in coordinates {
        let fractional = inverse * *coordinate;
        let wrapped = Vec3::new(
            fractional.x.rem_euclid(1.0),
            fractional.y.rem_euclid(1.0),
            fractional.z.rem_euclid(1.0),
        );
        *coordinate = box_vectors * wrapped;
    }
}

/// Shift a sequence of positions by minimum-image displacements so that it is
/// continuous across periodic boundaries.
pub fn unwrap_positions(coordinates: &mut [Vec3], box_lengths: Vec3) {
    if coordinates.len() < 2 {
        return;
    }
    for index in 1..coordinates.len() {
        let displacement = minimum_image(coordinates[index] - coordinates[index - 1], box_lengths);
        coordinates[index] = coordinates[index - 1] + displacement;
    }
}

/// Return an unwrapped copy of a periodic coordinate sequence.
#[must_use]
pub fn unwrapped_positions(coordinates: &[Vec3], box_lengths: Vec3) -> Vec<Vec3> {
    let mut unwrapped = coordinates.to_vec();
    unwrap_positions(&mut unwrapped, box_lengths);
    unwrapped
}

/// Return the minimum-image displacement in an orthorhombic box.
#[must_use]
pub fn minimum_image(displacement: Vec3, box_lengths: Vec3) -> Vec3 {
    Vec3::new(
        minimum_image_component(displacement.x, box_lengths.x),
        minimum_image_component(displacement.y, box_lengths.y),
        minimum_image_component(displacement.z, box_lengths.z),
    )
}

fn wrap_component(value: f64, period: f64) -> f64 {
    if period.is_finite() && period > 0.0 {
        value.rem_euclid(period)
    } else {
        value
    }
}

fn minimum_image_component(value: f64, period: f64) -> f64 {
    if period.is_finite() && period > 0.0 {
        value - period * (value / period).round()
    } else {
        value
    }
}

fn quaternion_matrix(quaternion: [f64; 4]) -> Matrix3 {
    let [w, x, y, z] = quaternion;
    Matrix3::new([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}

// Jacobi diagonalization for the tiny, symmetric 4x4 matrix in Horn's
// method.  This avoids pulling a full linear algebra dependency into the
// crate for a fixed-size problem.
fn principal_eigenvector(mut matrix: [[f64; 4]; 4]) -> [f64; 4] {
    let mut vectors = [[0.0; 4]; 4];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..64 {
        let mut p = 0;
        let mut q = 1;
        let mut largest = matrix[0][1].abs();
        for i in 0..4 {
            for j in (i + 1)..4 {
                if matrix[i][j].abs() > largest {
                    largest = matrix[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= 1.0e-14 {
            break;
        }
        let theta = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sin, cos) = theta.sin_cos();
        for i in 0..4 {
            if i != p && i != q {
                let aip = matrix[i][p];
                let aiq = matrix[i][q];
                matrix[i][p] = cos * aip - sin * aiq;
                matrix[p][i] = matrix[i][p];
                matrix[i][q] = sin * aip + cos * aiq;
                matrix[q][i] = matrix[i][q];
            }
        }
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = cos * cos * app - 2.0 * sin * cos * apq + sin * sin * aqq;
        matrix[q][q] = sin * sin * app + 2.0 * sin * cos * apq + cos * cos * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in &mut vectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = cos * vip - sin * viq;
            row[q] = sin * vip + cos * viq;
        }
    }
    let mut largest_index = 0;
    for index in 1..4 {
        if matrix[index][index] > matrix[largest_index][largest_index] {
            largest_index = index;
        }
    }
    let mut result = [0.0; 4];
    for index in 0..4 {
        result[index] = vectors[index][largest_index];
    }
    let norm = result.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        result.map(|value| value / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-10, "{left} != {right}");
    }

    #[test]
    fn translation_and_axis_rotation() {
        let mut points = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        translate(&mut points, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(points[0], Vec3::new(2.0, 2.0, 3.0));
        let rotated = rotate_axis(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_2,
        );
        close(rotated.x, 0.0);
        close(rotated.y, 1.0);
    }

    #[test]
    fn fit_recovers_rotation_and_translation() {
        let source = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let rotation = rotation_matrix(std::f64::consts::FRAC_PI_2, Vec3::new(0.0, 0.0, 1.0));
        let translation = Vec3::new(2.0, -1.0, 0.5);
        let target: Vec<Vec3> = source
            .iter()
            .map(|&point| rotation * point + translation)
            .collect();
        let result = fit_to_reference(&source, &target, None);
        close(result.rmsd, 0.0);
        for (&actual, &expected) in source.iter().zip(&target) {
            close(result.apply(actual).x, expected.x);
            close(result.apply(actual).y, expected.y);
            close(result.apply(actual).z, expected.z);
        }
    }

    #[test]
    fn wrapping_and_unwrapping_use_minimum_image() {
        let mut points = [
            Vec3::new(-0.2, 1.5, 10.0),
            Vec3::new(9.8, -0.5, 0.0),
            Vec3::new(10.1, 0.2, 0.0),
        ];
        wrap_positions(&mut points, Vec3::splat(10.0));
        assert_eq!(points[0], Vec3::new(9.8, 1.5, 0.0));
        assert_eq!(points[1], Vec3::new(9.8, 9.5, 0.0));
        let mut wrapped = [Vec3::new(9.8, 0.0, 0.0), Vec3::new(0.1, 0.0, 0.0)];
        unwrap_positions(&mut wrapped, Vec3::splat(10.0));
        close(wrapped[1].x, 10.1);
        close(
            minimum_image(Vec3::new(5.1, -5.1, 0.0), Vec3::splat(10.0)).x,
            -4.9,
        );
    }

    #[test]
    fn triclinic_wrap_round_trips_fractional_coordinates() {
        let box_vectors = Matrix3::new([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let mut points = [Vec3::new(2.4, 2.1, -0.5)];
        wrap_positions_triclinic(&mut points, box_vectors);
        let fractional = box_vectors.inverse().expect("box is invertible") * points[0];
        assert!((0.0..1.0).contains(&fractional.x));
        assert!((0.0..1.0).contains(&fractional.y));
        assert!((0.0..1.0).contains(&fractional.z));
    }
}
