//! Coordinate transformations used by molecular dynamics workflows.
//!
//! The routines in this module operate on [`crate::geometry::Vec3`] values and
//! deliberately do not depend on an atom or trajectory container.  This makes
//! them useful both for one frame and for callers that keep coordinates in a
//! custom data structure.

use crate::geometry::{Matrix3, Vec3, center_of_mass};

/// Errors returned by frame-level coordinate transformations.
#[derive(Clone, Debug, PartialEq)]
pub enum TransformationError {
    /// The requested atom index is not present in the frame.
    AtomIndexOutOfBounds { index: usize, count: usize },
    /// A coordinate-dependent operation was requested without a unit cell.
    MissingDimensions,
    /// The supplied dimensions are not a valid six-component unit cell.
    InvalidDimensions([f64; 6]),
    /// Per-atom metadata does not have the same length as the frame.
    LengthMismatch { expected: usize, found: usize },
    /// A point, direction, or weight vector was invalid.
    InvalidVector(&'static str),
}

impl std::fmt::Display for TransformationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AtomIndexOutOfBounds { index, count } => {
                write!(
                    formatter,
                    "atom index {index} is out of bounds for {count} coordinates"
                )
            }
            Self::MissingDimensions => {
                formatter.write_str("transformation requires unit-cell dimensions")
            }
            Self::InvalidDimensions(dimensions) => {
                write!(formatter, "invalid unit-cell dimensions {dimensions:?}")
            }
            Self::LengthMismatch { expected, found } => {
                write!(
                    formatter,
                    "transformation expected {expected} values, found {found}"
                )
            }
            Self::InvalidVector(name) => write!(formatter, "{name} must be finite and non-zero"),
        }
    }
}

impl std::error::Error for TransformationError {}

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
    // `Matrix3` multiplies column vectors, while the lattice vectors are
    // stored as rows (the convention used by `mdamath::triclinic_vectors`).
    let lattice = box_vectors.transpose();
    let Some(inverse) = lattice.inverse() else {
        return;
    };
    for coordinate in coordinates {
        let fractional = inverse * *coordinate;
        let wrapped = Vec3::new(
            fractional.x.rem_euclid(1.0),
            fractional.y.rem_euclid(1.0),
            fractional.z.rem_euclid(1.0),
        );
        *coordinate = lattice * wrapped;
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

/// Set one frame's unit-cell dimensions in `[a, b, c, alpha, beta, gamma]`
/// form.  The values are copied into the frame and are not normalized.
pub fn set_dimensions(
    frame: &mut crate::core::Frame,
    dimensions: [f64; 6],
) -> Result<(), TransformationError> {
    validate_dimensions(dimensions)?;
    frame.dimensions = Some(dimensions);
    Ok(())
}

/// Set dimensions for every frame.  A single dimensions vector is reused for
/// all frames; otherwise one vector must be supplied per frame.
pub fn set_dimensions_for_frames(
    frames: &mut [crate::core::Frame],
    dimensions: &[[f64; 6]],
) -> Result<(), TransformationError> {
    if dimensions.len() != 1 && dimensions.len() != frames.len() {
        return Err(TransformationError::LengthMismatch {
            expected: frames.len(),
            found: dimensions.len(),
        });
    }
    for (index, frame) in frames.iter_mut().enumerate() {
        let dimensions = dimensions[if dimensions.len() == 1 { 0 } else { index }];
        set_dimensions(frame, dimensions)?;
    }
    Ok(())
}

/// Translate all positions in a frame.  Velocities and forces are unchanged.
pub fn translate_frame(frame: &mut crate::core::Frame, offset: Vec3) {
    for position in &mut frame.positions {
        *position = (Vec3::from(*position) + offset).to_array();
    }
}

/// Rotate positions, velocities, and forces in a frame around an arbitrary
/// axis through `origin`.  `angle` is in radians.
pub fn rotate_frame(
    frame: &mut crate::core::Frame,
    angle: f64,
    direction: Vec3,
    origin: Vec3,
) -> Result<(), TransformationError> {
    if !direction.x.is_finite()
        || !direction.y.is_finite()
        || !direction.z.is_finite()
        || direction.norm_squared() <= f64::EPSILON
    {
        return Err(TransformationError::InvalidVector("direction"));
    }
    if !origin.x.is_finite() || !origin.y.is_finite() || !origin.z.is_finite() || !angle.is_finite()
    {
        return Err(TransformationError::InvalidVector("rotation origin/angle"));
    }
    let rotation = rotation_matrix(angle, direction);
    for position in &mut frame.positions {
        *position = (origin + rotation * (Vec3::from(*position) - origin)).to_array();
    }
    if let Some(velocities) = &mut frame.velocities {
        for velocity in velocities {
            *velocity = (rotation * Vec3::from(*velocity)).to_array();
        }
    }
    if let Some(forces) = &mut frame.forces {
        for force in forces {
            *force = (rotation * Vec3::from(*force)).to_array();
        }
    }
    Ok(())
}

/// Centre selected frame coordinates at a point or at the centre of its unit
/// cell.  `masses`, when supplied, must match `atom_indices` and selects a
/// mass-weighted centre; `None` uses the geometric centre.  When `wrap` is
/// true, selected positions are wrapped into the primary triclinic cell before
/// calculating the centre, without changing unselected positions.
pub fn center_in_box(
    frame: &mut crate::core::Frame,
    atom_indices: &[usize],
    masses: Option<&[f64]>,
    point: Option<Vec3>,
    wrap: bool,
) -> Result<(), TransformationError> {
    if atom_indices.is_empty() {
        return Err(TransformationError::InvalidVector("atom_indices"));
    }
    if let Some(masses) = masses
        && masses.len() != atom_indices.len()
    {
        return Err(TransformationError::LengthMismatch {
            expected: atom_indices.len(),
            found: masses.len(),
        });
    }
    let dimensions = frame.dimensions;
    if let Some(dimensions) = dimensions {
        validate_dimensions(dimensions)?;
    }
    let lattice = dimensions.map(|dimensions| {
        let vectors = crate::mdamath::triclinic_vectors(dimensions);
        Matrix3::from_cols([
            Vec3::from(vectors[0]),
            Vec3::from(vectors[1]),
            Vec3::from(vectors[2]),
        ])
    });
    let mut selected = Vec::with_capacity(atom_indices.len());
    for &index in atom_indices {
        let position =
            *frame
                .positions
                .get(index)
                .ok_or(TransformationError::AtomIndexOutOfBounds {
                    index,
                    count: frame.positions.len(),
                })?;
        let position = if wrap {
            let lattice = lattice.ok_or(TransformationError::MissingDimensions)?;
            let inverse = lattice
                .inverse()
                .ok_or(TransformationError::InvalidDimensions(
                    dimensions.unwrap_or_default(),
                ))?;
            let fractional = inverse * Vec3::from(position);
            (lattice
                * Vec3::new(
                    fractional.x.rem_euclid(1.0),
                    fractional.y.rem_euclid(1.0),
                    fractional.z.rem_euclid(1.0),
                ))
            .to_array()
        } else {
            position
        };
        selected.push(Vec3::from(position));
    }
    let center = if let Some(masses) = masses {
        if masses.iter().any(|mass| !mass.is_finite() || *mass < 0.0) {
            return Err(TransformationError::InvalidVector("masses"));
        }
        let total: f64 = masses.iter().sum();
        if total <= f64::EPSILON {
            return Err(TransformationError::InvalidVector("masses"));
        }
        selected
            .iter()
            .zip(masses)
            .fold(Vec3::ZERO, |sum, (position, mass)| sum + *position * *mass)
            / total
    } else {
        selected
            .iter()
            .copied()
            .fold(Vec3::ZERO, |sum, position| sum + position)
            / selected.len() as f64
    };
    let target = if let Some(point) = point {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(TransformationError::InvalidVector("point"));
        }
        point
    } else {
        let lattice = lattice.ok_or(TransformationError::MissingDimensions)?;
        (lattice * Vec3::splat(0.5)).to_array().into()
    };
    translate_frame(frame, target - center);
    Ok(())
}

/// Unwrap a selected sequence using triclinic minimum-image displacements.
/// The first selected position is retained and each following position is
/// shifted to the nearest image relative to its predecessor.
pub fn unwrap_positions_triclinic(
    coordinates: &mut [Vec3],
    dimensions: [f64; 6],
) -> Result<(), TransformationError> {
    validate_dimensions(dimensions)?;
    if coordinates.len() < 2 {
        return Ok(());
    }
    for index in 1..coordinates.len() {
        let displacement = coordinates[index] - coordinates[index - 1];
        let image = crate::distances::minimum_image_triclinic(displacement.to_array(), dimensions)
            .map_err(|_| TransformationError::InvalidDimensions(dimensions))?;
        coordinates[index] = coordinates[index - 1] + Vec3::from(image);
    }
    Ok(())
}

fn validate_dimensions(dimensions: [f64; 6]) -> Result<(), TransformationError> {
    crate::distances::transform_s_to_r(&[[0.0; 3]], dimensions)
        .map(|_| ())
        .map_err(|_| TransformationError::InvalidDimensions(dimensions))
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
#[allow(clippy::needless_range_loop)]
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
        let fractional = box_vectors
            .transpose()
            .inverse()
            .expect("box is invertible")
            * points[0];
        assert!((0.0..1.0).contains(&fractional.x));
        assert!((0.0..1.0).contains(&fractional.y));
        assert!((0.0..1.0).contains(&fractional.z));
    }

    #[test]
    fn frame_transformations_update_coordinates_and_vectors() {
        let mut frame = crate::core::Frame::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        frame.velocities = Some(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        frame.forces = Some(vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
        translate_frame(&mut frame, Vec3::new(1.0, 2.0, 3.0));
        rotate_frame(
            &mut frame,
            std::f64::consts::FRAC_PI_2,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        close(frame.positions[0][0], -2.0);
        close(frame.positions[0][1], 1.0);
        close(frame.velocities.as_ref().unwrap()[0][0], 0.0);
        close(frame.velocities.as_ref().unwrap()[0][1], 1.0);
        close(frame.forces.as_ref().unwrap()[0][2], 1.0);
    }

    #[test]
    fn dimensions_and_center_transformations_validate_and_apply() {
        let mut frame = crate::core::Frame::new(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        set_dimensions(&mut frame, [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]).unwrap();
        center_in_box(&mut frame, &[0, 1], None, None, false).unwrap();
        close(frame.positions[0][0], 4.0);
        close(frame.positions[1][0], 6.0);
        assert!(matches!(
            rotate_frame(&mut frame, 1.0, Vec3::ZERO, Vec3::ZERO),
            Err(TransformationError::InvalidVector("direction"))
        ));
        assert!(matches!(
            set_dimensions(&mut frame, [0.0; 6]),
            Err(TransformationError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn triclinic_unwrap_uses_minimum_image() {
        let dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 60.0];
        let mut points = [Vec3::new(9.8, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0)];
        unwrap_positions_triclinic(&mut points, dimensions).unwrap();
        close(points[1].x, 10.2);
    }
}
