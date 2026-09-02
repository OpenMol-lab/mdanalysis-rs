//! Numerical analysis helpers for molecular coordinate data.
//!
//! This module contains small, allocation-conscious wrappers around the
//! geometry and transformation primitives.  Coordinates are represented by
//! [`Vec3`] values; `*_array` variants are provided for callers that keep
//! coordinates in the conventional `[[f64; 3]]` form.

use crate::geometry::{self, Vec3};
use crate::transformations::{self, FitResult};

/// Fit `coordinates` onto `reference` with the Kabsch/Horn rigid transform.
///
/// The two arrays must contain the same, non-zero number of points.  `None`
/// denotes a shape mismatch or an empty input.  The returned transform maps a
/// point from `coordinates` to the corresponding point in `reference`.
#[must_use]
pub fn kabsch_fit(coordinates: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<FitResult> {
    if coordinates.is_empty() || coordinates.len() != reference.len() {
        return None;
    }
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    let reference: Vec<Vec3> = reference.iter().copied().map(Vec3::from).collect();
    Some(transformations::fit_to_reference(
        &coordinates,
        &reference,
        None,
    ))
}

/// Alias for [`kabsch_fit`] using a name common in analysis code.
#[must_use]
pub fn kabsch(coordinates: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<FitResult> {
    kabsch_fit(coordinates, reference)
}

/// RMSD between corresponding points without rotational fitting.
#[must_use]
pub fn rmsd_array(reference: &[[f64; 3]], coordinates: &[[f64; 3]]) -> Option<f64> {
    if reference.len() != coordinates.len() {
        return None;
    }
    let reference: Vec<Vec3> = reference.iter().copied().map(Vec3::from).collect();
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    Some(geometry::rmsd(&reference, &coordinates))
}

/// Alias for [`rmsd_array`].
#[must_use]
pub fn rmsd_arrays(reference: &[[f64; 3]], coordinates: &[[f64; 3]]) -> Option<f64> {
    rmsd_array(reference, coordinates)
}

/// RMSD after optimally superposing `coordinates` onto `reference`.
#[must_use]
pub fn kabsch_rmsd(reference: &[[f64; 3]], coordinates: &[[f64; 3]]) -> Option<f64> {
    kabsch_fit(coordinates, reference).map(|fit| fit.rmsd)
}

/// Build a symmetric, diagonal-free contact map using a distance cutoff.
///
/// Non-finite or negative cutoffs produce an all-false map.  A zero cutoff is
/// useful for detecting coincident points and is therefore accepted.
#[must_use]
pub fn contact_map(coordinates: &[Vec3], cutoff: f64) -> Vec<Vec<bool>> {
    let size = coordinates.len();
    let mut map = vec![vec![false; size]; size];
    if !cutoff.is_finite() || cutoff < 0.0 {
        return map;
    }
    let cutoff_squared = cutoff * cutoff;
    for i in 0..size {
        for j in (i + 1)..size {
            if (coordinates[i] - coordinates[j]).norm_squared() <= cutoff_squared {
                map[i][j] = true;
                map[j][i] = true;
            }
        }
    }
    map
}

/// Array-coordinate wrapper for [`contact_map`].
#[must_use]
pub fn contact_map_array(coordinates: &[[f64; 3]], cutoff: f64) -> Vec<Vec<bool>> {
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    contact_map(&coordinates, cutoff)
}

/// Fraction of contacts present in `coordinates` that are native contacts in
/// `reference`.
///
/// The denominator is the number of unique contacts in the reference map.  A
/// reference with no contacts has no defined fraction and returns `None`.
#[must_use]
pub fn contact_score(reference: &[Vec3], coordinates: &[Vec3], cutoff: f64) -> Option<f64> {
    if reference.len() != coordinates.len() || reference.len() < 2 {
        return None;
    }
    let native = contact_map(reference, cutoff);
    let current = contact_map(coordinates, cutoff);
    let mut native_count = 0usize;
    let mut matched_count = 0usize;
    for i in 0..reference.len() {
        for j in (i + 1)..reference.len() {
            if native[i][j] {
                native_count += 1;
                if current[i][j] {
                    matched_count += 1;
                }
            }
        }
    }
    (native_count > 0).then_some(matched_count as f64 / native_count as f64)
}

/// Array-coordinate wrapper for [`contact_score`].
#[must_use]
pub fn contact_score_array(
    reference: &[[f64; 3]],
    coordinates: &[[f64; 3]],
    cutoff: f64,
) -> Option<f64> {
    if reference.len() != coordinates.len() {
        return None;
    }
    let reference: Vec<Vec3> = reference.iter().copied().map(Vec3::from).collect();
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    contact_score(&reference, &coordinates, cutoff)
}

/// Alias for [`contact_score`].
#[must_use]
pub fn contact_fraction(reference: &[Vec3], coordinates: &[Vec3], cutoff: f64) -> Option<f64> {
    contact_score(reference, coordinates, cutoff)
}

/// Return all consecutive four-point torsion angles in radians.
#[must_use]
pub fn dihedral_series(coordinates: &[Vec3]) -> Vec<f64> {
    coordinates
        .windows(4)
        .map(|window| geometry::dihedral_points(window[0], window[1], window[2], window[3]))
        .collect()
}

/// Array-coordinate wrapper for [`dihedral_series`].
#[must_use]
pub fn dihedral_series_array(coordinates: &[[f64; 3]]) -> Vec<f64> {
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    dihedral_series(&coordinates)
}

/// A best-fit axis through a set of helix/line coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HelixAxis {
    /// Centroid of the input points, lying on the fitted axis.
    pub origin: Vec3,
    /// Unit direction of the fitted axis.  Its sign follows first-to-last
    /// point order where that direction is defined.
    pub direction: Vec3,
    /// RMS distance of the points from the axis.
    pub rmsd: f64,
    /// Extent of the points along the axis (max projection minus min).
    pub length: f64,
}

/// Calculate the principal (least-squares) axis of a helix-like coordinate
/// set.  The axis is the first principal component of the centered points.
#[must_use]
pub fn helix_best_fit_axis(coordinates: &[Vec3]) -> Option<HelixAxis> {
    let origin = coordinates
        .iter()
        .copied()
        .reduce(|sum, point| sum + point)
        .map(|sum| sum / coordinates.len() as f64)?;
    if coordinates.len() == 1 {
        return Some(HelixAxis {
            origin,
            direction: Vec3::ZERO,
            rmsd: 0.0,
            length: 0.0,
        });
    }

    // Covariance is symmetric, so power iteration converges to the dominant
    // eigenvector without requiring a general eigensolver dependency.
    let mut covariance = [[0.0; 3]; 3];
    for &point in coordinates {
        let centered = point - origin;
        let values = centered.to_array();
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] += values[row] * values[column];
            }
        }
    }
    let mut direction = (coordinates[coordinates.len() - 1] - coordinates[0]).normalized();
    if direction == Vec3::ZERO {
        direction = Vec3::new(1.0, 0.0, 0.0);
    }
    for _ in 0..64 {
        let next = Vec3::new(
            covariance[0][0] * direction.x
                + covariance[0][1] * direction.y
                + covariance[0][2] * direction.z,
            covariance[1][0] * direction.x
                + covariance[1][1] * direction.y
                + covariance[1][2] * direction.z,
            covariance[2][0] * direction.x
                + covariance[2][1] * direction.y
                + covariance[2][2] * direction.z,
        )
        .normalized();
        if next == Vec3::ZERO {
            break;
        }
        if (next - direction).norm() < 1.0e-14 || (next + direction).norm() < 1.0e-14 {
            direction = next;
            break;
        }
        direction = next;
    }
    // Keep a stable orientation for a helix traversed in either direction.
    if direction.dot(coordinates[coordinates.len() - 1] - coordinates[0]) < 0.0 {
        direction = -direction;
    }
    let mut min_projection = f64::INFINITY;
    let mut max_projection = f64::NEG_INFINITY;
    let mut squared_error = 0.0;
    for &point in coordinates {
        let centered = point - origin;
        let projection = centered.dot(direction);
        min_projection = min_projection.min(projection);
        max_projection = max_projection.max(projection);
        squared_error += (centered - direction * projection).norm_squared();
    }
    Some(HelixAxis {
        origin,
        direction,
        rmsd: (squared_error / coordinates.len() as f64).sqrt(),
        length: max_projection - min_projection,
    })
}

/// Array-coordinate wrapper for [`helix_best_fit_axis`].
#[must_use]
pub fn helix_best_fit_axis_array(coordinates: &[[f64; 3]]) -> Option<HelixAxis> {
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    helix_best_fit_axis(&coordinates)
}

/// Return the unit vector of best fit, oriented toward the first point.
///
/// This is the vector-only form used by MDAnalysis' helix analysis helper.
/// For a single point (or coincident points) the direction is zero.
#[must_use]
pub fn vector_of_best_fit(coordinates: &[Vec3]) -> Option<Vec3> {
    let axis = helix_best_fit_axis(coordinates)?;
    if coordinates.len() > 1 && axis.direction.dot(coordinates[0] - axis.origin) < 0.0 {
        Some(-axis.direction)
    } else {
        Some(axis.direction)
    }
}

/// Array-coordinate wrapper for [`vector_of_best_fit`].
#[must_use]
pub fn vector_of_best_fit_array(coordinates: &[[f64; 3]]) -> Option<Vec3> {
    let coordinates: Vec<Vec3> = coordinates.iter().copied().map(Vec3::from).collect();
    vector_of_best_fit(&coordinates)
}

/// Return the centroid and direction of a best-fit helix axis.
#[must_use]
pub fn best_fit_axis(coordinates: &[Vec3]) -> Option<(Vec3, Vec3)> {
    helix_best_fit_axis(coordinates).map(|axis| (axis.origin, axis.direction))
}

/// Ordinary least-squares fit `y = intercept + slope * x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRegression {
    /// Fitted slope.
    pub slope: f64,
    /// Fitted y-intercept.
    pub intercept: f64,
    /// Coefficient of determination, when the response has non-zero variance.
    pub r_squared: f64,
}

/// Perform an ordinary least-squares linear regression.
#[must_use]
pub fn linear_regression(x: &[f64], y: &[f64]) -> Option<LinearRegression> {
    if x.len() != y.len() || x.len() < 2 || x.iter().chain(y).any(|value| !value.is_finite()) {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let denominator = x.iter().map(|value| (value - mean_x).powi(2)).sum::<f64>();
    if denominator <= f64::EPSILON {
        return None;
    }
    let covariance = x
        .iter()
        .zip(y)
        .map(|(x_value, y_value)| (x_value - mean_x) * (y_value - mean_y))
        .sum::<f64>();
    let slope = covariance / denominator;
    let intercept = mean_y - slope * mean_x;
    let total = y.iter().map(|value| (value - mean_y).powi(2)).sum::<f64>();
    let residual = y
        .iter()
        .zip(x)
        .map(|(actual, x_value)| (actual - (intercept + slope * x_value)).powi(2))
        .sum::<f64>();
    let r_squared = if total <= f64::EPSILON {
        1.0
    } else {
        (1.0 - residual / total).clamp(0.0, 1.0)
    };
    Some(LinearRegression {
        slope,
        intercept,
        r_squared,
    })
}

/// Parameters of an exponential fit `value = amplitude * exp(-rate * time)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialDecay {
    /// Value at time zero (`exp(intercept)` in log space).
    pub amplitude: f64,
    /// Decay rate.  A negative rate describes exponential growth.
    pub rate: f64,
    /// Coefficient of determination in log space.
    pub r_squared: f64,
}

/// Fit an exponential decay by linear regression in log space.
#[must_use]
pub fn fit_exponential_decay(time: &[f64], values: &[f64]) -> Option<ExponentialDecay> {
    if values.iter().any(|value| *value <= 0.0) {
        return None;
    }
    let logarithms: Vec<f64> = values.iter().map(|value| value.ln()).collect();
    let regression = linear_regression(time, &logarithms)?;
    Some(ExponentialDecay {
        amplitude: regression.intercept.exp(),
        rate: -regression.slope,
        r_squared: regression.r_squared,
    })
}

/// Alias for [`fit_exponential_decay`].
#[must_use]
pub fn exponential_decay_fit(time: &[f64], values: &[f64]) -> Option<ExponentialDecay> {
    fit_exponential_decay(time, values)
}

/// Lag-averaged mean-square displacement of a trajectory.
///
/// The trajectory is indexed as `trajectory[time][particle]`.  The returned
/// vector contains lags `0..trajectory.len()`, including the exact zero lag.
/// Every frame must contain the same non-zero number of particles.
#[must_use]
pub fn mean_square_displacement_series(trajectory: &[Vec<Vec3>]) -> Option<Vec<f64>> {
    let first = trajectory.first()?;
    if first.is_empty() || trajectory.iter().any(|frame| frame.len() != first.len()) {
        return None;
    }
    let frame_count = trajectory.len();
    let mut result = Vec::with_capacity(frame_count);
    for lag in 0..frame_count {
        let mut sum = 0.0;
        let mut samples = 0usize;
        for start in 0..(frame_count - lag) {
            for (&later, &earlier) in trajectory[start + lag].iter().zip(&trajectory[start]) {
                sum += (later - earlier).norm_squared();
                samples += 1;
            }
        }
        result.push(sum / samples as f64);
    }
    Some(result)
}

/// Array-coordinate wrapper for [`mean_square_displacement_series`].
#[must_use]
pub fn mean_square_displacement_series_array(trajectory: &[Vec<[f64; 3]>]) -> Option<Vec<f64>> {
    let trajectory: Vec<Vec<Vec3>> = trajectory
        .iter()
        .map(|frame| frame.iter().copied().map(Vec3::from).collect())
        .collect();
    mean_square_displacement_series(&trajectory)
}

/// Alias for [`mean_square_displacement_series`].
#[must_use]
pub fn msd_time_series(trajectory: &[Vec<Vec3>]) -> Option<Vec<f64>> {
    mean_square_displacement_series(trajectory)
}

/// Estimate a diffusion coefficient from an MSD-vs-time curve.
///
/// For `dimensions` spatial dimensions, Einstein's relation is
/// `MSD = 2 * dimensions * D * time`; the slope is obtained by ordinary least
/// squares and the fitted intercept is allowed to absorb localization noise.
#[must_use]
pub fn diffusion_coefficient(time: &[f64], msd: &[f64], dimensions: usize) -> Option<f64> {
    if !(1..=3).contains(&dimensions) || msd.iter().any(|value| *value < 0.0) {
        return None;
    }
    let regression = linear_regression(time, msd)?;
    Some(regression.slope / (2.0 * dimensions as f64))
}

/// Alias for [`diffusion_coefficient`].
#[must_use]
pub fn estimate_diffusion_coefficient(time: &[f64], msd: &[f64], dimensions: usize) -> Option<f64> {
    diffusion_coefficient(time, msd, dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-10, "{left} != {right}");
    }

    #[test]
    fn kabsch_and_array_rmsd_wrappers() {
        let reference = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let coordinates = [[2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0.0]];
        let fit = kabsch_fit(&coordinates, &reference).expect("matching coordinates");
        close(fit.rmsd, 0.0);
        close(kabsch_rmsd(&reference, &coordinates).unwrap(), 0.0);
        close(
            rmsd_array(&reference, &coordinates).unwrap(),
            (11.0_f64 / 3.0).sqrt(),
        );
        assert!(kabsch_fit(&[], &[]).is_none());
    }

    #[test]
    fn contact_map_and_score_are_symmetric() {
        let reference = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let moved = [
            Vec3::ZERO,
            Vec3::new(1.1, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let map = contact_map(&reference, 1.2);
        assert_eq!(map[0][1], map[1][0]);
        assert!(!map[0][0]);
        close(contact_score(&reference, &moved, 1.2).unwrap(), 1.0);
    }

    #[test]
    fn dihedral_and_helix_axis_have_expected_geometry() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        close(
            dihedral_series(&points)[0].abs(),
            std::f64::consts::FRAC_PI_2,
        );
        let line = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let axis = helix_best_fit_axis(&line).unwrap();
        close(axis.direction.x, 1.0);
        close(axis.rmsd, 0.0);
        close(axis.length, 3.0);
        close(vector_of_best_fit(&line).unwrap().x, -1.0);
    }

    #[test]
    fn regression_and_exponential_decay_recover_parameters() {
        let time = [0.0, 1.0, 2.0, 3.0];
        let values = [4.0, 2.0, 1.0, 0.5];
        let fit = fit_exponential_decay(&time, &values).unwrap();
        close(fit.amplitude, 4.0);
        close(fit.rate, 2.0_f64.ln());
        let line = linear_regression(&time, &[1.0, 3.0, 5.0, 7.0]).unwrap();
        close(line.slope, 2.0);
        close(line.intercept, 1.0);
        close(line.r_squared, 1.0);
    }

    #[test]
    fn msd_series_and_diffusion_follow_einstein_relation() {
        let trajectory = vec![
            vec![Vec3::ZERO],
            vec![Vec3::new(1.0, 0.0, 0.0)],
            vec![Vec3::new(2.0, 0.0, 0.0)],
        ];
        let msd = mean_square_displacement_series(&trajectory).unwrap();
        assert_eq!(msd, vec![0.0, 1.0, 4.0]);
        close(
            diffusion_coefficient(&[0.0, 1.0, 2.0], &[0.0, 2.0, 4.0], 1).unwrap(),
            1.0,
        );
        assert!(mean_square_displacement_series(&[Vec::new()]).is_none());
    }
}
