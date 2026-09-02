//! Small, dependency-free geometry helpers for molecular simulations.
//!
//! Coordinates are represented by [`Vec3`] values and matrices by [`Matrix3`].
//! The functions in this module intentionally operate on slices so they can be
//! used with any atom container without requiring an allocation-heavy wrapper.

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

/// A three-dimensional vector with `f64` components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Construct a vector from its three components.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Construct a vector with all components equal to `value`.
    #[must_use]
    pub const fn splat(value: f64) -> Self {
        Self::new(value, value, value)
    }

    /// Return the components as an array.
    #[must_use]
    pub const fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Dot product with another vector.
    #[must_use]
    pub const fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product with another vector.
    #[must_use]
    pub const fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean norm.
    #[must_use]
    pub const fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean norm.
    #[must_use]
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Return a unit vector, or [`Vec3::ZERO`] for a zero-length vector.
    #[must_use]
    pub fn normalized(self) -> Self {
        let length = self.norm();
        if length == 0.0 {
            Self::ZERO
        } else {
            self / length
        }
    }

    /// Euclidean distance to another vector.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).norm()
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(value: [f64; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<(f64, f64, f64)> for Vec3 {
    fn from(value: (f64, f64, f64)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<Vec3> for [f64; 3] {
    fn from(value: Vec3) -> Self {
        value.to_array()
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {index} is out of bounds"),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index {index} is out of bounds"),
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

/// A 3x3 matrix in row-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix3 {
    /// Matrix entries, indexed as `[row][column]`.
    pub m: [[f64; 3]; 3],
}

impl Matrix3 {
    /// Construct a matrix from rows.
    #[must_use]
    pub const fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// The identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// The zero matrix.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new([[0.0; 3]; 3])
    }

    /// Construct a matrix from three row vectors.
    #[must_use]
    pub const fn from_rows(rows: [Vec3; 3]) -> Self {
        Self::new([rows[0].to_array(), rows[1].to_array(), rows[2].to_array()])
    }

    /// Construct a matrix from three column vectors.
    #[must_use]
    pub const fn from_cols(cols: [Vec3; 3]) -> Self {
        Self::new([
            [cols[0].x, cols[1].x, cols[2].x],
            [cols[0].y, cols[1].y, cols[2].y],
            [cols[0].z, cols[1].z, cols[2].z],
        ])
    }

    /// Matrix transpose.
    #[must_use]
    pub const fn transpose(self) -> Self {
        Self::new([
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        ])
    }

    /// Matrix determinant.
    #[must_use]
    pub const fn determinant(self) -> f64 {
        let a = self.m;
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    }

    /// Matrix inverse, if the matrix is nonsingular.
    #[must_use]
    pub fn inverse(self) -> Option<Self> {
        let a = self.m;
        let det = self.determinant();
        if det.abs() <= f64::EPSILON {
            return None;
        }
        let cofactors = [
            [
                a[1][1] * a[2][2] - a[1][2] * a[2][1],
                a[1][2] * a[2][0] - a[1][0] * a[2][2],
                a[1][0] * a[2][1] - a[1][1] * a[2][0],
            ],
            [
                a[0][2] * a[2][1] - a[0][1] * a[2][2],
                a[0][0] * a[2][2] - a[0][2] * a[2][0],
                a[0][1] * a[2][0] - a[0][0] * a[2][1],
            ],
            [
                a[0][1] * a[1][2] - a[0][2] * a[1][1],
                a[0][2] * a[1][0] - a[0][0] * a[1][2],
                a[0][0] * a[1][1] - a[0][1] * a[1][0],
            ],
        ];
        Some(Self::new(cofactors).transpose() * (1.0 / det))
    }

    /// Multiply this matrix by a vector.
    #[must_use]
    pub const fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }
}

impl Default for Matrix3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<[[f64; 3]; 3]> for Matrix3 {
    fn from(value: [[f64; 3]; 3]) -> Self {
        Self::new(value)
    }
}

impl Index<(usize, usize)> for Matrix3 {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.m[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix3 {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.m[index.0][index.1]
    }
}

impl Add for Matrix3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..3 {
            for j in 0..3 {
                result.m[i][j] += rhs.m[i][j];
            }
        }
        result
    }
}

impl Sub for Matrix3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        for i in 0..3 {
            for j in 0..3 {
                result.m[i][j] -= rhs.m[i][j];
            }
        }
        result
    }
}

impl Mul<f64> for Matrix3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = self;
        for row in &mut result.m {
            for value in row {
                *value *= rhs;
            }
        }
        result
    }
}

impl Mul<Matrix3> for f64 {
    type Output = Matrix3;

    fn mul(self, rhs: Matrix3) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for Matrix3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Mul<Vec3> for Matrix3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        self.mul_vec(rhs)
    }
}

impl Mul for Matrix3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result.m[i][j] += self.m[i][k] * rhs.m[k][j];
                }
            }
        }
        result
    }
}

/// Euclidean distance between two points.
#[must_use]
pub fn distance(a: Vec3, b: Vec3) -> f64 {
    a.distance(b)
}

/// Squared Euclidean distance between two points.
#[must_use]
pub fn distance_squared(a: Vec3, b: Vec3) -> f64 {
    (a - b).norm_squared()
}

/// Pairwise distances between two coordinate collections.
///
/// The result has `reference.len()` rows and `configuration.len()` columns.
#[must_use]
pub fn distance_array(reference: &[Vec3], configuration: &[Vec3]) -> Vec<Vec<f64>> {
    reference
        .iter()
        .map(|a| configuration.iter().map(|b| a.distance(*b)).collect())
        .collect()
}

/// Pairwise distances within one collection in condensed upper-triangle form.
///
/// Values are ordered as `(0, 1), (0, 2), ..., (1, 2), ...`, matching the
/// ordering returned by MDAnalysis' `self_distance_array`.
#[must_use]
pub fn self_distance_array(coordinates: &[Vec3]) -> Vec<f64> {
    let size = coordinates.len();
    let mut result = Vec::with_capacity(size.saturating_mul(size.saturating_sub(1)) / 2);
    for i in 0..size {
        for j in (i + 1)..size {
            result.push(coordinates[i].distance(coordinates[j]));
        }
    }
    result
}

/// Pairwise distances within one collection as a square matrix.
#[must_use]
pub fn self_distance_matrix(coordinates: &[Vec3]) -> Vec<Vec<f64>> {
    let size = coordinates.len();
    let mut result = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in (i + 1)..size {
            let value = coordinates[i].distance(coordinates[j]);
            result[i][j] = value;
            result[j][i] = value;
        }
    }
    result
}

/// Center of geometry (the unweighted mean position).
#[must_use]
pub fn center_of_geometry(coordinates: &[Vec3]) -> Vec3 {
    if coordinates.is_empty() {
        return Vec3::ZERO;
    }
    coordinates.iter().copied().fold(Vec3::ZERO, Add::add) / coordinates.len() as f64
}

/// Mass-weighted center of mass.
///
/// A zero vector is returned for empty input or a zero total mass. The two
/// slices must have equal lengths.
#[must_use]
pub fn center_of_mass(coordinates: &[Vec3], masses: &[f64]) -> Vec3 {
    assert_eq!(
        coordinates.len(),
        masses.len(),
        "coordinates and masses must have equal lengths"
    );
    let mut weighted_sum = Vec3::ZERO;
    let mut total_mass = 0.0;
    for (&coordinate, &mass) in coordinates.iter().zip(masses) {
        weighted_sum += coordinate * mass;
        total_mass += mass;
    }
    if total_mass == 0.0 {
        Vec3::ZERO
    } else {
        weighted_sum / total_mass
    }
}

/// Fallible variant of [`center_of_mass`] for callers that need to distinguish
/// an empty or zero-mass collection from a real origin-valued center.
#[must_use]
pub fn try_center_of_mass(coordinates: &[Vec3], masses: &[f64]) -> Option<Vec3> {
    assert_eq!(
        coordinates.len(),
        masses.len(),
        "coordinates and masses must have equal lengths"
    );
    let total_mass: f64 = masses.iter().sum();
    if total_mass == 0.0 {
        None
    } else {
        Some(
            coordinates
                .iter()
                .zip(masses)
                .fold(Vec3::ZERO, |sum, (&position, &mass)| sum + position * mass)
                / total_mass,
        )
    }
}

/// Root-mean-square distance between corresponding points.
#[must_use]
pub fn rmsd(reference: &[Vec3], coordinates: &[Vec3]) -> f64 {
    assert_eq!(
        reference.len(),
        coordinates.len(),
        "reference and coordinates must have equal lengths"
    );
    if reference.is_empty() {
        return 0.0;
    }
    let sum: f64 = reference
        .iter()
        .zip(coordinates)
        .map(|(&a, &b)| distance_squared(a, b))
        .sum();
    (sum / reference.len() as f64).sqrt()
}

/// Mass-weighted root-mean-square distance between corresponding points.
#[must_use]
pub fn weighted_rmsd(reference: &[Vec3], coordinates: &[Vec3], weights: &[f64]) -> f64 {
    assert_eq!(
        reference.len(),
        coordinates.len(),
        "reference and coordinates must have equal lengths"
    );
    assert_eq!(
        reference.len(),
        weights.len(),
        "coordinates and weights must have equal lengths"
    );
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }
    let sum: f64 = reference
        .iter()
        .zip(coordinates)
        .zip(weights)
        .map(|((&a, &b), &weight)| weight * distance_squared(a, b))
        .sum();
    (sum / total_weight).sqrt()
}

/// Radius of gyration around the mass-weighted center of mass.
#[must_use]
pub fn radius_of_gyration(coordinates: &[Vec3], masses: &[f64]) -> f64 {
    assert_eq!(
        coordinates.len(),
        masses.len(),
        "coordinates and masses must have equal lengths"
    );
    let total_mass: f64 = masses.iter().sum();
    if total_mass == 0.0 {
        return 0.0;
    }
    let center = center_of_mass(coordinates, masses);
    let moment: f64 = coordinates
        .iter()
        .zip(masses)
        .map(|(&position, &mass)| mass * distance_squared(position, center))
        .sum();
    (moment / total_mass).sqrt()
}

/// Euclidean norm of a vector (equivalent to `v.norm()`).
#[must_use]
pub fn norm(v: Vec3) -> f64 {
    v.norm()
}

/// Unit normal to two vectors, or zero for collinear vectors.
#[must_use]
pub fn normal(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(b).normalized()
}

/// Angle between two vectors in radians.
///
/// The result is in `[0, pi]`; a zero vector produces `0.0`.
#[must_use]
pub fn angle(a: Vec3, b: Vec3) -> f64 {
    let denominator = a.norm() * b.norm();
    if denominator == 0.0 {
        return 0.0;
    }
    (a.dot(b) / denominator).clamp(-1.0, 1.0).acos()
}

/// Dihedral (torsion) angle for three consecutive bond vectors in radians.
///
/// This follows the convention used by `MDAnalysis.lib.mdamath.dihedral`: the
/// returned angle lies in `[-pi, pi]`.
#[must_use]
pub fn dihedral(ab: Vec3, bc: Vec3, cd: Vec3) -> f64 {
    let n1 = ab.cross(bc);
    let n2 = bc.cross(cd);
    let magnitude = angle(n1, n2);
    if ab.dot(bc.cross(cd)) <= 0.0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Dihedral (torsion) angle for four points in radians.
#[must_use]
pub fn dihedral_points(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    dihedral(b - a, c - b, d - c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-12, "{left} != {right}");
    }

    #[test]
    fn vector_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-2.0, 1.0, 4.0);
        assert_eq!(a + b, Vec3::new(-1.0, 3.0, 7.0));
        assert_eq!(a.cross(b), Vec3::new(5.0, -10.0, 5.0));
        close(a.dot(b), 12.0);
        close(a.norm(), 14.0_f64.sqrt());
        close(a.distance(b), 11.0_f64.sqrt());
    }

    #[test]
    fn matrix_operations_and_inverse() {
        let matrix = Matrix3::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        assert_eq!(
            matrix * Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(14.0, 14.0, 17.0)
        );
        close(matrix.determinant(), 1.0);
        let product = matrix * matrix.inverse().expect("matrix is invertible");
        for i in 0..3 {
            for j in 0..3 {
                close(product[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn distance_arrays_have_expected_shapes_and_order() {
        let points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let matrix = distance_array(&points[..2], &points[1..]);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        close(matrix[0][0], 1.0);
        close(matrix[0][1], 2.0);
        close(matrix[1][0], 0.0);
        close(matrix[1][1], 5.0_f64.sqrt());
        let condensed = self_distance_array(&points);
        assert_eq!(condensed.len(), 3);
        assert_eq!(condensed, vec![1.0, 2.0, 5.0_f64.sqrt()]);
    }

    #[test]
    fn centers_and_radii_are_mass_weighted() {
        let points = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let masses = [1.0, 3.0];
        assert_eq!(center_of_geometry(&points), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(center_of_mass(&points, &masses), Vec3::new(1.5, 0.0, 0.0));
        close(radius_of_gyration(&points, &masses), 0.8660254037844386);
        close(
            rmsd(
                &points,
                &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)],
            ),
            1.0,
        );
        close(
            weighted_rmsd(
                &points,
                &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)],
                &masses,
            ),
            1.0,
        );
    }

    #[test]
    fn angular_helpers() {
        close(
            angle(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            std::f64::consts::FRAC_PI_2,
        );
        assert_eq!(
            normal(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 0.0, 1.0)
        );
        close(
            dihedral_points(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::ZERO,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
            ),
            std::f64::consts::FRAC_PI_2,
        );
    }
}
