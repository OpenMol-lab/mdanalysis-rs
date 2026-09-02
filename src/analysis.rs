//! Reusable analysis routines operating on slices of coordinates and frames.

use crate::core::{AtomGroup, Frame, Trajectory};

/// Compute a symmetric boolean contact matrix using a Euclidean cutoff.
pub fn contact_matrix(coordinates: &[[f64; 3]], cutoff: f64) -> Vec<Vec<bool>> {
    let mut matrix = vec![vec![false; coordinates.len()]; coordinates.len()];
    let cutoff2 = cutoff * cutoff;
    for i in 0..coordinates.len() {
        for j in (i + 1)..coordinates.len() {
            let d2 = squared_distance(coordinates[i], coordinates[j]);
            if d2 <= cutoff2 {
                matrix[i][j] = true;
                matrix[j][i] = true;
            }
        }
    }
    matrix
}

/// Calculate the mean square displacement between two coordinate sets.
pub fn mean_square_displacement(reference: &[[f64; 3]], coordinates: &[[f64; 3]]) -> Option<f64> {
    if reference.len() != coordinates.len() || reference.is_empty() {
        return None;
    }
    Some(
        reference
            .iter()
            .zip(coordinates)
            .map(|(a, b)| squared_distance(*a, *b))
            .sum::<f64>()
            / reference.len() as f64,
    )
}

/// Radius of gyration around the (mass-weighted when available) centre.
pub fn radius_of_gyration(group: &AtomGroup) -> Option<f64> {
    let center = group.center_of_mass()?;
    let total_mass = group.total_mass();
    if total_mass > 0.0 {
        let weighted = group
            .atoms
            .iter()
            .map(|atom| atom.mass * squared_distance(atom.position, center))
            .sum::<f64>();
        Some((weighted / total_mass).sqrt())
    } else {
        let mean = group
            .atoms
            .iter()
            .map(|atom| squared_distance(atom.position, center))
            .sum::<f64>()
            / group.len() as f64;
        Some(mean.sqrt())
    }
}

/// Histogram-based radial distribution function for two point sets.
pub fn radial_distribution_function(
    first: &[[f64; 3]],
    second: &[[f64; 3]],
    bin_width: f64,
    max_radius: f64,
) -> Option<(Vec<f64>, Vec<usize>)> {
    if bin_width <= 0.0 || max_radius <= 0.0 || first.is_empty() || second.is_empty() {
        return None;
    }
    let bins = (max_radius / bin_width).ceil() as usize;
    let mut counts = vec![0usize; bins];
    for &a in first {
        for &b in second {
            let distance = squared_distance(a, b).sqrt();
            if distance < max_radius {
                let index = (distance / bin_width) as usize;
                if index < bins {
                    counts[index] += 1;
                }
            }
        }
    }
    let edges = (0..=bins).map(|index| index as f64 * bin_width).collect();
    Some((edges, counts))
}

/// A small analysis interface for running a calculation over every frame.
pub trait Analysis {
    type Output;

    fn prepare(&mut self, _trajectory: &Trajectory) {}
    fn process_frame(&mut self, frame: &Frame) -> Result<(), String>;
    fn finalize(self) -> Self::Output;

    fn run(mut self, trajectory: &Trajectory) -> Result<Self::Output, String>
    where
        Self: Sized,
    {
        self.prepare(trajectory);
        for frame in trajectory {
            self.process_frame(frame)?;
        }
        Ok(self.finalize())
    }
}

/// Collects the centre of mass for every frame using fixed per-atom masses.
#[derive(Clone, Debug)]
pub struct CenterOfMassAnalysis {
    masses: Vec<f64>,
    values: Vec<[f64; 3]>,
}

impl CenterOfMassAnalysis {
    pub fn new(masses: Vec<f64>) -> Self {
        Self {
            masses,
            values: Vec::new(),
        }
    }

    pub fn values(&self) -> &[[f64; 3]] {
        &self.values
    }
}

impl Analysis for CenterOfMassAnalysis {
    type Output = Vec<[f64; 3]>;

    fn process_frame(&mut self, frame: &Frame) -> Result<(), String> {
        if frame.positions.len() != self.masses.len() {
            return Err("mass and coordinate lengths differ".to_string());
        }
        let positions: Vec<crate::geometry::Vec3> =
            frame.positions.iter().copied().map(Into::into).collect();
        let center = crate::geometry::try_center_of_mass(&positions, &self.masses)
            .ok_or_else(|| "cannot calculate centre of mass".to_string())?;
        self.values.push(center.to_array());
        Ok(())
    }

    fn finalize(self) -> Self::Output {
        self.values
    }
}

fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, Frame, Universe};

    #[test]
    fn contacts_are_symmetric_and_exclude_diagonal() {
        let matrix = contact_matrix(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 1.1);
        assert_eq!(matrix, vec![vec![false, true], vec![true, false]]);
    }

    #[test]
    fn msd_and_rdf_have_expected_bins() {
        assert_eq!(
            mean_square_displacement(&[[0.0, 0.0, 0.0]], &[[2.0, 0.0, 0.0]]),
            Some(4.0)
        );
        let (_, counts) =
            radial_distribution_function(&[[0.0, 0.0, 0.0]], &[[0.5, 0.0, 0.0]], 1.0, 3.0).unwrap();
        assert_eq!(counts, vec![1, 0, 0]);
    }

    #[test]
    fn analysis_runs_over_frames() {
        let universe =
            Universe::from_atoms(vec![Atom::new(0, "X", [0.0, 0.0, 0.0]).with_mass(1.0)]);
        let mut trajectory = universe.trajectory.clone();
        trajectory.frames.push(Frame::new(vec![[1.0, 0.0, 0.0]]));
        let result = CenterOfMassAnalysis::new(vec![1.0])
            .run(&trajectory)
            .unwrap();
        assert_eq!(result, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    }
}
