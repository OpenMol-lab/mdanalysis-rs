//! Periodic and non-periodic neighbour searches.
//!
//! The original MDAnalysis project exposes several spatial backends (the
//! brute-force distance routine, a periodic KD tree, and a cell grid).  This
//! module provides one deterministic, dependency-free implementation with the
//! same useful semantics.  It deliberately favours predictable behaviour and
//! a small API over a backend-specific tree representation; callers can use it
//! for both one-off and repeated searches.

use std::fmt;

use crate::core::AtomGroup;
use crate::distances::{self, Coordinate, DistanceError};

/// Errors returned by neighbour-search construction or queries.
#[derive(Clone, Debug, PartialEq)]
pub enum NeighborSearchError {
    /// A query was attempted before coordinates were installed.
    Unbuilt,
    /// Periodic searches require a cutoff at coordinate-installation time.
    MissingCutoff,
    /// A query radius is negative, non-finite, or exceeds the configured
    /// periodic-image cutoff.
    InvalidRadius(f64),
    /// The coordinate collection or box was invalid.
    InvalidInput(String),
    /// An underlying distance operation rejected the unit cell.
    Distance(DistanceError),
}

impl fmt::Display for NeighborSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unbuilt => formatter.write_str("neighbour search has no coordinates"),
            Self::MissingCutoff => {
                formatter.write_str("a periodic neighbour search requires a coordinate cutoff")
            }
            Self::InvalidRadius(radius) => write!(
                formatter,
                "search radius must be finite and non-negative and no greater than the configured cutoff (got {radius})"
            ),
            Self::InvalidInput(message) => {
                write!(formatter, "invalid neighbour-search input: {message}")
            }
            Self::Distance(error) => write!(formatter, "distance error: {error}"),
        }
    }
}

impl std::error::Error for NeighborSearchError {}

impl From<DistanceError> for NeighborSearchError {
    fn from(error: DistanceError) -> Self {
        Self::Distance(error)
    }
}

/// Result of a neighbour query between a set of centers and installed points.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborPairs {
    /// Pairs are `(center_index, installed_coordinate_index)` in row-major
    /// order.  A center can match the same installed point only once.
    pub pairs: Vec<(usize, usize)>,
    /// Distances corresponding to [`Self::pairs`].
    pub distances: Vec<f64>,
}

impl NeighborPairs {
    /// Return only the installed-coordinate indices, preserving first-seen
    /// order and removing duplicates.
    #[must_use]
    pub fn unique_indices(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for &(_, index) in &self.pairs {
            if !result.contains(&index) {
                result.push(index);
            }
        }
        result
    }
}

/// A reusable brute-force neighbour search over a fixed coordinate set.
#[derive(Clone, Debug, Default)]
pub struct NeighborSearch {
    coordinates: Option<Vec<Coordinate>>,
    /// Six-dimensional unit-cell dimensions.  `None` disables PBC.
    dimensions: Option<[f64; 6]>,
    /// Maximum image distance used when constructing a periodic search.
    cutoff: Option<f64>,
}

impl NeighborSearch {
    /// Construct a non-periodic search.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a search that uses the supplied orthorhombic or triclinic
    /// dimensions for minimum-image distances.
    pub fn with_box(dimensions: [f64; 6]) -> Result<Self, NeighborSearchError> {
        validate_dimensions(dimensions)?;
        Ok(Self {
            coordinates: None,
            dimensions: Some(dimensions),
            cutoff: None,
        })
    }

    /// Whether this search applies periodic boundary conditions.
    #[must_use]
    pub const fn is_periodic(&self) -> bool {
        self.dimensions.is_some()
    }

    /// Install coordinates and, for periodic searches, generate the required
    /// image-search cutoff.  Coordinates are copied so the search remains
    /// valid if the caller later mutates its input.
    pub fn set_coords(
        &mut self,
        coordinates: &[[f64; 3]],
        cutoff: Option<f64>,
    ) -> Result<(), NeighborSearchError> {
        if coordinates
            .iter()
            .flat_map(|coordinate| coordinate.iter())
            .any(|value| !value.is_finite())
        {
            return Err(NeighborSearchError::InvalidInput(
                "coordinates must be finite".to_owned(),
            ));
        }
        match self.dimensions {
            Some(dimensions) => {
                let cutoff = cutoff.ok_or(NeighborSearchError::MissingCutoff)?;
                validate_radius(cutoff)?;
                // Validate once here, then wrap points into the primary cell.
                self.coordinates = Some(distances::apply_pbc(coordinates, dimensions)?);
                self.cutoff = Some(cutoff);
            }
            None => {
                if cutoff.is_some() {
                    return Err(NeighborSearchError::InvalidInput(
                        "a non-periodic search cannot receive a cutoff at set_coords".to_owned(),
                    ));
                }
                self.coordinates = Some(coordinates.to_vec());
                self.cutoff = None;
            }
        }
        Ok(())
    }

    /// Return the currently installed coordinates.
    pub fn coordinates(&self) -> Result<&[[f64; 3]], NeighborSearchError> {
        self.coordinates
            .as_deref()
            .ok_or(NeighborSearchError::Unbuilt)
    }

    /// Find installed points within `radius` of one or more centers.
    pub fn search(
        &self,
        centers: &[[f64; 3]],
        radius: f64,
    ) -> Result<Vec<usize>, NeighborSearchError> {
        Ok(self
            .search_with_distances(centers, radius)?
            .unique_indices())
    }

    /// Find installed points and retain pairwise distances for each center.
    pub fn search_with_distances(
        &self,
        centers: &[[f64; 3]],
        radius: f64,
    ) -> Result<NeighborPairs, NeighborSearchError> {
        let coordinates = self.coordinates()?.to_vec();
        self.validate_query_radius(radius)?;
        let mut pairs = Vec::new();
        let mut distances = Vec::new();
        for (center_index, &center) in centers.iter().enumerate() {
            if center.iter().any(|value| !value.is_finite()) {
                return Err(NeighborSearchError::InvalidInput(
                    "query coordinates must be finite".to_owned(),
                ));
            }
            for (coordinate_index, &coordinate) in coordinates.iter().enumerate() {
                let distance = self.distance(center, coordinate)?;
                if distance <= radius {
                    pairs.push((center_index, coordinate_index));
                    distances.push(distance);
                }
            }
        }
        Ok(NeighborPairs { pairs, distances })
    }

    /// Find all unique installed-coordinate pairs within `radius`.
    pub fn search_pairs(&self, radius: f64) -> Result<Vec<(usize, usize)>, NeighborSearchError> {
        let coordinates = self.coordinates()?.to_vec();
        self.validate_query_radius(radius)?;
        let mut pairs = Vec::new();
        for first in 0..coordinates.len() {
            for second in (first + 1)..coordinates.len() {
                if self.distance(coordinates[first], coordinates[second])? <= radius {
                    pairs.push((first, second));
                }
            }
        }
        Ok(pairs)
    }

    /// Search a second center collection against installed coordinates.
    pub fn search_tree(
        &self,
        centers: &[[f64; 3]],
        radius: f64,
    ) -> Result<NeighborPairs, NeighborSearchError> {
        self.search_with_distances(centers, radius)
    }

    fn validate_query_radius(&self, radius: f64) -> Result<(), NeighborSearchError> {
        validate_radius(radius)?;
        if let Some(cutoff) = self.cutoff
            && radius > cutoff
        {
            return Err(NeighborSearchError::InvalidRadius(radius));
        }
        Ok(())
    }

    fn distance(&self, first: Coordinate, second: Coordinate) -> Result<f64, NeighborSearchError> {
        if let Some(dimensions) = self.dimensions {
            let displacement = [
                first[0] - second[0],
                first[1] - second[1],
                first[2] - second[2],
            ];
            Ok(
                distances::minimum_image_triclinic(displacement, dimensions)?
                    .iter()
                    .map(|component| component * component)
                    .sum::<f64>()
                    .sqrt(),
            )
        } else {
            Ok(((first[0] - second[0]).powi(2)
                + (first[1] - second[1]).powi(2)
                + (first[2] - second[2]).powi(2))
            .sqrt())
        }
    }
}

/// Compatibility spelling matching the Python backend's class name.
pub type PeriodicKDTree = NeighborSearch;

/// A level at which atom-neighbour results can be grouped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchLevel {
    /// Return atom indices.
    Atom,
    /// Return residue indices containing matching atoms.
    Residue,
    /// Return segment indices containing matching atoms.
    Segment,
}

/// Atom-group convenience wrapper around [`NeighborSearch`].
#[derive(Clone, Debug)]
pub struct AtomNeighborSearch {
    atom_group: AtomGroup,
    search: NeighborSearch,
}

impl AtomNeighborSearch {
    /// Build a search over a copied atom group.
    pub fn new(
        atom_group: &AtomGroup,
        dimensions: Option<[f64; 6]>,
    ) -> Result<Self, NeighborSearchError> {
        let mut search = match dimensions {
            Some(dimensions) => NeighborSearch::with_box(dimensions)?,
            None => NeighborSearch::new(),
        };
        let coordinates = atom_group.positions();
        search.set_coords(&coordinates, None)?;
        Ok(Self {
            atom_group: atom_group.clone(),
            search,
        })
    }

    /// Build and install an atom group with an explicit periodic image cutoff.
    pub fn with_cutoff(
        atom_group: &AtomGroup,
        dimensions: Option<[f64; 6]>,
        cutoff: Option<f64>,
    ) -> Result<Self, NeighborSearchError> {
        let mut search = match dimensions {
            Some(dimensions) => NeighborSearch::with_box(dimensions)?,
            None => NeighborSearch::new(),
        };
        search.set_coords(&atom_group.positions(), cutoff)?;
        Ok(Self {
            atom_group: atom_group.clone(),
            search,
        })
    }

    /// Search from arbitrary center coordinates and return atom indices.
    pub fn search(
        &self,
        centers: &[[f64; 3]],
        radius: f64,
    ) -> Result<Vec<usize>, NeighborSearchError> {
        self.search.search(centers, radius)
    }

    /// Search around one installed atom.
    pub fn search_atom(
        &self,
        atom_index: usize,
        radius: f64,
    ) -> Result<Vec<usize>, NeighborSearchError> {
        let atom = self.atom_group.get(atom_index).ok_or_else(|| {
            NeighborSearchError::InvalidInput(format!("atom index {atom_index} is out of bounds"))
        })?;
        self.search.search(&[atom.position], radius)
    }

    /// Search around selected installed atoms and group matching results at a
    /// topology level.  Returned indices are sorted and duplicate-free.
    pub fn search_level(
        &self,
        atom_indices: &[usize],
        radius: f64,
        level: SearchLevel,
    ) -> Result<Vec<usize>, NeighborSearchError> {
        let centers: Vec<Coordinate> = atom_indices
            .iter()
            .map(|&index| {
                self.atom_group
                    .get(index)
                    .map(|atom| atom.position)
                    .ok_or_else(|| {
                        NeighborSearchError::InvalidInput(format!(
                            "atom index {index} is out of bounds"
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        let indices = self.search.search(&centers, radius)?;
        let mut result = match level {
            SearchLevel::Atom => indices,
            SearchLevel::Residue => indices
                .iter()
                .map(|&index| self.atom_group[index].residue_index)
                .collect(),
            SearchLevel::Segment => indices
                .iter()
                .map(|&index| self.atom_group[index].segment_index)
                .collect(),
        };
        result.sort_unstable();
        result.dedup();
        Ok(result)
    }
}

fn validate_radius(radius: f64) -> Result<(), NeighborSearchError> {
    if radius.is_finite() && radius >= 0.0 {
        Ok(())
    } else {
        Err(NeighborSearchError::InvalidRadius(radius))
    }
}

fn validate_dimensions(dimensions: [f64; 6]) -> Result<(), NeighborSearchError> {
    // Calling a transform also validates the positive lengths/angles and
    // rejects physically singular triclinic combinations.
    distances::transform_s_to_r(&[[0.0; 3]], dimensions)
        .map(|_| ())
        .map_err(NeighborSearchError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Atom;

    #[test]
    fn non_periodic_search_returns_unique_indices_and_pairs() {
        let mut search = NeighborSearch::new();
        search
            .set_coords(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], None)
            .unwrap();
        assert_eq!(
            search
                .search(&[[0.5, 0.0, 0.0], [3.0, 0.0, 0.0]], 0.6)
                .unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(search.search_pairs(1.1).unwrap(), vec![(0, 1)]);
        assert_eq!(search.search_pairs(2.0).unwrap(), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn periodic_search_handles_triclinic_boundaries() {
        let dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 60.0];
        let mut search = NeighborSearch::with_box(dimensions).unwrap();
        search
            .set_coords(&[[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]], Some(1.0))
            .unwrap();
        assert_eq!(search.search(&[[0.0, 0.1, 0.1]], 0.3).unwrap(), vec![0, 1]);
        assert!(matches!(
            search.search(&[[0.0; 3]], 1.1),
            Err(NeighborSearchError::InvalidRadius(_))
        ));
    }

    #[test]
    fn atom_wrapper_groups_results() {
        let mut first = Atom::new(0, "A", [0.0, 0.0, 0.0]);
        first.residue_index = 1;
        first.segment_index = 2;
        let mut second = Atom::new(1, "B", [1.0, 0.0, 0.0]);
        second.residue_index = 3;
        second.segment_index = 4;
        let group = AtomGroup::new(vec![first, second]);
        let search = AtomNeighborSearch::new(&group, None).unwrap();
        assert_eq!(search.search_atom(0, 1.0).unwrap(), vec![0, 1]);
        assert_eq!(
            search
                .search_level(&[0], 1.0, SearchLevel::Residue)
                .unwrap(),
            vec![1, 3]
        );
        assert!(matches!(
            search.search_atom(9, 1.0),
            Err(NeighborSearchError::InvalidInput(_))
        ));
    }

    #[test]
    fn invalid_configuration_is_reported() {
        let periodic = NeighborSearch::with_box([10.0, 10.0, 10.0, 0.0, 90.0, 90.0]);
        assert!(periodic.is_err());
        let mut search = NeighborSearch::new();
        assert!(matches!(
            search.set_coords(&[[f64::NAN, 0.0, 0.0]], None),
            Err(NeighborSearchError::InvalidInput(_))
        ));
        assert!(matches!(
            search.search_pairs(1.0),
            Err(NeighborSearchError::Unbuilt)
        ));
    }
}
