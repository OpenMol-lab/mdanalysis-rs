//! Reusable topology and atom-group operations.
//!
//! The core objects intentionally stay small and data-oriented.  This module
//! adds the operations that are useful when working with collections of atoms
//! without adding more state to [`crate::core`].  Methods are exposed through
//! [`TopologyGroupExt`], so applications can opt in with a single import.

use crate::core::{Atom, AtomGroup, Topology};
use crate::geometry::{Vec3, angle, dihedral_points, distance};

/// A bond index pair and its geometric length.
pub type BondLength = (usize, usize, f64);

/// An angle index triple `(first, center, last)` and its value in radians.
pub type AngleValue = (usize, usize, usize, f64);

/// A dihedral index quadruple and its value in radians.
pub type DihedralValue = (usize, usize, usize, usize, f64);

/// Convenience operations shared by atoms, atom groups, and topologies.
///
/// The batch accessors deliberately use names that do not overlap the
/// existing inherent methods on [`AtomGroup`] (for example, `coordinates`
/// rather than `positions`).
pub trait TopologyGroupExt {
    /// Replace the position of a single-item collection.
    ///
    /// Collections containing zero or multiple atoms return an input error;
    /// use [`Self::update_positions`] for a collection update.
    fn update_position(&mut self, position: [f64; 3]) -> crate::Result<()>;

    /// Replace all coordinates, requiring exactly one coordinate per item.
    fn update_positions(&mut self, positions: &[[f64; 3]]) -> crate::Result<()>;

    /// Return a copy of all coordinates in collection order.
    fn coordinates(&self) -> Vec<[f64; 3]>;

    /// Return atom indices in collection order.
    fn atom_indices(&self) -> Vec<usize>;

    /// Return selected atom metadata in collection order.
    fn atom_names(&self) -> Vec<String>;
    fn atom_types(&self) -> Vec<Option<String>>;
    fn residue_ids(&self) -> Vec<i32>;
    fn residue_names(&self) -> Vec<String>;
    fn segment_ids(&self) -> Vec<String>;
    fn chain_ids(&self) -> Vec<String>;
    fn elements(&self) -> Vec<Option<String>>;
    fn masses_array(&self) -> Vec<f64>;
    fn charges_array(&self) -> Vec<f64>;
    fn velocities(&self) -> Vec<Option<[f64; 3]>>;
    fn forces(&self) -> Vec<Option<[f64; 3]>>;
    fn temperature_factors(&self) -> Vec<Option<f64>>;
    fn occupancies(&self) -> Vec<Option<f64>>;
    fn residue_indices(&self) -> Vec<usize>;
    fn segment_indices(&self) -> Vec<usize>;

    /// Return first-seen unique indices, preserving collection order.
    fn unique_indices(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for index in self.atom_indices() {
            if !result.contains(&index) {
                result.push(index);
            }
        }
        result
    }

    /// Return all indices sorted numerically.  Duplicates are retained.
    fn sorted_indices(&self) -> Vec<usize> {
        let mut result = self.atom_indices();
        result.sort_unstable();
        result
    }

    /// Return unique indices sorted numerically.
    fn unique_sorted_indices(&self) -> Vec<usize> {
        let mut result = self.unique_indices();
        result.sort_unstable();
        result
    }

    /// Group atoms by residue, preserving first-seen residue order.
    fn residue_groups(&self) -> Vec<AtomGroup>;

    /// Group atoms by segment, preserving first-seen segment order.
    fn segment_groups(&self) -> Vec<AtomGroup>;

    /// Return an atom-index adjacency list.  Collections without topology
    /// connectivity (an individual atom or an [`AtomGroup`]) contain empty
    /// neighbor lists.
    fn adjacency_list(&self) -> Vec<Vec<usize>>;

    /// Alias for [`Self::adjacency_list`].
    fn neighbors(&self) -> Vec<Vec<usize>> {
        self.adjacency_list()
    }

    /// Compute the length of each bond in topology order.
    fn bond_lengths(&self) -> Vec<BondLength>;

    /// Compute all angles implied by the bond graph.  Values are in radians.
    fn angle_values(&self) -> Vec<AngleValue>;

    /// Compute all dihedrals implied by the bond graph.  Values are in
    /// radians, with one orientation per central bond.
    fn dihedral_values(&self) -> Vec<DihedralValue>;

    // Small aliases make the extension convenient while keeping the primary
    // names distinct from existing inherent methods.
    fn positions_array(&self) -> Vec<[f64; 3]> {
        self.coordinates()
    }

    fn names_array(&self) -> Vec<String> {
        self.atom_names()
    }

    fn resids_array(&self) -> Vec<i32> {
        self.residue_ids()
    }

    fn resnames_array(&self) -> Vec<String> {
        self.residue_names()
    }

    fn segids_array(&self) -> Vec<String> {
        self.segment_ids()
    }

    fn compute_bond_lengths(&self) -> Vec<BondLength> {
        self.bond_lengths()
    }

    fn compute_angles(&self) -> Vec<AngleValue> {
        self.angle_values()
    }

    fn compute_dihedrals(&self) -> Vec<DihedralValue> {
        self.dihedral_values()
    }
}

fn invalid_position_count(expected: usize, actual: usize) -> crate::Error {
    crate::Error::InvalidInput(format!(
        "expected {expected} coordinates, received {actual}"
    ))
}

fn group_atoms_by<F>(atoms: &[Atom], key: F) -> Vec<AtomGroup>
where
    F: Fn(&Atom) -> String,
{
    let mut groups: Vec<(String, Vec<Atom>)> = Vec::new();
    for atom in atoms {
        let group_key = key(atom);
        if let Some((_, members)) = groups
            .iter_mut()
            .find(|(candidate, _)| *candidate == group_key)
        {
            members.push(atom.clone());
        } else {
            groups.push((group_key, vec![atom.clone()]));
        }
    }
    groups
        .into_iter()
        .map(|(_, atoms)| AtomGroup::new(atoms))
        .collect()
}

fn residue_key(atom: &Atom) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        atom.residue_index, atom.resid, atom.resname, atom.segment_index, atom.segid
    )
}

fn segment_key(atom: &Atom) -> String {
    format!("{}\u{1f}{}", atom.segment_index, atom.segid)
}

fn groups_from_atoms(atoms: &[Atom]) -> (Vec<AtomGroup>, Vec<AtomGroup>) {
    (
        group_atoms_by(atoms, residue_key),
        group_atoms_by(atoms, segment_key),
    )
}

fn no_bonds() -> Vec<BondLength> {
    Vec::new()
}

impl TopologyGroupExt for Atom {
    fn update_position(&mut self, position: [f64; 3]) -> crate::Result<()> {
        self.position = position;
        Ok(())
    }

    fn update_positions(&mut self, positions: &[[f64; 3]]) -> crate::Result<()> {
        if positions.len() != 1 {
            return Err(invalid_position_count(1, positions.len()));
        }
        self.position = positions[0];
        Ok(())
    }

    fn coordinates(&self) -> Vec<[f64; 3]> {
        vec![self.position]
    }

    fn atom_indices(&self) -> Vec<usize> {
        vec![self.index]
    }

    fn atom_names(&self) -> Vec<String> {
        vec![self.name.clone()]
    }

    fn residue_ids(&self) -> Vec<i32> {
        vec![self.resid]
    }

    fn residue_names(&self) -> Vec<String> {
        vec![self.resname.clone()]
    }

    fn atom_types(&self) -> Vec<Option<String>> {
        vec![self.atom_type.clone()]
    }

    fn segment_ids(&self) -> Vec<String> {
        vec![self.segid.clone()]
    }

    fn chain_ids(&self) -> Vec<String> {
        vec![self.chain_id.clone()]
    }

    fn elements(&self) -> Vec<Option<String>> {
        vec![self.element.clone()]
    }

    fn masses_array(&self) -> Vec<f64> {
        vec![self.mass]
    }

    fn charges_array(&self) -> Vec<f64> {
        vec![self.charge]
    }

    fn velocities(&self) -> Vec<Option<[f64; 3]>> {
        vec![self.velocity]
    }

    fn forces(&self) -> Vec<Option<[f64; 3]>> {
        vec![self.force]
    }

    fn temperature_factors(&self) -> Vec<Option<f64>> {
        vec![self.temp_factor]
    }

    fn occupancies(&self) -> Vec<Option<f64>> {
        vec![self.occupancy]
    }

    fn residue_indices(&self) -> Vec<usize> {
        vec![self.residue_index]
    }

    fn segment_indices(&self) -> Vec<usize> {
        vec![self.segment_index]
    }

    fn residue_groups(&self) -> Vec<AtomGroup> {
        vec![AtomGroup::new(vec![self.clone()])]
    }

    fn segment_groups(&self) -> Vec<AtomGroup> {
        vec![AtomGroup::new(vec![self.clone()])]
    }

    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        vec![Vec::new()]
    }

    fn bond_lengths(&self) -> Vec<BondLength> {
        no_bonds()
    }

    fn angle_values(&self) -> Vec<AngleValue> {
        Vec::new()
    }

    fn dihedral_values(&self) -> Vec<DihedralValue> {
        Vec::new()
    }
}

impl TopologyGroupExt for AtomGroup {
    fn update_position(&mut self, position: [f64; 3]) -> crate::Result<()> {
        if self.atoms.len() != 1 {
            return Err(invalid_position_count(1, self.atoms.len()));
        }
        self.atoms[0].position = position;
        Ok(())
    }

    fn update_positions(&mut self, positions: &[[f64; 3]]) -> crate::Result<()> {
        if positions.len() != self.atoms.len() {
            return Err(invalid_position_count(self.atoms.len(), positions.len()));
        }
        for (atom, position) in self.atoms.iter_mut().zip(positions) {
            atom.position = *position;
        }
        Ok(())
    }

    fn coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|atom| atom.position).collect()
    }

    fn atom_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.index).collect()
    }

    fn atom_names(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.name.clone()).collect()
    }

    fn atom_types(&self) -> Vec<Option<String>> {
        self.atoms
            .iter()
            .map(|atom| atom.atom_type.clone())
            .collect()
    }

    fn residue_ids(&self) -> Vec<i32> {
        self.atoms.iter().map(|atom| atom.resid).collect()
    }

    fn residue_names(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.resname.clone()).collect()
    }

    fn segment_ids(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.segid.clone()).collect()
    }

    fn chain_ids(&self) -> Vec<String> {
        self.atoms
            .iter()
            .map(|atom| atom.chain_id.clone())
            .collect()
    }

    fn elements(&self) -> Vec<Option<String>> {
        self.atoms.iter().map(|atom| atom.element.clone()).collect()
    }

    fn masses_array(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.mass).collect()
    }

    fn charges_array(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.charge).collect()
    }

    fn velocities(&self) -> Vec<Option<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.velocity).collect()
    }

    fn forces(&self) -> Vec<Option<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.force).collect()
    }

    fn temperature_factors(&self) -> Vec<Option<f64>> {
        self.atoms.iter().map(|atom| atom.temp_factor).collect()
    }

    fn occupancies(&self) -> Vec<Option<f64>> {
        self.atoms.iter().map(|atom| atom.occupancy).collect()
    }

    fn residue_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.residue_index).collect()
    }

    fn segment_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.segment_index).collect()
    }

    fn residue_groups(&self) -> Vec<AtomGroup> {
        groups_from_atoms(&self.atoms).0
    }

    fn segment_groups(&self) -> Vec<AtomGroup> {
        groups_from_atoms(&self.atoms).1
    }

    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        vec![Vec::new(); self.atoms.len()]
    }

    fn bond_lengths(&self) -> Vec<BondLength> {
        no_bonds()
    }

    fn angle_values(&self) -> Vec<AngleValue> {
        Vec::new()
    }

    fn dihedral_values(&self) -> Vec<DihedralValue> {
        Vec::new()
    }
}

impl TopologyGroupExt for Topology {
    fn update_position(&mut self, position: [f64; 3]) -> crate::Result<()> {
        if self.atoms.len() != 1 {
            return Err(invalid_position_count(1, self.atoms.len()));
        }
        self.atoms[0].position = position;
        Ok(())
    }

    fn update_positions(&mut self, positions: &[[f64; 3]]) -> crate::Result<()> {
        if positions.len() != self.atoms.len() {
            return Err(invalid_position_count(self.atoms.len(), positions.len()));
        }
        for (atom, position) in self.atoms.iter_mut().zip(positions) {
            atom.position = *position;
        }
        Ok(())
    }

    fn coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|atom| atom.position).collect()
    }

    fn atom_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.index).collect()
    }

    fn atom_names(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.name.clone()).collect()
    }

    fn atom_types(&self) -> Vec<Option<String>> {
        self.atoms
            .iter()
            .map(|atom| atom.atom_type.clone())
            .collect()
    }

    fn residue_ids(&self) -> Vec<i32> {
        self.atoms.iter().map(|atom| atom.resid).collect()
    }

    fn residue_names(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.resname.clone()).collect()
    }

    fn segment_ids(&self) -> Vec<String> {
        self.atoms.iter().map(|atom| atom.segid.clone()).collect()
    }

    fn chain_ids(&self) -> Vec<String> {
        self.atoms
            .iter()
            .map(|atom| atom.chain_id.clone())
            .collect()
    }

    fn elements(&self) -> Vec<Option<String>> {
        self.atoms.iter().map(|atom| atom.element.clone()).collect()
    }

    fn masses_array(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.mass).collect()
    }

    fn charges_array(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.charge).collect()
    }

    fn velocities(&self) -> Vec<Option<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.velocity).collect()
    }

    fn forces(&self) -> Vec<Option<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.force).collect()
    }

    fn temperature_factors(&self) -> Vec<Option<f64>> {
        self.atoms.iter().map(|atom| atom.temp_factor).collect()
    }

    fn occupancies(&self) -> Vec<Option<f64>> {
        self.atoms.iter().map(|atom| atom.occupancy).collect()
    }

    fn residue_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.residue_index).collect()
    }

    fn segment_indices(&self) -> Vec<usize> {
        self.atoms.iter().map(|atom| atom.segment_index).collect()
    }

    fn residue_groups(&self) -> Vec<AtomGroup> {
        groups_from_atoms(&self.atoms).0
    }

    fn segment_groups(&self) -> Vec<AtomGroup> {
        groups_from_atoms(&self.atoms).1
    }

    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adjacency = vec![Vec::new(); self.atoms.len()];
        for bond in &self.bonds {
            if bond.atom1 >= self.atoms.len() || bond.atom2 >= self.atoms.len() {
                continue;
            }
            if !adjacency[bond.atom1].contains(&bond.atom2) {
                adjacency[bond.atom1].push(bond.atom2);
            }
            if !adjacency[bond.atom2].contains(&bond.atom1) {
                adjacency[bond.atom2].push(bond.atom1);
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }
        adjacency
    }

    fn bond_lengths(&self) -> Vec<BondLength> {
        self.bonds
            .iter()
            .filter_map(|bond| {
                let first = self.atoms.get(bond.atom1)?;
                let second = self.atoms.get(bond.atom2)?;
                Some((
                    bond.atom1,
                    bond.atom2,
                    distance(first.position.into(), second.position.into()),
                ))
            })
            .collect()
    }

    fn angle_values(&self) -> Vec<AngleValue> {
        let adjacency = self.adjacency_list();
        let mut values = Vec::new();
        for (center, neighbors) in adjacency.iter().enumerate() {
            for left in 0..neighbors.len() {
                for right in (left + 1)..neighbors.len() {
                    let first = neighbors[left];
                    let last = neighbors[right];
                    let (Some(a), Some(b), Some(c)) = (
                        self.atoms.get(first),
                        self.atoms.get(center),
                        self.atoms.get(last),
                    ) else {
                        continue;
                    };
                    let value = angle(
                        Vec3::from(a.position) - Vec3::from(b.position),
                        Vec3::from(c.position) - Vec3::from(b.position),
                    );
                    values.push((first, center, last, value));
                }
            }
        }
        values
    }

    fn dihedral_values(&self) -> Vec<DihedralValue> {
        let adjacency = self.adjacency_list();
        let mut values = Vec::new();
        for center_left in 0..adjacency.len() {
            for &center_right in &adjacency[center_left] {
                // One orientation per central bond avoids reverse duplicates.
                if center_left >= center_right {
                    continue;
                }
                for &first in &adjacency[center_left] {
                    if first == center_right {
                        continue;
                    }
                    for &last in &adjacency[center_right] {
                        if last == center_left {
                            continue;
                        }
                        let (Some(a), Some(b), Some(c), Some(d)) = (
                            self.atoms.get(first),
                            self.atoms.get(center_left),
                            self.atoms.get(center_right),
                            self.atoms.get(last),
                        ) else {
                            continue;
                        };
                        values.push((
                            first,
                            center_left,
                            center_right,
                            last,
                            dihedral_points(
                                a.position.into(),
                                b.position.into(),
                                c.position.into(),
                                d.position.into(),
                            ),
                        ));
                    }
                }
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Bond;

    fn topology() -> Topology {
        let mut first = Atom::new(0, "A", [0.0, 0.0, 0.0]);
        first.resid = 1;
        first.resname = "RES".into();
        first.segid = "A".into();
        let mut second = Atom::new(1, "B", [1.0, 0.0, 0.0]);
        second.resid = 1;
        second.resname = "RES".into();
        second.segid = "A".into();
        let mut third = Atom::new(2, "C", [1.0, 1.0, 0.0]);
        third.resid = 2;
        third.resname = "RES".into();
        third.segid = "B".into();
        let mut topology = Topology::new(vec![first, second, third]);
        topology.add_bond(Bond::new(0, 1));
        topology.add_bond(Bond::new(1, 2));
        topology
    }

    #[test]
    fn updates_coordinates_and_reads_attributes() {
        let mut topology = topology();
        topology
            .update_positions(&[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
            .unwrap();
        assert_eq!(topology.coordinates()[0], [2.0, 0.0, 0.0]);
        assert_eq!(topology.atom_names(), vec!["A", "B", "C"]);
        assert!(topology.update_positions(&[[0.0, 0.0, 0.0]]).is_err());
    }

    #[test]
    fn groups_and_indices_are_stable() {
        let mut topology = topology();
        topology.atoms.push(topology.atoms[0].clone());
        let group = AtomGroup::new(topology.atoms.clone());
        assert_eq!(group.unique_indices(), vec![0, 1, 2]);
        assert_eq!(group.sorted_indices(), vec![0, 0, 1, 2]);
        assert_eq!(topology.residue_groups().len(), 2);
        assert_eq!(topology.segment_groups().len(), 2);
    }

    #[test]
    fn connectivity_geometry_is_computed() {
        let topology = topology();
        assert_eq!(
            topology.adjacency_list(),
            vec![vec![1], vec![0, 2], vec![1]]
        );
        let bonds = topology.bond_lengths();
        assert_eq!(bonds.len(), 2);
        assert!((bonds[0].2 - 1.0).abs() < 1.0e-12);
        let angles = topology.angle_values();
        assert_eq!(angles.len(), 1);
        assert!((angles[0].3 - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert!(topology.dihedral_values().is_empty());
    }
}
