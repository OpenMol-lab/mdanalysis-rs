//! Core topology and trajectory objects.

use crate::coordinates::CoordinateFile;
use crate::pdb::{PdbAtom, PdbStructure, read_pdb};
use crate::selection::{AtomLike, SelectionError, select};
use std::path::Path;

/// A single atom and its topology metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub index: usize,
    pub name: String,
    pub atom_type: Option<String>,
    pub element: Option<String>,
    pub mass: f64,
    pub charge: f64,
    pub resid: i32,
    pub residue_index: usize,
    pub resname: String,
    pub segid: String,
    pub segment_index: usize,
    pub chain_id: String,
    pub position: [f64; 3],
    pub velocity: Option<[f64; 3]>,
    pub force: Option<[f64; 3]>,
    pub temp_factor: Option<f64>,
    pub occupancy: Option<f64>,
}

impl Atom {
    pub fn new(index: usize, name: impl Into<String>, position: [f64; 3]) -> Self {
        Self {
            index,
            name: name.into(),
            atom_type: None,
            element: None,
            mass: 0.0,
            charge: 0.0,
            resid: 1,
            residue_index: 0,
            resname: "UNK".to_string(),
            segid: "SYSTEM".to_string(),
            segment_index: 0,
            chain_id: String::new(),
            position,
            velocity: None,
            force: None,
            temp_factor: None,
            occupancy: None,
        }
    }

    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    pub fn distance_to(&self, other: &Self) -> f64 {
        crate::geometry::distance(self.position.into(), other.position.into())
    }
}

impl AtomLike for Atom {
    fn index(&self) -> usize {
        self.index
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn resname(&self) -> &str {
        &self.resname
    }
    fn resid(&self) -> i32 {
        self.resid
    }
    fn element(&self) -> Option<&str> {
        self.element.as_deref()
    }
    fn chain_id(&self) -> &str {
        &self.chain_id
    }
    fn segid(&self) -> &str {
        &self.segid
    }
}

impl From<PdbAtom> for Atom {
    fn from(atom: PdbAtom) -> Self {
        let element = atom.element.clone();
        let position = atom.position();
        Self {
            index: atom.serial.saturating_sub(1) as usize,
            name: atom.name,
            atom_type: None,
            mass: element.as_deref().and_then(element_mass).unwrap_or(0.0),
            element,
            charge: 0.0,
            resid: atom.residue_sequence,
            residue_index: 0,
            resname: atom.residue_name,
            segid: String::new(),
            segment_index: 0,
            chain_id: atom.chain_id.map(|id| id.to_string()).unwrap_or_default(),
            position,
            velocity: None,
            force: None,
            temp_factor: atom.temperature_factor,
            occupancy: atom.occupancy,
        }
    }
}

fn element_mass(element: &str) -> Option<f64> {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => Some(1.008),
        "C" => Some(12.011),
        "N" => Some(14.007),
        "O" => Some(15.999),
        "F" => Some(18.998),
        "P" => Some(30.974),
        "S" => Some(32.06),
        "CL" => Some(35.45),
        "BR" => Some(79.904),
        "I" => Some(126.904),
        "NA" => Some(22.990),
        "MG" => Some(24.305),
        "CA" => Some(40.078),
        "FE" => Some(55.845),
        "ZN" => Some(65.38),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Residue {
    pub index: usize,
    pub resid: i32,
    pub name: String,
    pub segment_index: usize,
    pub atom_indices: Vec<usize>,
}

impl Residue {
    pub fn new(index: usize, resid: i32, name: impl Into<String>, segment_index: usize) -> Self {
        Self {
            index,
            resid,
            name: name.into(),
            segment_index,
            atom_indices: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub index: usize,
    pub id: String,
    pub residue_indices: Vec<usize>,
}

impl Segment {
    pub fn new(index: usize, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            residue_indices: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub order: Option<u8>,
    pub guessed: bool,
}

impl Bond {
    pub fn new(atom1: usize, atom2: usize) -> Self {
        Self {
            atom1,
            atom2,
            order: None,
            guessed: false,
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.atom1 == index || self.atom2 == index
    }

    pub fn partner(&self, index: usize) -> Option<usize> {
        if self.atom1 == index {
            Some(self.atom2)
        } else if self.atom2 == index {
            Some(self.atom1)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Topology {
    pub atoms: Vec<Atom>,
    pub residues: Vec<Residue>,
    pub segments: Vec<Segment>,
    pub bonds: Vec<Bond>,
}

impl Topology {
    pub fn new(atoms: Vec<Atom>) -> Self {
        let mut topology = Self {
            atoms,
            ..Self::default()
        };
        topology.rebuild_hierarchy();
        topology
    }

    pub fn add_bond(&mut self, bond: Bond) {
        if bond.atom1 < self.atoms.len()
            && bond.atom2 < self.atoms.len()
            && !self.bonds.iter().any(|existing| {
                (existing.atom1 == bond.atom1 && existing.atom2 == bond.atom2)
                    || (existing.atom1 == bond.atom2 && existing.atom2 == bond.atom1)
            })
        {
            self.bonds.push(bond);
        }
    }

    pub fn rebuild_hierarchy(&mut self) {
        self.residues.clear();
        self.segments.clear();
        let mut residue_keys: Vec<(i32, String, usize, String)> = Vec::new();
        for atom_index in 0..self.atoms.len() {
            self.atoms[atom_index].index = atom_index;
            let atom = &self.atoms[atom_index];
            let key = (
                atom.resid,
                atom.resname.clone(),
                atom.segment_index,
                atom.segid.clone(),
            );
            let residue_index = residue_keys
                .iter()
                .position(|candidate| candidate == &key)
                .unwrap_or_else(|| {
                    residue_keys.push(key);
                    self.residues.push(Residue::new(
                        self.residues.len(),
                        atom.resid,
                        atom.resname.clone(),
                        atom.segment_index,
                    ));
                    self.residues.len() - 1
                });
            self.atoms[atom_index].residue_index = residue_index;
            self.residues[residue_index].atom_indices.push(atom_index);
        }
        let mut segment_ids: Vec<String> = Vec::new();
        for residue_index in 0..self.residues.len() {
            let atom_index = self.residues[residue_index].atom_indices[0];
            let id = self.atoms[atom_index].segid.clone();
            let segment_index = segment_ids
                .iter()
                .position(|candidate| candidate == &id)
                .unwrap_or_else(|| {
                    segment_ids.push(id.clone());
                    self.segments.push(Segment::new(self.segments.len(), id));
                    self.segments.len() - 1
                });
            self.residues[residue_index].segment_index = segment_index;
            self.segments[segment_index]
                .residue_indices
                .push(residue_index);
            let atom_indices = self.residues[residue_index].atom_indices.clone();
            for atom_index in atom_indices {
                self.atoms[atom_index].segment_index = segment_index;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Option<Vec<[f64; 3]>>,
    pub forces: Option<Vec<[f64; 3]>>,
    /// Unit-cell dimensions as `[a, b, c, alpha, beta, gamma]`.
    pub dimensions: Option<[f64; 6]>,
    pub time: f64,
    pub step: usize,
}

impl Frame {
    pub fn new(positions: Vec<[f64; 3]>) -> Self {
        Self {
            positions,
            velocities: None,
            forces: None,
            dimensions: None,
            time: 0.0,
            step: 0,
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trajectory {
    pub frames: Vec<Frame>,
    pub current: usize,
}

impl Trajectory {
    pub fn new(frames: Vec<Frame>) -> Self {
        Self { frames, current: 0 }
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn frame(&self, index: usize) -> Option<&Frame> {
        self.frames.get(index)
    }

    pub fn frame_mut(&mut self, index: usize) -> Option<&mut Frame> {
        self.frames.get_mut(index)
    }

    pub fn rewind(&mut self) {
        self.current = 0;
    }

    pub fn next_frame(&mut self) -> Option<&Frame> {
        let frame = self.frames.get(self.current);
        if frame.is_some() {
            self.current += 1;
        }
        frame
    }
}

impl<'a> IntoIterator for &'a Trajectory {
    type Item = &'a Frame;
    type IntoIter = std::slice::Iter<'a, Frame>;
    fn into_iter(self) -> Self::IntoIter {
        self.frames.iter()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtomGroup {
    pub atoms: Vec<Atom>,
}

impl AtomGroup {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self { atoms }
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|atom| atom.position).collect()
    }

    pub fn masses(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.mass).collect()
    }

    pub fn center_of_mass(&self) -> Option<[f64; 3]> {
        let positions: Vec<crate::geometry::Vec3> =
            self.positions().into_iter().map(Into::into).collect();
        crate::geometry::try_center_of_mass(&positions, &self.masses()).map(Into::into)
    }

    pub fn total_mass(&self) -> f64 {
        self.atoms.iter().map(|atom| atom.mass).sum()
    }

    pub fn select_atoms(&self, expression: &str) -> Result<Self, SelectionError> {
        Ok(Self::new(
            select(&self.atoms, expression)?
                .into_iter()
                .cloned()
                .collect(),
        ))
    }

    pub fn get(&self, index: usize) -> Option<&Atom> {
        self.atoms.get(index)
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self::new(self.atoms[range].to_vec())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Universe {
    pub topology: Topology,
    pub trajectory: Trajectory,
}

impl Universe {
    pub fn new(topology: Topology) -> Self {
        let positions = topology.atoms.iter().map(|atom| atom.position).collect();
        Self {
            topology,
            trajectory: Trajectory::new(vec![Frame::new(positions)]),
        }
    }

    pub fn from_atoms(atoms: Vec<Atom>) -> Self {
        Self::new(Topology::new(atoms))
    }

    pub fn from_pdb(path: impl AsRef<Path>) -> crate::Result<Self> {
        let structure = read_pdb(path)?;
        let atoms: Vec<Atom> = structure.atoms.into_iter().map(Atom::from).collect();
        let mut universe = Self::from_atoms(atoms);
        if structure.frames.len() > 1 {
            universe.trajectory =
                Trajectory::new(structure.frames.into_iter().map(Frame::new).collect());
        }
        if let Some(cell) = structure.cryst1 {
            let dimensions = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma];
            for frame in &mut universe.trajectory.frames {
                frame.dimensions = Some(dimensions);
            }
        }
        Ok(universe)
    }

    /// Construct a universe from a text XYZ file.
    pub fn from_xyz(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::read_xyz(std::fs::File::open(path)?)?)
    }

    /// Construct a universe from a Gromacs GRO file. Coordinates retain the
    /// nanometre units used by GRO; callers can convert with [`crate::units`].
    pub fn from_gro(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::read_gro(std::fs::File::open(path)?)?)
    }

    fn from_coordinate_file(file: CoordinateFile) -> crate::Result<Self> {
        let first = file.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("coordinate file has no frames".to_string())
        })?;
        let mut atoms = Vec::with_capacity(first.n_atoms());
        for index in 0..first.n_atoms() {
            let name = first
                .names
                .get(index)
                .filter(|name| !name.is_empty())
                .map_or("X", String::as_str);
            let mut atom = Atom::new(index, name, first.positions[index]);
            if let Some(resname) = first.residue_names.get(index) {
                atom.resname = resname.clone();
            }
            if let Some(resid) = first.residue_ids.get(index) {
                atom.resid = *resid;
            }
            atom.element = infer_element(name);
            atom.mass = atom
                .element
                .as_deref()
                .and_then(element_mass)
                .unwrap_or(0.0);
            atoms.push(atom);
        }
        let topology = Topology::new(atoms);
        let frames = file
            .frames
            .into_iter()
            .map(|frame| {
                let mut result = Frame::new(frame.positions);
                result.velocities = frame.velocities;
                result.dimensions = frame.dimensions;
                result
            })
            .collect();
        Ok(Self {
            topology,
            trajectory: Trajectory::new(frames),
        })
    }

    /// Write the current topology and all trajectory frames as a PDB file.
    pub fn write_pdb(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let atoms = self
            .topology
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| PdbAtom {
                serial: u32::try_from(index + 1).unwrap_or(u32::MAX),
                name: atom.name.clone(),
                alt_loc: None,
                residue_name: atom.resname.clone(),
                chain_id: atom.chain_id.chars().next(),
                residue_sequence: atom.resid,
                insertion_code: None,
                x: atom.position[0],
                y: atom.position[1],
                z: atom.position[2],
                occupancy: atom.occupancy,
                temperature_factor: atom.temp_factor,
                element: atom.element.clone(),
                charge: None,
                hetatm: false,
            })
            .collect();
        let frames = self
            .trajectory
            .frames
            .iter()
            .map(|frame| frame.positions.clone())
            .collect();
        PdbStructure {
            atoms,
            frames,
            cryst1: None,
        }
        .write_file(path)?;
        Ok(())
    }

    pub fn atoms(&self) -> AtomGroup {
        AtomGroup::new(self.topology.atoms.clone())
    }

    pub fn select_atoms(&self, expression: &str) -> Result<AtomGroup, SelectionError> {
        self.atoms().select_atoms(expression)
    }

    pub fn add_frame(&mut self, mut frame: Frame) -> crate::Result<()> {
        if frame.n_atoms() != self.topology.atoms.len() {
            return Err(crate::Error::InvalidInput(format!(
                "frame contains {} atoms, topology contains {}",
                frame.n_atoms(),
                self.topology.atoms.len()
            )));
        }
        frame.step = self.trajectory.frames.len();
        self.trajectory.frames.push(frame);
        Ok(())
    }

    pub fn n_atoms(&self) -> usize {
        self.topology.atoms.len()
    }

    pub fn n_residues(&self) -> usize {
        self.topology.residues.len()
    }

    pub fn n_segments(&self) -> usize {
        self.topology.segments.len()
    }

    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.trajectory
            .frame(self.trajectory.current.saturating_sub(1))
            .or_else(|| self.trajectory.frames.first())
            .map(|frame| frame.positions.clone())
            .unwrap_or_default()
    }
}

fn infer_element(name: &str) -> Option<String> {
    let letters: String = name.chars().filter(char::is_ascii_alphabetic).collect();
    if letters.is_empty() {
        None
    } else {
        let upper = letters.to_ascii_uppercase();
        let two_letter = matches!(
            upper.as_str(),
            "CL" | "BR" | "NA" | "MG" | "FE" | "ZN" | "CU" | "MN" | "LI" | "SI"
        );
        Some(if two_letter {
            upper.chars().take(2).collect()
        } else {
            upper.chars().take(1).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Universe {
        let mut a = Atom::new(0, "CA", [0.0, 0.0, 0.0]).with_mass(12.0);
        a.resid = 1;
        a.resname = "ALA".into();
        a.element = Some("C".into());
        let mut b = Atom::new(1, "O", [1.0, 0.0, 0.0]).with_mass(16.0);
        b.resid = 2;
        b.resname = "HOH".into();
        b.element = Some("O".into());
        Universe::from_atoms(vec![a, b])
    }

    #[test]
    fn topology_hierarchy_and_mass_center() {
        let universe = sample();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.n_residues(), 2);
        assert_eq!(
            universe.atoms().center_of_mass(),
            Some([0.5714285714285714, 0.0, 0.0])
        );
    }

    #[test]
    fn selection_is_available_from_universe() {
        let universe = sample();
        assert_eq!(universe.select_atoms("name CA").unwrap().len(), 1);
        assert_eq!(universe.select_atoms("resid 2").unwrap().len(), 1);
    }

    #[test]
    fn trajectory_rejects_wrong_atom_count() {
        let mut universe = sample();
        assert!(
            universe
                .add_frame(Frame::new(vec![[0.0, 0.0, 0.0]]))
                .is_err()
        );
    }
}
