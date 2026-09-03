//! Core topology and trajectory objects.

use crate::amber::{InpcrdFile, NamdBinFile, read_inpcrd, read_namdbin};
use crate::coordinates::{CoordinateFile, read_gro, read_xyz};
use crate::dcd::DcdFile;
use crate::formats::Structure;
use crate::gsd::{GsdFile, read_gsd};
use crate::pdb::{PdbAtom, PdbBond, PdbStructure, read_pdb};
use crate::pdbqt::{PdbqtAtom, PdbqtStructure, read_pdbqt};
use crate::psf::{PsfStructure, read_psf};
use crate::selection::{
    AtomLike, Selection, SelectionError, select, select_with_bonds, select_with_bonds_and_groups,
};
use crate::xdr::{TrrFile, XtcFile, read_trr, read_xtc};
use std::collections::BTreeMap;
use std::ops::{Bound, Index, IndexMut, RangeBounds};
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
    /// Optional PDB insertion code distinguishing residues with the same
    /// numeric identifier.
    pub insertion_code: Option<char>,
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
            insertion_code: None,
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
    fn insertion_code(&self) -> Option<char> {
        self.insertion_code
    }
    fn segid(&self) -> &str {
        &self.segid
    }
    fn atom_type(&self) -> Option<&str> {
        self.atom_type.as_deref()
    }
    fn position(&self) -> [f64; 3] {
        self.position
    }
    fn mass(&self) -> Option<f64> {
        Some(self.mass)
    }
    fn charge(&self) -> Option<f64> {
        Some(self.charge)
    }
}

impl From<PdbAtom> for Atom {
    fn from(atom: PdbAtom) -> Self {
        let element = atom.element.clone().or_else(|| infer_element(&atom.name));
        let position = atom.position();
        let chain_id = atom.chain_id.map(|id| id.to_string()).unwrap_or_default();
        let segid = atom
            .segid
            .clone()
            .or_else(|| (!chain_id.is_empty()).then(|| chain_id.clone()))
            .unwrap_or_else(|| "SYSTEM".to_owned());
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
            segid,
            segment_index: 0,
            chain_id,
            insertion_code: atom.insertion_code,
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
        "LI" => Some(6.94),
        "AL" => Some(26.982),
        "SI" => Some(28.085),
        "K" => Some(39.098),
        "CR" => Some(51.996),
        "MN" => Some(54.938),
        "FE" => Some(55.845),
        "CO" => Some(58.933),
        "NI" => Some(58.693),
        "CU" => Some(63.546),
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

fn normalize_range<R>(range: R, length: usize) -> (usize, usize)
where
    R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&value) => value.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&value) => value,
        Bound::Unbounded => length,
    };
    (start, end)
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
        let mut previous_residue_key: Option<(i32, String, String, Option<char>)> = None;
        for atom_index in 0..self.atoms.len() {
            self.atoms[atom_index].index = atom_index;
            let atom = &self.atoms[atom_index];
            let key = (
                atom.resid,
                atom.resname.clone(),
                atom.segid.clone(),
                atom.insertion_code,
            );
            let residue_index = if previous_residue_key.as_ref() == Some(&key) {
                self.residues.len() - 1
            } else {
                self.residues.push(Residue::new(
                    self.residues.len(),
                    atom.resid,
                    atom.resname.clone(),
                    atom.segment_index,
                ));
                previous_residue_key = Some(key);
                self.residues.len() - 1
            };
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

    /// Return the residue at `index`, if it exists.
    pub fn residue(&self, index: usize) -> Option<&Residue> {
        self.residues.get(index)
    }

    /// Return the segment at `index`, if it exists.
    pub fn segment(&self, index: usize) -> Option<&Segment> {
        self.segments.get(index)
    }

    /// Return all atoms belonging to the residue at `index`.
    ///
    /// Atoms are returned in the same order as [`Residue::atom_indices`].
    /// The returned group owns its atoms, so changing its coordinates does
    /// not mutate the topology.
    pub fn residue_atoms(&self, index: usize) -> Option<AtomGroup> {
        let residue = self.residues.get(index)?;
        let atoms = residue
            .atom_indices
            .iter()
            .map(|&atom_index| self.atoms.get(atom_index).cloned())
            .collect::<Option<Vec<_>>>()?;
        Some(AtomGroup::new(atoms))
    }

    /// Return all atoms belonging to the segment at `index`.
    ///
    /// Atoms are grouped by residue order and retain each residue's atom
    /// order.  An invalid hierarchy reference yields `None` rather than a
    /// partially populated group.
    pub fn segment_atoms(&self, index: usize) -> Option<AtomGroup> {
        let segment = self.segments.get(index)?;
        let atom_indices = segment
            .residue_indices
            .iter()
            .map(|&residue_index| self.residues.get(residue_index))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .flat_map(|residue| residue.atom_indices.iter().copied())
            .collect::<Vec<_>>();
        let atoms = atom_indices
            .iter()
            .map(|&atom_index| self.atoms.get(atom_index).cloned())
            .collect::<Option<Vec<_>>>()?;
        Some(AtomGroup::new(atoms))
    }

    /// Alias for [`Topology::residue_atoms`].
    pub fn atoms_for_residue(&self, index: usize) -> Option<AtomGroup> {
        self.residue_atoms(index)
    }

    /// Alias for [`Topology::segment_atoms`].
    pub fn atoms_for_segment(&self, index: usize) -> Option<AtomGroup> {
        self.segment_atoms(index)
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
    /// Additional frame-level data supplied by formats that support named
    /// blocks (for example TNG).
    pub data: BTreeMap<String, Vec<f64>>,
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
            data: BTreeMap::new(),
        }
    }

    /// Number of atom coordinate triplets in this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_atoms()
    }

    /// Return whether this frame contains no atom coordinates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Return the position of the atom at `index`, if it exists.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&[f64; 3]> {
        self.positions.get(index)
    }

    /// Iterate over atom positions without changing the frame metadata.
    pub fn iter(&self) -> std::slice::Iter<'_, [f64; 3]> {
        self.positions.iter()
    }

    /// Mutably iterate over atom positions without changing the frame metadata.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, [f64; 3]> {
        self.positions.iter_mut()
    }

    /// Copy a contiguous atom range, retaining frame-level metadata.
    pub fn slice<R>(&self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = normalize_range(range, self.positions.len());
        let velocities = self.velocities.as_ref().map(|values| {
            if values.len() == self.positions.len() {
                values[start..end].to_vec()
            } else {
                Vec::new()
            }
        });
        let forces = self.forces.as_ref().map(|values| {
            if values.len() == self.positions.len() {
                values[start..end].to_vec()
            } else {
                Vec::new()
            }
        });
        Self {
            positions: self.positions[start..end].to_vec(),
            velocities,
            forces,
            dimensions: self.dimensions,
            time: self.time,
            step: self.step,
            data: self.data.clone(),
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

impl<'a> IntoIterator for &'a Frame {
    type Item = &'a [f64; 3];
    type IntoIter = std::slice::Iter<'a, [f64; 3]>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Frame {
    type Item = &'a mut [f64; 3];
    type IntoIter = std::slice::IterMut<'a, [f64; 3]>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl IntoIterator for Frame {
    type Item = [f64; 3];
    type IntoIter = std::vec::IntoIter<[f64; 3]>;

    fn into_iter(self) -> Self::IntoIter {
        self.positions.into_iter()
    }
}

impl std::ops::Index<usize> for Frame {
    type Output = [f64; 3];

    fn index(&self, index: usize) -> &Self::Output {
        &self.positions[index]
    }
}

impl std::ops::IndexMut<usize> for Frame {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.positions[index]
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

    /// Alias for [`Trajectory::n_frames`] matching collection terminology.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_frames()
    }

    /// Return whether this trajectory contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Iterate over frames without changing the current-frame cursor.
    pub fn iter(&self) -> std::slice::Iter<'_, Frame> {
        self.frames.iter()
    }

    /// Mutably iterate over frames without changing the current-frame cursor.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Frame> {
        self.frames.iter_mut()
    }

    /// Number of atoms represented by each frame, or zero for an empty
    /// trajectory.
    pub fn n_atoms(&self) -> usize {
        self.frames.first().map_or(0, Frame::n_atoms)
    }

    pub fn frame(&self, index: usize) -> Option<&Frame> {
        self.frames.get(index)
    }

    /// Return the frame most recently yielded by [`Self::next`]. Before any
    /// advancement, this is the first frame when one exists.
    pub fn current_frame(&self) -> Option<&Frame> {
        if self.current == 0 {
            self.frames.first()
        } else {
            self.frames.get(self.current - 1)
        }
    }

    /// Return the frame most recently yielded by [`Self::next`] mutably.
    /// Before any advancement, this is the first frame when one exists.
    pub fn current_frame_mut(&mut self) -> Option<&mut Frame> {
        let index = self.current.saturating_sub(1);
        self.frame_mut(index)
    }

    pub fn frame_mut(&mut self, index: usize) -> Option<&mut Frame> {
        self.frames.get_mut(index)
    }

    /// Return a frame by zero-based index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Frame> {
        self.frame(index)
    }

    /// Copy a contiguous range of frames and reset the new cursor.
    #[must_use]
    pub fn slice<R>(&self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = normalize_range(range, self.frames.len());
        Self::new(self.frames[start..end].to_vec())
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

    /// Advance to and return the next frame.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&Frame> {
        self.next_frame()
    }
}

impl<'a> IntoIterator for &'a Trajectory {
    type Item = &'a Frame;
    type IntoIter = std::slice::Iter<'a, Frame>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for Trajectory {
    type Item = Frame;
    type IntoIter = std::vec::IntoIter<Frame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
    }
}

impl std::ops::Index<usize> for Trajectory {
    type Output = Frame;

    fn index(&self, index: usize) -> &Self::Output {
        &self.frames[index]
    }
}

impl std::ops::IndexMut<usize> for Trajectory {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.frames[index]
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

    /// Return per-atom velocities when every atom has velocity data.
    pub fn velocities(&self) -> Option<Vec<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.velocity).collect()
    }

    /// Return per-atom forces when every atom has force data.
    pub fn forces(&self) -> Option<Vec<[f64; 3]>> {
        self.atoms.iter().map(|atom| atom.force).collect()
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

    pub fn center_of_geometry(&self) -> Option<[f64; 3]> {
        if self.atoms.is_empty() {
            return None;
        }
        let positions: Vec<crate::geometry::Vec3> =
            self.positions().into_iter().map(Into::into).collect();
        Some(crate::geometry::center_of_geometry(&positions).to_array())
    }

    pub fn radius_of_gyration(&self) -> Option<f64> {
        crate::analysis::radius_of_gyration(self)
    }

    pub fn translate_in_place(&mut self, offset: [f64; 3]) {
        for atom in &mut self.atoms {
            atom.position[0] += offset[0];
            atom.position[1] += offset[1];
            atom.position[2] += offset[2];
        }
    }

    pub fn bounding_box(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = self.atoms.first()?.position;
        let mut minimum = first;
        let mut maximum = first;
        for atom in &self.atoms[1..] {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(atom.position[axis]);
                maximum[axis] = maximum[axis].max(atom.position[axis]);
            }
        }
        Some((minimum, maximum))
    }

    pub fn select_atoms(&self, expression: &str) -> Result<Self, SelectionError> {
        Ok(Self::new(
            select(&self.atoms, expression)?
                .into_iter()
                .cloned()
                .collect(),
        ))
    }

    /// Select atoms using named groups supplied as `(name, group)` pairs.
    pub fn select_atoms_with_groups(
        &self,
        expression: &str,
        groups: &[(&str, &AtomGroup)],
    ) -> Result<Self, SelectionError> {
        let index_groups: Vec<(&str, Vec<usize>)> = groups
            .iter()
            .map(|(name, group)| (*name, group.atoms.iter().map(|atom| atom.index).collect()))
            .collect();
        let group_slices: Vec<(&str, &[usize])> = index_groups
            .iter()
            .map(|(name, indices)| (*name, indices.as_slice()))
            .collect();
        let selection = Selection::parse(expression)?;
        let selected = if selection.expression_is_global_root() {
            let global_atoms: Vec<Atom> = groups
                .iter()
                .flat_map(|(_, group)| group.atoms.iter().cloned())
                .collect();
            selection
                .apply_with_global_scope(&global_atoms, &[], &group_slices, &global_atoms)?
                .into_iter()
                .cloned()
                .collect()
        } else {
            selection
                .apply_with_bonds_and_groups(&self.atoms, &[], &group_slices)?
                .into_iter()
                .cloned()
                .collect()
        };
        Ok(Self::new(selected))
    }

    pub fn get(&self, index: usize) -> Option<&Atom> {
        self.atoms.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Atom> {
        self.atoms.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Atom> {
        self.atoms.iter_mut()
    }

    pub fn slice<R>(&self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = normalize_range(range, self.atoms.len());
        Self::new(self.atoms[start..end].to_vec())
    }
}

impl Index<usize> for AtomGroup {
    type Output = Atom;

    fn index(&self, index: usize) -> &Self::Output {
        &self.atoms[index]
    }
}

impl IndexMut<usize> for AtomGroup {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.atoms[index]
    }
}

impl<'a> IntoIterator for &'a AtomGroup {
    type Item = &'a Atom;
    type IntoIter = std::slice::Iter<'a, Atom>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut AtomGroup {
    type Item = &'a mut Atom;
    type IntoIter = std::slice::IterMut<'a, Atom>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl IntoIterator for AtomGroup {
    type Item = Atom;
    type IntoIter = std::vec::IntoIter<Atom>;

    fn into_iter(self) -> Self::IntoIter {
        self.atoms.into_iter()
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
        Self::from_pdb_structure(structure)
    }

    /// Construct a universe directly from a PDB document held in memory.
    pub fn from_pdb_str(input: &str) -> crate::Result<Self> {
        Self::from_pdb_structure(PdbStructure::from_str(input)?)
    }

    /// Construct a universe from an extended-PDB file with five-digit
    /// residue identifiers.  The underlying record layout is otherwise the
    /// same as standard PDB, so this delegates to [`Universe::from_pdb`].
    pub fn from_xpdb(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_pdb(path)
    }

    /// Construct a universe from an extended-PDB document held in memory.
    pub fn from_xpdb_str(input: &str) -> crate::Result<Self> {
        Self::from_pdb_str(input)
    }

    /// Construct a universe from a single-frame AutoDock PDBQT file.
    pub fn from_pdbqt(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_pdbqt_structure(read_pdbqt(path)?)
    }

    /// Construct a universe directly from an AutoDock PDBQT document held in
    /// memory.
    pub fn from_pdbqt_str(input: &str) -> crate::Result<Self> {
        Self::from_pdbqt_structure(PdbqtStructure::from_str(input)?)
    }

    fn from_pdbqt_structure(structure: PdbqtStructure) -> crate::Result<Self> {
        let PdbqtStructure {
            atoms: pdbqt_atoms,
            cryst1,
            ..
        } = structure;
        if pdbqt_atoms.is_empty() {
            return Err(crate::Error::InvalidInput(
                "PDBQT coordinate file contains no atoms".to_owned(),
            ));
        }
        let atoms = pdbqt_atoms
            .into_iter()
            .enumerate()
            .map(|(index, source)| atom_from_pdbqt(index, source))
            .collect::<Vec<_>>();
        let mut universe = Self::from_atoms(atoms);
        if let Some(cell) = cryst1 {
            let dimensions = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma];
            for frame in &mut universe.trajectory.frames {
                frame.dimensions = Some(dimensions);
            }
        }
        Ok(universe)
    }

    fn from_pdb_structure(structure: PdbStructure) -> crate::Result<Self> {
        let PdbStructure {
            atoms: pdb_atoms,
            frames: pdb_frames,
            cryst1,
            bonds: pdb_bonds,
        } = structure;
        let serial_to_index: std::collections::HashMap<u32, usize> = pdb_atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom.serial, index))
            .collect();
        // Match MDAnalysis' structure-wide fallback: chain IDs supply
        // segment IDs only when the PDB has no segment field populated at
        // all. Mixed files preserve blank segment IDs as blank values.
        let has_segment_ids = pdb_atoms.iter().any(|atom| atom.segid.is_some());
        let atoms: Vec<Atom> = pdb_atoms
            .into_iter()
            .map(|mut source| {
                if has_segment_ids && source.segid.is_none() {
                    source.segid = Some(String::new());
                }
                Atom::from(source)
            })
            .collect();
        let mut universe = Self::from_atoms(atoms);
        if !pdb_frames.is_empty() {
            universe.trajectory = Trajectory::new(
                pdb_frames
                    .into_iter()
                    .enumerate()
                    .map(|(step, positions)| {
                        let mut frame = Frame::new(positions);
                        frame.step = step;
                        frame
                    })
                    .collect(),
            );
        }
        for bond in pdb_bonds {
            if let (Some(&atom1), Some(&atom2)) = (
                serial_to_index.get(&bond.atom1),
                serial_to_index.get(&bond.atom2),
            ) {
                universe.topology.add_bond(Bond::new(atom1, atom2));
            }
        }
        if let Some(cell) = cryst1 {
            let dimensions = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma];
            for frame in &mut universe.trajectory.frames {
                frame.dimensions = Some(dimensions);
            }
        }
        Ok(universe)
    }

    /// Construct a universe from a text XYZ file.
    pub fn from_xyz(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(read_xyz(path)?)
    }

    /// Construct a universe directly from an XYZ document held in memory.
    pub fn from_xyz_str(input: &str) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::from_xyz_str(input)?)
    }

    /// Construct a universe from a Gromacs GRO file. Coordinates retain the
    /// nanometre units used by GRO; callers can convert with [`crate::units`].
    pub fn from_gro(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(read_gro(path)?)
    }

    /// Construct a universe directly from a GRO document held in memory.
    pub fn from_gro_str(input: &str) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::from_gro_str(input)?)
    }

    /// Construct a universe from a HOOMD GSD topology and trajectory.
    pub fn from_gsd(path: impl AsRef<Path>) -> crate::Result<Self> {
        read_gsd(path)?.to_universe()
    }

    /// Construct a universe from an in-memory GSD document.
    pub fn from_gsd_bytes(bytes: &[u8]) -> crate::Result<Self> {
        GsdFile::from_bytes(bytes)?.to_universe()
    }

    /// Construct a universe from a DCD trajectory without a separate
    /// topology. Atom metadata defaults to the DCD atom count and positions.
    pub fn from_dcd(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dcd_file(DcdFile::read(std::fs::File::open(path)?)?)
    }

    /// Construct a universe from DCD bytes without a separate topology.
    pub fn from_dcd_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_dcd_file(DcdFile::from_bytes(bytes)?)
    }

    /// Construct a universe from a Gromacs XTC trajectory without a separate
    /// topology. Atom metadata defaults to the XTC atom count and positions.
    pub fn from_xtc(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_xtc_file(read_xtc(path)?)
    }

    /// Construct a universe from XTC bytes without a separate topology.
    pub fn from_xtc_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_xtc_file(XtcFile::from_bytes(bytes)?)
    }

    /// Construct a universe from a Gromacs TRR trajectory without a separate
    /// topology. Atom metadata defaults to the TRR atom count and positions.
    pub fn from_trr(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_trr_file(read_trr(path)?)
    }

    /// Construct a universe from TRR bytes without a separate topology.
    pub fn from_trr_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_trr_file(TrrFile::from_bytes(bytes)?)
    }

    /// Construct a universe from a PSF topology without coordinates.
    ///
    /// A zero-filled frame is provided so that coordinate-consuming methods
    /// remain available; use one of the `from_psf_and_*` constructors to
    /// attach an actual trajectory.
    pub fn from_psf(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_psf_structure(read_psf(path)?)
    }

    /// Construct a universe from a PSF topology held in memory.
    pub fn from_psf_str(input: &str) -> crate::Result<Self> {
        Self::from_psf_structure(PsfStructure::from_str(input)?)
    }

    /// Construct a universe from PSF topology and a PDB coordinate file.
    pub fn from_psf_and_pdb(
        psf_path: impl AsRef<Path>,
        pdb_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_pdb_structures(read_psf(psf_path)?, read_pdb(pdb_path)?)
    }

    /// Construct a universe from PSF and PDB documents held in memory.
    pub fn from_psf_and_pdb_str(psf: &str, pdb: &str) -> crate::Result<Self> {
        Self::from_psf_and_pdb_structures(
            PsfStructure::from_str(psf)?,
            PdbStructure::from_str(pdb)?,
        )
    }

    /// Construct a universe from PSF topology and an XYZ trajectory file.
    pub fn from_psf_and_xyz(
        psf_path: impl AsRef<Path>,
        xyz_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_coordinate_file(read_psf(psf_path)?, read_xyz(xyz_path)?)
    }

    /// Construct a universe from PSF and XYZ documents held in memory.
    pub fn from_psf_and_xyz_str(psf: &str, xyz: &str) -> crate::Result<Self> {
        Self::from_psf_and_coordinate_file(
            PsfStructure::from_str(psf)?,
            CoordinateFile::from_xyz_str(xyz)?,
        )
    }

    /// Construct a universe from PSF topology and a GRO trajectory file.
    pub fn from_psf_and_gro(
        psf_path: impl AsRef<Path>,
        gro_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_coordinate_file(read_psf(psf_path)?, read_gro(gro_path)?)
    }

    /// Construct a universe from PSF and GRO documents held in memory.
    pub fn from_psf_and_gro_str(psf: &str, gro: &str) -> crate::Result<Self> {
        Self::from_psf_and_coordinate_file(
            PsfStructure::from_str(psf)?,
            CoordinateFile::from_gro_str(gro)?,
        )
    }

    /// Construct a universe from PSF topology and a DCD trajectory file.
    pub fn from_psf_and_dcd(
        psf_path: impl AsRef<Path>,
        dcd_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_dcd_file(
            read_psf(psf_path)?,
            DcdFile::read(std::fs::File::open(dcd_path)?)?,
        )
    }

    /// Construct a universe from PSF topology and DCD bytes held in memory.
    pub fn from_psf_and_dcd_bytes(psf: &str, dcd: &[u8]) -> crate::Result<Self> {
        Self::from_psf_and_dcd_file(PsfStructure::from_str(psf)?, DcdFile::from_bytes(dcd)?)
    }

    /// Construct a universe from PSF topology and an XTC trajectory file.
    pub fn from_psf_and_xtc(
        psf_path: impl AsRef<Path>,
        xtc_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_xtc_file(read_psf(psf_path)?, read_xtc(xtc_path)?)
    }

    /// Construct a universe from PSF and XTC documents held in memory.
    pub fn from_psf_and_xtc_bytes(psf: &str, xtc: &[u8]) -> crate::Result<Self> {
        Self::from_psf_and_xtc_file(PsfStructure::from_str(psf)?, XtcFile::from_bytes(xtc)?)
    }

    /// Construct a universe from PSF topology and a TRR trajectory file.
    pub fn from_psf_and_trr(
        psf_path: impl AsRef<Path>,
        trr_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_psf_and_trr_file(read_psf(psf_path)?, read_trr(trr_path)?)
    }

    /// Construct a universe from PSF and TRR documents held in memory.
    pub fn from_psf_and_trr_bytes(psf: &str, trr: &[u8]) -> crate::Result<Self> {
        Self::from_psf_and_trr_file(PsfStructure::from_str(psf)?, TrrFile::from_bytes(trr)?)
    }

    /// Construct a universe from PDB topology and a DCD trajectory file.
    pub fn from_pdb_and_dcd(
        pdb_path: impl AsRef<Path>,
        dcd_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_pdb_and_dcd_structures(
            read_pdb(pdb_path)?,
            DcdFile::read(std::fs::File::open(dcd_path)?)?,
        )
    }

    /// Construct a universe from PDB topology and DCD bytes held in memory.
    pub fn from_pdb_and_dcd_bytes(pdb: &str, dcd: &[u8]) -> crate::Result<Self> {
        Self::from_pdb_and_dcd_structures(PdbStructure::from_str(pdb)?, DcdFile::from_bytes(dcd)?)
    }

    /// Construct a universe from PDB topology and an XTC trajectory file.
    pub fn from_pdb_and_xtc(
        pdb_path: impl AsRef<Path>,
        xtc_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_pdb_and_xtc_structures(read_pdb(pdb_path)?, read_xtc(xtc_path)?)
    }

    /// Construct a universe from PDB and XTC documents held in memory.
    pub fn from_pdb_and_xtc_bytes(pdb: &str, xtc: &[u8]) -> crate::Result<Self> {
        Self::from_pdb_and_xtc_structures(PdbStructure::from_str(pdb)?, XtcFile::from_bytes(xtc)?)
    }

    /// Construct a universe from PDB topology and a TRR trajectory file.
    pub fn from_pdb_and_trr(
        pdb_path: impl AsRef<Path>,
        trr_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_pdb_and_trr_structures(read_pdb(pdb_path)?, read_trr(trr_path)?)
    }

    /// Construct a universe from PDB and TRR documents held in memory.
    pub fn from_pdb_and_trr_bytes(pdb: &str, trr: &[u8]) -> crate::Result<Self> {
        Self::from_pdb_and_trr_structures(PdbStructure::from_str(pdb)?, TrrFile::from_bytes(trr)?)
    }

    pub fn from_pqr(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_format_structure(crate::formats::read_pqr(path)?)
    }

    pub fn from_pqr_str(input: &str) -> crate::Result<Self> {
        Self::from_format_structure(Structure::from_pqr_str(input)?)
    }

    pub fn from_mol2(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_format_structure(crate::formats::read_mol2(path)?)
    }

    pub fn from_mol2_str(input: &str) -> crate::Result<Self> {
        Self::from_format_structure(Structure::from_mol2_str(input)?)
    }

    /// Construct a universe from a CHARMM CRD/CARD coordinate file.
    pub fn from_crd(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_format_structure(crate::formats::read_crd(path)?)
    }

    /// Construct a universe from a CRD/CARD document held in memory.
    pub fn from_crd_str(input: &str) -> crate::Result<Self> {
        Self::from_format_structure(Structure::from_crd_str(input)?)
    }

    /// Construct a universe from an Amber restart coordinate file.
    pub fn from_inpcrd(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(read_inpcrd(path)?.coordinates)
    }

    /// Construct a universe from an Amber restart document held in memory.
    pub fn from_inpcrd_str(input: &str) -> crate::Result<Self> {
        Self::from_coordinate_file(InpcrdFile::from_str(input)?.coordinates)
    }

    /// Construct a universe from a NAMD double-precision binary coordinate file.
    pub fn from_namdbin(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(read_namdbin(path)?.coordinates)
    }

    /// Construct a universe from NAMD binary coordinate bytes held in memory.
    pub fn from_namdbin_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_coordinate_file(NamdBinFile::from_bytes(bytes)?.coordinates)
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
                result.step = frame.step;
                result.time = frame.time;
                result
            })
            .collect();
        Ok(Self {
            topology,
            trajectory: Trajectory::new(frames),
        })
    }

    fn from_dcd_file(file: DcdFile) -> crate::Result<Self> {
        let first = file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("DCD file has no coordinate frames".to_owned())
        })?;
        let atoms = first
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| Atom::new(index, "X", *position))
            .collect::<Vec<_>>();
        let topology = Topology::new(atoms);
        let frames = file
            .coordinates
            .frames
            .into_iter()
            .enumerate()
            .map(|(index, coordinate)| {
                let mut frame = Frame::new(coordinate.positions);
                frame.dimensions = coordinate.dimensions;
                let step =
                    i64::from(file.header.istart) + i64::from(file.header.nsavc) * index as i64;
                frame.step = usize::try_from(step).unwrap_or(0);
                frame.time = file.header.delta * step as f64;
                frame
            })
            .collect();
        Ok(Self {
            topology,
            trajectory: Trajectory::new(frames),
        })
    }

    fn from_xtc_file(file: XtcFile) -> crate::Result<Self> {
        Self::from_coordinate_file(file.coordinates)
    }

    fn from_trr_file(file: TrrFile) -> crate::Result<Self> {
        let mut universe = Self::from_coordinate_file(file.coordinates)?;
        for (frame, forces) in universe.trajectory.frames.iter_mut().zip(file.forces) {
            frame.forces = forces;
        }
        Ok(universe)
    }

    fn from_psf_and_dcd_file(psf: PsfStructure, dcd: DcdFile) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        universe.attach_dcd(dcd)?;
        Ok(universe)
    }

    fn from_psf_and_xtc_file(psf: PsfStructure, xtc: XtcFile) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        universe.attach_coordinate_file(xtc.coordinates)?;
        Ok(universe)
    }

    fn from_psf_and_trr_file(psf: PsfStructure, trr: TrrFile) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        universe.attach_trr(trr)?;
        Ok(universe)
    }

    fn from_pdb_and_dcd_structures(pdb: PdbStructure, dcd: DcdFile) -> crate::Result<Self> {
        let mut universe = Self::from_pdb_structure(pdb)?;
        universe.attach_dcd(dcd)?;
        Ok(universe)
    }

    fn from_pdb_and_xtc_structures(pdb: PdbStructure, xtc: XtcFile) -> crate::Result<Self> {
        let mut universe = Self::from_pdb_structure(pdb)?;
        universe.attach_coordinate_file(xtc.coordinates)?;
        Ok(universe)
    }

    fn from_pdb_and_trr_structures(pdb: PdbStructure, trr: TrrFile) -> crate::Result<Self> {
        let mut universe = Self::from_pdb_structure(pdb)?;
        universe.attach_trr(trr)?;
        Ok(universe)
    }

    fn attach_dcd(&mut self, dcd: DcdFile) -> crate::Result<()> {
        if dcd.header.n_atoms != self.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "DCD contains {} atoms, topology contains {}",
                dcd.header.n_atoms,
                self.n_atoms()
            )));
        }
        if dcd.coordinates.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "DCD coordinate file has no frames".to_owned(),
            ));
        }
        let frames = dcd
            .coordinates
            .frames
            .into_iter()
            .enumerate()
            .map(|(index, coordinate)| {
                let step =
                    i64::from(dcd.header.istart) + i64::from(dcd.header.nsavc) * index as i64;
                let mut frame = Frame::new(coordinate.positions);
                frame.dimensions = coordinate.dimensions;
                frame.step = usize::try_from(step).unwrap_or(0);
                frame.time = dcd.header.delta * step as f64;
                frame
            })
            .collect();
        self.trajectory = Trajectory::new(frames);
        Ok(())
    }

    fn attach_coordinate_file(&mut self, coordinates: CoordinateFile) -> crate::Result<()> {
        if coordinates.n_atoms() != self.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "trajectory contains {} atoms, topology contains {}",
                coordinates.n_atoms(),
                self.n_atoms()
            )));
        }
        if coordinates.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "coordinate file has no frames".to_owned(),
            ));
        }
        let frames = coordinates
            .frames
            .into_iter()
            .map(|coordinate| {
                let mut frame = Frame::new(coordinate.positions);
                frame.velocities = coordinate.velocities;
                frame.dimensions = coordinate.dimensions;
                frame.step = coordinate.step;
                frame.time = coordinate.time;
                frame
            })
            .collect();
        self.trajectory = Trajectory::new(frames);
        Ok(())
    }

    fn attach_trr(&mut self, trr: TrrFile) -> crate::Result<()> {
        if trr.n_atoms != self.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "TRR contains {} atoms, topology contains {}",
                trr.n_atoms,
                self.n_atoms()
            )));
        }
        if trr.coordinates.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "TRR coordinate file has no frames".to_owned(),
            ));
        }
        let forces = trr.forces;
        self.attach_coordinate_file(trr.coordinates)?;
        for (frame, force) in self.trajectory.frames.iter_mut().zip(forces) {
            frame.forces = force;
        }
        Ok(())
    }

    fn from_format_structure(structure: Structure) -> crate::Result<Self> {
        let mut atoms = Vec::with_capacity(structure.atoms.len());
        for (index, source) in structure.atoms.iter().enumerate() {
            let mut atom = Atom::new(index, source.name.clone(), source.position());
            atom.resid = source.residue_id;
            atom.resname = source.residue_name.clone();
            atom.segid = source
                .segment_id
                .clone()
                .or_else(|| source.chain_id.clone())
                .unwrap_or_else(|| "SYSTEM".to_string());
            atom.chain_id = source.chain_id.clone().unwrap_or_default();
            atom.charge = source.charge.unwrap_or(0.0);
            atom.element = infer_element(source.atom_type.as_deref().unwrap_or(&source.name));
            atom.atom_type = source.atom_type.clone();
            atom.mass = atom
                .element
                .as_deref()
                .and_then(element_mass)
                .unwrap_or(0.0);
            atoms.push(atom);
        }
        let mut topology = Topology::new(atoms);
        for bond in structure.bonds {
            let atom1 = structure
                .atoms
                .iter()
                .position(|atom| atom.serial == bond.atom1);
            let atom2 = structure
                .atoms
                .iter()
                .position(|atom| atom.serial == bond.atom2);
            if let (Some(atom1), Some(atom2)) = (atom1, atom2) {
                let mut topology_bond = Bond::new(atom1, atom2);
                topology_bond.order = bond.bond_type.parse::<u8>().ok();
                topology.add_bond(topology_bond);
            }
        }
        Ok(Self::new(topology))
    }

    fn from_psf_structure(psf: PsfStructure) -> crate::Result<Self> {
        let mut atoms = Vec::with_capacity(psf.atoms.len());
        let mut serial_to_index = std::collections::HashMap::with_capacity(psf.atoms.len());
        for (position, source) in psf.atoms.iter().enumerate() {
            let serial = if source.index == 0 {
                position + 1
            } else {
                source.index
            };
            serial_to_index.insert(serial, position);
            let mut atom = Atom::new(position, source.name.clone(), [0.0, 0.0, 0.0]);
            atom.atom_type = Some(source.atom_type.clone());
            atom.element = infer_element(&source.atom_type).or_else(|| infer_element(&source.name));
            atom.mass = source.mass;
            atom.charge = source.charge;
            atom.resid = source.resid;
            atom.resname = source.resname.clone();
            atom.segid = source.segid.clone();
            atoms.push(atom);
        }
        let mut topology = Topology::new(atoms);
        for bond in psf.bonds {
            if let (Some(&atom1), Some(&atom2)) = (
                serial_to_index.get(&bond.atom1),
                serial_to_index.get(&bond.atom2),
            ) {
                topology.add_bond(Bond::new(atom1, atom2));
            }
        }
        Ok(Self::new(topology))
    }

    fn from_psf_and_pdb_structures(psf: PsfStructure, pdb: PdbStructure) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        if pdb.atoms.len() != universe.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "PDB contains {} atoms, PSF contains {}",
                pdb.atoms.len(),
                universe.n_atoms()
            )));
        }
        if pdb.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "PDB coordinate file has no frames".to_string(),
            ));
        }
        universe.trajectory = Trajectory::new(
            pdb.frames
                .into_iter()
                .enumerate()
                .map(|(step, positions)| {
                    let mut frame = Frame::new(positions);
                    frame.step = step;
                    frame
                })
                .collect(),
        );
        if let Some(cell) = pdb.cryst1 {
            let dimensions = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma];
            for frame in &mut universe.trajectory.frames {
                frame.dimensions = Some(dimensions);
            }
        }
        Ok(universe)
    }

    fn from_psf_and_coordinate_file(
        psf: PsfStructure,
        coordinate_file: CoordinateFile,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        let expected = universe.n_atoms();
        if coordinate_file.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "coordinate file has no frames".to_string(),
            ));
        }
        let mut frames = Vec::with_capacity(coordinate_file.frames.len());
        for (step, source) in coordinate_file.frames.into_iter().enumerate() {
            if source.positions.len() != expected {
                return Err(crate::Error::InvalidInput(format!(
                    "coordinate frame contains {} atoms, PSF contains {}",
                    source.positions.len(),
                    expected
                )));
            }
            let mut frame = Frame::new(source.positions);
            frame.velocities = source.velocities;
            frame.dimensions = source.dimensions;
            frame.step = step;
            frames.push(frame);
        }
        universe.trajectory = Trajectory::new(frames);
        Ok(universe)
    }

    /// Write the current topology and all trajectory frames as a PDB file.
    pub fn write_pdb(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let first_positions = self
            .trajectory
            .frames
            .first()
            .map(|frame| frame.positions.clone())
            .unwrap_or_else(|| {
                self.topology
                    .atoms
                    .iter()
                    .map(|atom| atom.position)
                    .collect()
            });
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
                // `SYSTEM` is the crate's synthetic fallback and cannot be
                // represented losslessly in PDB's four-column segment field.
                segid: (!atom.segid.is_empty() && atom.segid != "SYSTEM")
                    .then(|| atom.segid.clone()),
                residue_sequence: atom.resid,
                insertion_code: atom.insertion_code,
                x: first_positions.get(index).unwrap_or(&atom.position)[0],
                y: first_positions.get(index).unwrap_or(&atom.position)[1],
                z: first_positions.get(index).unwrap_or(&atom.position)[2],
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
        let cryst1 = self
            .trajectory
            .frames
            .first()
            .and_then(|frame| frame.dimensions)
            .map(|dimensions| crate::pdb::PdbCryst1 {
                a: dimensions[0],
                b: dimensions[1],
                c: dimensions[2],
                alpha: dimensions[3],
                beta: dimensions[4],
                gamma: dimensions[5],
                space_group: "P 1".to_string(),
                z: None,
            });
        PdbStructure {
            atoms,
            frames,
            cryst1,
            bonds: self
                .topology
                .bonds
                .iter()
                .map(|bond| PdbBond::new(bond.atom1 as u32 + 1, bond.atom2 as u32 + 1))
                .collect(),
        }
        .write_file(path)?;
        Ok(())
    }

    /// Write the current topology and coordinates as a single-frame PDBQT
    /// document.
    pub fn write_pdbqt(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let current_positions = self
            .current_frame()
            .map(|frame| frame.positions.clone())
            .unwrap_or_else(|| {
                self.topology
                    .atoms
                    .iter()
                    .map(|atom| atom.position)
                    .collect()
            });
        let atoms = self
            .topology
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| {
                let position = current_positions
                    .get(index)
                    .copied()
                    .unwrap_or(atom.position);
                let atom_type = atom
                    .atom_type
                    .clone()
                    .or_else(|| atom.element.clone())
                    .unwrap_or_default();
                PdbqtAtom {
                    serial: u32::try_from(index + 1).unwrap_or(u32::MAX),
                    name: atom.name.clone(),
                    alt_loc: None,
                    residue_name: atom.resname.clone(),
                    chain_id: atom.chain_id.chars().next(),
                    residue_sequence: atom.resid,
                    insertion_code: atom.insertion_code,
                    x: position[0],
                    y: position[1],
                    z: position[2],
                    occupancy: atom.occupancy.unwrap_or(1.0),
                    temperature_factor: atom.temp_factor.unwrap_or(0.0),
                    charge: atom.charge,
                    atom_type,
                    hetatm: false,
                }
            })
            .collect();
        let cryst1 = self
            .current_frame()
            .and_then(|frame| frame.dimensions)
            .map(|dimensions| crate::pdb::PdbCryst1 {
                a: dimensions[0],
                b: dimensions[1],
                c: dimensions[2],
                alpha: dimensions[3],
                beta: dimensions[4],
                gamma: dimensions[5],
                space_group: "P 1".to_owned(),
                z: None,
            });
        PdbqtStructure {
            atoms,
            cryst1,
            title: String::new(),
        }
        .write_file(path)?;
        Ok(())
    }

    pub fn atoms(&self) -> AtomGroup {
        let frame = self.current_frame();
        let mut atoms = self.topology.atoms.clone();
        for (index, atom) in atoms.iter_mut().enumerate() {
            if let Some(frame) = frame {
                if let Some(position) = frame.positions.get(index) {
                    atom.position = *position;
                }
                atom.velocity = frame
                    .velocities
                    .as_ref()
                    .and_then(|values| values.get(index).copied());
                atom.force = frame
                    .forces
                    .as_ref()
                    .and_then(|values| values.get(index).copied());
            } else {
                atom.velocity = None;
                atom.force = None;
            }
        }
        AtomGroup::new(atoms)
    }

    /// Return the residue at `index` as an atom group using the current
    /// trajectory frame's coordinates.
    pub fn residue_atoms(&self, index: usize) -> Option<AtomGroup> {
        let indices = self.topology.residue(index)?.atom_indices.clone();
        let atoms = self.atoms();
        let selected = indices
            .iter()
            .map(|&atom_index| atoms.atoms.get(atom_index).cloned())
            .collect::<Option<Vec<_>>>()?;
        Some(AtomGroup::new(selected))
    }

    /// Return the segment at `index` as an atom group using the current
    /// trajectory frame's coordinates.
    pub fn segment_atoms(&self, index: usize) -> Option<AtomGroup> {
        let residue_indices = self.topology.segment(index)?.residue_indices.clone();
        let atom_indices = residue_indices
            .iter()
            .map(|&residue_index| self.topology.residue(residue_index))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .flat_map(|residue| residue.atom_indices.iter().copied())
            .collect::<Vec<_>>();
        let atoms = self.atoms();
        let selected = atom_indices
            .iter()
            .map(|&atom_index| atoms.atoms.get(atom_index).cloned())
            .collect::<Option<Vec<_>>>()?;
        Some(AtomGroup::new(selected))
    }

    /// Alias for [`Universe::residue_atoms`].
    pub fn atoms_for_residue(&self, index: usize) -> Option<AtomGroup> {
        self.residue_atoms(index)
    }

    /// Alias for [`Universe::segment_atoms`].
    pub fn atoms_for_segment(&self, index: usize) -> Option<AtomGroup> {
        self.segment_atoms(index)
    }

    /// Return the residue metadata at `index`, if it exists.
    pub fn residue(&self, index: usize) -> Option<&Residue> {
        self.topology.residue(index)
    }

    /// Return the segment metadata at `index`, if it exists.
    pub fn segment(&self, index: usize) -> Option<&Segment> {
        self.topology.segment(index)
    }

    /// Return all residues as atom groups in topology order.
    ///
    /// Each group uses the coordinates and per-atom frame data from the
    /// currently selected trajectory frame, matching [`Universe::atoms`].
    pub fn residues(&self) -> Vec<AtomGroup> {
        (0..self.n_residues())
            .filter_map(|index| self.residue_atoms(index))
            .collect()
    }

    /// Return all segments as atom groups in topology order.
    ///
    /// Each group uses the coordinates and per-atom frame data from the
    /// currently selected trajectory frame, matching [`Universe::atoms`].
    pub fn segments(&self) -> Vec<AtomGroup> {
        (0..self.n_segments())
            .filter_map(|index| self.segment_atoms(index))
            .collect()
    }

    pub fn select_atoms(&self, expression: &str) -> Result<AtomGroup, SelectionError> {
        let atoms = self.atoms();
        let bonds: Vec<(usize, usize)> = self
            .topology
            .bonds
            .iter()
            .map(|bond| (bond.atom1, bond.atom2))
            .collect();
        Ok(AtomGroup::new(
            select_with_bonds(&atoms.atoms, expression, &bonds)?
                .into_iter()
                .cloned()
                .collect(),
        ))
    }

    /// Select atoms using named groups, retaining the Universe topology and
    /// evaluating `global` modifiers against the full Universe atom set.
    pub fn select_atoms_with_groups(
        &self,
        expression: &str,
        groups: &[(&str, &AtomGroup)],
    ) -> Result<AtomGroup, SelectionError> {
        let atoms = self.atoms();
        let bonds: Vec<(usize, usize)> = self
            .topology
            .bonds
            .iter()
            .map(|bond| (bond.atom1, bond.atom2))
            .collect();
        let index_groups: Vec<(&str, Vec<usize>)> = groups
            .iter()
            .map(|(name, group)| (*name, group.atoms.iter().map(|atom| atom.index).collect()))
            .collect();
        let group_slices: Vec<(&str, &[usize])> = index_groups
            .iter()
            .map(|(name, indices)| (*name, indices.as_slice()))
            .collect();
        let selected = select_with_bonds_and_groups(
            &atoms.atoms,
            expression,
            &bonds,
            &group_slices,
            Some(&atoms.atoms),
        )?;
        Ok(AtomGroup::new(selected.into_iter().cloned().collect()))
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

    pub fn n_frames(&self) -> usize {
        self.trajectory.n_frames()
    }

    pub fn current_frame(&self) -> Option<&Frame> {
        self.trajectory.current_frame()
    }

    /// Return the currently selected trajectory frame mutably.
    pub fn current_frame_mut(&mut self) -> Option<&mut Frame> {
        self.trajectory.current_frame_mut()
    }

    /// Reset trajectory iteration to its first frame.
    pub fn rewind(&mut self) {
        self.trajectory.rewind();
    }

    /// Return the next trajectory frame.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&Frame> {
        self.trajectory.next_frame()
    }

    /// Alias for [`Universe::next`].
    pub fn next_frame(&mut self) -> Option<&Frame> {
        self.next()
    }

    pub fn set_frame(&mut self, index: usize) -> crate::Result<()> {
        if index >= self.trajectory.frames.len() {
            return Err(crate::Error::InvalidInput(format!(
                "frame index {index} is out of bounds for {} frames",
                self.trajectory.frames.len()
            )));
        }
        self.trajectory.current = index + 1;
        Ok(())
    }

    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.current_frame()
            .map(|frame| frame.positions.clone())
            .unwrap_or_default()
    }

    /// Return the currently selected coordinates for in-place modification.
    pub fn positions_mut(&mut self) -> Option<&mut [[f64; 3]]> {
        self.current_frame_mut()
            .map(|frame| frame.positions.as_mut_slice())
    }
}

fn infer_element(name: &str) -> Option<String> {
    let letters = name
        .split(|character: char| !character.is_ascii_alphabetic())
        .find(|part| !part.is_empty())
        .unwrap_or_default()
        .to_string();
    if letters.is_empty() {
        None
    } else {
        let upper = letters.to_ascii_uppercase();
        let two_letter = matches!(
            upper.as_str(),
            "CL" | "BR"
                | "NA"
                | "MG"
                | "FE"
                | "ZN"
                | "CU"
                | "MN"
                | "LI"
                | "SI"
                | "CR"
                | "CO"
                | "NI"
                | "AL"
        );
        Some(if two_letter {
            upper.chars().take(2).collect()
        } else {
            upper.chars().take(1).collect()
        })
    }
}

fn atom_from_pdbqt(index: usize, source: PdbqtAtom) -> Atom {
    let position = source.position();
    let chain_id = source
        .chain_id
        .map_or_else(String::new, |value| value.to_string());
    let element = infer_element(&source.atom_type).or_else(|| infer_element(&source.name));
    // PDBQT `atom_type` values are AutoDock types, not necessarily element
    // symbols (for example, `HD` and `OA`).  MDAnalysis guesses masses from
    // the raw type string, so only exact element-like types receive a mass.
    let mass = if source.atom_type.trim().is_empty() {
        element.as_deref().and_then(element_mass).unwrap_or(0.0)
    } else {
        element_mass(&source.atom_type).unwrap_or(0.0)
    };
    Atom {
        index,
        name: source.name,
        atom_type: (!source.atom_type.is_empty()).then_some(source.atom_type),
        element,
        mass,
        charge: source.charge,
        resid: source.residue_sequence,
        residue_index: 0,
        resname: source.residue_name,
        segid: if chain_id.is_empty() {
            "SYSTEM".to_owned()
        } else {
            chain_id.clone()
        },
        segment_index: 0,
        chain_id,
        insertion_code: source.insertion_code,
        position,
        velocity: None,
        force: None,
        temp_factor: Some(source.temperature_factor),
        occupancy: Some(source.occupancy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_groups::TopologyGroupExt;

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
    fn repeated_noncontiguous_residue_keys_remain_distinct() {
        let mut first = Atom::new(0, "CA", [0.0, 0.0, 0.0]);
        first.resid = 1;
        first.resname = "ALA".to_owned();
        first.segid = "S".to_owned();
        let mut middle = Atom::new(1, "CA", [1.0, 0.0, 0.0]);
        middle.resid = 2;
        middle.resname = "GLY".to_owned();
        middle.segid = "S".to_owned();
        let mut repeated = Atom::new(2, "CA", [2.0, 0.0, 0.0]);
        repeated.resid = 1;
        repeated.resname = "ALA".to_owned();
        repeated.segid = "S".to_owned();

        let topology = Topology::new(vec![first, middle, repeated]);
        assert_eq!(topology.residues.len(), 3);
        assert_eq!(topology.residue(0).unwrap().atom_indices, vec![0]);
        assert_eq!(topology.residue(1).unwrap().atom_indices, vec![1]);
        assert_eq!(topology.residue(2).unwrap().atom_indices, vec![2]);
    }

    #[test]
    fn hierarchy_accessors_return_indexed_atoms_in_order() {
        let mut first = Atom::new(0, "N", [0.0, 0.0, 0.0]);
        first.resid = 1;
        first.resname = "ALA".into();
        first.segid = "A".into();
        let mut second = Atom::new(1, "CA", [1.0, 0.0, 0.0]);
        second.resid = 1;
        second.resname = "ALA".into();
        second.segid = "A".into();
        let mut third = Atom::new(2, "C", [2.0, 0.0, 0.0]);
        third.resid = 2;
        third.resname = "GLY".into();
        third.segid = "A".into();
        let mut fourth = Atom::new(3, "O", [3.0, 0.0, 0.0]);
        fourth.resid = 3;
        fourth.resname = "HOH".into();
        fourth.segid = "B".into();
        let mut universe = Universe::from_atoms(vec![first, second, third, fourth]);

        assert_eq!(universe.topology.residue(0).unwrap().resid, 1);
        assert_eq!(universe.topology.segment(0).unwrap().id, "A");
        assert_eq!(
            universe.topology.residue_atoms(0).unwrap().atom_names(),
            vec!["N", "CA"]
        );
        assert_eq!(
            universe.topology.segment_atoms(0).unwrap().atom_names(),
            vec!["N", "CA", "C"]
        );
        assert_eq!(
            universe.topology.atoms_for_segment(1).unwrap().atom_names(),
            vec!["O"]
        );
        assert!(universe.topology.residue_atoms(99).is_none());
        assert!(universe.topology.segment_atoms(99).is_none());

        universe
            .add_frame(Frame::new(vec![
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ]))
            .unwrap();
        universe.set_frame(1).unwrap();
        assert_eq!(
            universe.residue_atoms(0).unwrap().positions(),
            vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
        );
        assert_eq!(
            universe.atoms_for_segment(0).unwrap().positions(),
            vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn universe_hierarchy_collections_follow_topology_and_current_frame() {
        let mut first = Atom::new(0, "N", [0.0, 0.0, 0.0]);
        first.resid = 1;
        first.resname = "ALA".into();
        first.segid = "A".into();
        let mut second = Atom::new(1, "CA", [1.0, 0.0, 0.0]);
        second.resid = 1;
        second.resname = "ALA".into();
        second.segid = "A".into();
        let mut third = Atom::new(2, "O", [2.0, 0.0, 0.0]);
        third.resid = 2;
        third.resname = "HOH".into();
        third.segid = "B".into();
        let mut universe = Universe::from_atoms(vec![first, second, third]);

        assert_eq!(universe.residue(0).unwrap().name, "ALA");
        assert_eq!(universe.segment(0).unwrap().id, "A");
        assert!(universe.residue(99).is_none());
        assert!(universe.segment(99).is_none());

        let residues = universe.residues();
        assert_eq!(residues.len(), 2);
        assert_eq!(residues[0].atom_names(), vec!["N", "CA"]);
        assert_eq!(residues[1].atom_names(), vec!["O"]);
        let segments = universe.segments();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].atom_names(), vec!["N", "CA"]);
        assert_eq!(segments[1].atom_names(), vec!["O"]);

        universe
            .add_frame(Frame::new(vec![
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]))
            .unwrap();
        universe.set_frame(1).unwrap();
        assert_eq!(
            universe.residues()[0].positions(),
            vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
        );
        assert_eq!(universe.segments()[1].positions(), vec![[12.0, 0.0, 0.0]]);
    }

    #[test]
    fn selection_is_available_from_universe() {
        let universe = sample();
        assert_eq!(universe.select_atoms("name CA").unwrap().len(), 1);
        assert_eq!(universe.select_atoms("resid 2").unwrap().len(), 1);
    }

    #[test]
    fn xpdb_constructor_preserves_five_digit_residues() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/5digitResid.pdb");
        let universe = Universe::from_xpdb(path).expect("valid XPDB fixture");
        assert_eq!(universe.n_atoms(), 5);
        assert_eq!(universe.n_residues(), 5);
        assert_eq!(universe.topology.atoms[4].resid, 10_000);
        assert_eq!(universe.topology.atoms[4].element.as_deref(), Some("O"));
    }

    #[test]
    fn bonded_selection_uses_universe_bonds() {
        let mut universe = sample();
        universe.topology.add_bond(Bond::new(0, 1));
        let selected = universe.select_atoms("bonded name O").unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "CA");
        assert_eq!(
            universe
                .select_atoms("name CA and bonded resname HOH")
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn named_group_selection_and_global_scope_work() {
        let universe = sample();
        let ca = universe.select_atoms("name CA").unwrap();
        assert_eq!(
            universe
                .select_atoms_with_groups("group backbone", &[("backbone", &ca)])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            universe.select_atoms_with_groups("group missing", &[]),
            Err(SelectionError::UnknownGroup("missing".to_owned()))
        );

        let oxygen = universe.select_atoms("name O").unwrap();
        let nearby = universe
            .select_atoms_with_groups("around 2 global group backbone", &[("backbone", &ca)])
            .unwrap();
        assert_eq!(nearby.len(), 1);
        assert_eq!(nearby[0].name, "O");

        let outside = ca
            .select_atoms_with_groups("global group solvent", &[("solvent", &oxygen)])
            .unwrap();
        assert_eq!(outside.len(), 1);
        assert_eq!(outside[0].name, "O");
    }

    #[test]
    fn pqr_and_mol2_constructors_build_topology() {
        let pqr = "ATOM      1  C1  LIG A   1       0.0  0.0  0.0  0.0  1.7\n";
        let universe = Universe::from_pqr_str(pqr).unwrap();
        assert_eq!(universe.n_atoms(), 1);
        assert_eq!(universe.topology.atoms[0].charge, 0.0);
        let mol2 = "@<TRIPOS>MOLECULE\nwater\n2 1 0 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n      1 O 0.0 0.0 0.0 O.2 1 HOH -0.8\n      2 H 1.0 0.0 0.0 H 1 HOH 0.4\n@<TRIPOS>BOND\n     1 1 2 1\n";
        let universe = Universe::from_mol2_str(mol2).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.topology.bonds.len(), 1);
        assert_eq!(universe.topology.bonds[0].partner(0), Some(1));
        assert_eq!(universe.topology.bonds[0].order, Some(1));
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("O.2"));
    }

    #[test]
    fn pdbqt_constructor_preserves_autodock_attributes_and_unit_cell() {
        let pdbqt = concat!(
            "CRYST1   10.000   20.000   30.000  90.00  90.00 120.00 P 1          1\n",
            "ATOM     42  OA  LIG A   7       1.000   2.000   3.000  1.00 10.00    -0.500 OA\n",
            "END\n",
        );
        let universe = Universe::from_pdbqt_str(pdbqt).expect("valid PDBQT");
        assert_eq!(universe.n_atoms(), 1);
        assert_eq!(universe.topology.atoms[0].index, 0);
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("OA"));
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("O"));
        assert_eq!(universe.topology.atoms[0].charge, -0.5);
        assert_eq!(universe.topology.atoms[0].segid, "A");
        assert_eq!(
            universe.trajectory.frames[0].dimensions,
            Some([10.0, 20.0, 30.0, 90.0, 90.0, 120.0])
        );
    }

    #[test]
    fn pdbqt_writer_round_trips_universe_coordinates_and_charge() {
        let input = concat!(
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00    -0.300 N\n",
            "END\n",
        );
        let universe = Universe::from_pdbqt_str(input).expect("valid PDBQT");
        let path = std::env::temp_dir().join(format!(
            "mdanalysis-rs-pdbqt-{}-{}.pdbqt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        universe.write_pdbqt(&path).expect("write PDBQT");
        let reparsed = Universe::from_pdbqt(&path).expect("read PDBQT");
        let _ = std::fs::remove_file(path);
        assert_eq!(reparsed.positions(), universe.positions());
        assert_eq!(reparsed.topology.atoms[0].charge, -0.3);
    }

    #[test]
    fn pdbqt_writer_uses_the_selected_trajectory_frame() {
        let mut universe = Universe::from_atoms(vec![Atom::new(0, "C", [1.0, 2.0, 3.0])]);
        let mut second_frame = Frame::new(vec![[4.0, 5.0, 6.0]]);
        second_frame.dimensions = Some([10.0, 11.0, 12.0, 90.0, 90.0, 90.0]);
        universe
            .add_frame(second_frame)
            .expect("matching frame atom count");
        universe.set_frame(1).expect("second frame exists");
        universe.positions_mut().expect("selected frame")[0] = [7.0, 8.0, 9.0];

        let path = std::env::temp_dir().join(format!(
            "mdanalysis-rs-pdbqt-frame-{}-{}.pdbqt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        universe.write_pdbqt(&path).expect("write PDBQT");
        let reparsed = Universe::from_pdbqt(&path).expect("read PDBQT");
        let _ = std::fs::remove_file(path);

        assert_eq!(reparsed.positions(), vec![[7.0, 8.0, 9.0]]);
        assert_eq!(
            reparsed.current_frame().unwrap().dimensions,
            Some([10.0, 11.0, 12.0, 90.0, 90.0, 90.0])
        );
    }

    #[test]
    fn pdbqt_mass_guessing_uses_autodock_type_tokens() {
        let path = Path::new("../mdanalysis/testsuite/MDAnalysisTests/data/pdbqt_inputpdbqt.pdbqt");
        let universe = Universe::from_pdbqt(path).expect("PDBQT fixture should parse");
        let masses = universe.atoms().masses();
        assert_eq!(
            &masses[..7],
            &[14.007, 0.0, 0.0, 12.011, 12.011, 0.0, 12.011]
        );
    }

    #[test]
    fn psf_constructors_preserve_topology_and_attach_pdb_frames() {
        let psf = concat!(
            "PSF\n\n",
            "       1 !NTITLE\n",
            "* test\n\n",
            "       2 !NATOM\n",
            "       1 SEG      7 ALA      N        NH1         -0.300000      14.007000        0\n",
            "       2 SEG      7 ALA      CA       CT1          0.100000      12.011000        0\n",
            "\n       1 !NBOND: bonds\n",
            "       1       2\n",
        );
        let pdb = concat!(
            "MODEL        1\n",
            "ATOM      1  N   ALA A   7       1.000   2.000   3.000  1.00 20.00           N  \n",
            "ATOM      2  CA  ALA A   7       2.000   2.000   3.000  1.00 20.00           C  \n",
            "ENDMDL\n",
            "MODEL        2\n",
            "ATOM      1  N   ALA A   7       4.000   5.000   6.000  1.00 20.00           N  \n",
            "ATOM      2  CA  ALA A   7       5.000   5.000   6.000  1.00 20.00           C  \n",
            "ENDMDL\n",
        );
        let mut universe = Universe::from_psf_and_pdb_str(psf, pdb).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.n_residues(), 1);
        assert_eq!(universe.n_segments(), 1);
        assert_eq!(universe.n_frames(), 2);
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("NH1"));
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("N"));
        assert_eq!(universe.topology.atoms[0].mass, 14.007);
        assert_eq!(universe.topology.atoms[0].charge, -0.3);
        assert_eq!(universe.topology.bonds, vec![Bond::new(0, 1)]);
        assert_eq!(universe.positions()[0], [1.0, 2.0, 3.0]);
        universe.set_frame(1).unwrap();
        assert_eq!(universe.positions()[0], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn psf_xyz_constructor_rejects_atom_count_mismatch() {
        let psf = concat!(
            "PSF\n\n",
            "       1 !NATOM\n",
            "       1 SEG      1 ALA      N        NH1          0.000000      14.007000        0\n",
        );
        let xyz = "2\nframe\nN 0 0 0\nH 1 0 0\n";
        let error = Universe::from_psf_and_xyz_str(psf, xyz).unwrap_err();
        assert!(
            matches!(error, crate::Error::InvalidInput(message) if message.contains("PSF contains 1"))
        );
    }

    #[test]
    fn psf_dcd_constructor_attaches_frames_and_times() {
        let psf = concat!(
            "PSF\n\n",
            "       2 !NATOM\n",
            "       1 SEG      1 ALA      N        NH1          0.000000      14.007000        0\n",
            "       2 SEG      1 ALA      CA       CT1          0.000000      12.011000        0\n",
        );
        let coordinates = CoordinateFile::new(vec![
            crate::coordinates::CoordinateFrame::new(vec![[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]),
            crate::coordinates::CoordinateFrame::new(vec![[4.0, 5.0, 6.0], [5.0, 5.0, 6.0]]),
        ]);
        let dcd = crate::dcd::DcdFile {
            header: crate::dcd::DcdHeader {
                n_frames: 2,
                n_atoms: 2,
                istart: 3,
                nsavc: 2,
                delta: 0.5,
                ..crate::dcd::DcdHeader::default()
            },
            coordinates,
        };
        let bytes = dcd.to_bytes().unwrap();
        let universe = Universe::from_psf_and_dcd_bytes(psf, &bytes).unwrap();
        assert_eq!(universe.n_frames(), 2);
        assert_eq!(universe.trajectory.frames[0].step, 3);
        assert_eq!(universe.trajectory.frames[1].step, 5);
        assert!((universe.trajectory.frames[1].time - 2.5).abs() < 1.0e-6);
        assert_eq!(universe.topology.bonds.len(), 0);
    }

    #[test]
    fn pdb_metal_elements_receive_standard_masses() {
        let pdb = concat!(
            "HETATM    1 CU    CU A   1       0.000   0.000   0.000  1.00  0.00          Cu  \n",
            "END\n",
        );
        let universe = Universe::from_pdb_str(pdb).unwrap();
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("Cu"));
        assert!((universe.topology.atoms[0].mass - 63.546).abs() < 1e-6);
    }

    #[test]
    fn pdb_insertion_codes_define_distinct_residues_and_round_trip() {
        let pdb = concat!(
            "ATOM      1  CA  GLY A   1      0.000   0.000   0.000  1.00  0.00           C  \n",
            "ATOM      2  CA  GLY A   1A     1.000   0.000   0.000  1.00  0.00           C  \n",
            "ATOM      3  CA  GLY A   1B     2.000   0.000   0.000  1.00  0.00           C  \n",
            "END\n",
        );
        let universe = Universe::from_pdb_str(pdb).unwrap();
        assert_eq!(universe.n_residues(), 3);
        assert_eq!(universe.topology.atoms[1].insertion_code, Some('A'));
        assert_eq!(
            universe
                .select_atoms("same residue as index 1")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            universe
                .select_atoms("same resid as index 1")
                .unwrap()
                .len(),
            3
        );
        let path = std::env::temp_dir().join(format!(
            "mdanalysis-rs-pdb-icode-{}-{}.pdb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        universe.write_pdb(&path).unwrap();
        let reparsed = Universe::from_pdb(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(reparsed.topology.atoms[2].insertion_code, Some('B'));
        assert_eq!(reparsed.n_residues(), 3);
    }

    #[test]
    fn pdb_segment_ids_are_preserved_for_selection_and_round_trip() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/adk_open.pdb");
        let universe = Universe::from_pdb(&path).expect("valid PDB fixture");
        assert_eq!(universe.topology.atoms[0].segid, "4AKE");
        assert_eq!(universe.topology.atoms[0].chain_id, "");
        assert_eq!(
            universe.select_atoms("segid 4AKE").unwrap().len(),
            universe.n_atoms()
        );

        let output = std::env::temp_dir().join(format!(
            "mdanalysis-rs-pdb-segid-{}-{}.pdb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        universe.write_pdb(&output).expect("write PDB");
        let reparsed = Universe::from_pdb(&output).expect("read written PDB");
        let _ = std::fs::remove_file(output);
        assert_eq!(reparsed.topology.atoms[0].segid, "4AKE");
        assert_eq!(
            reparsed.select_atoms("segid 4AKE").unwrap().len(),
            reparsed.n_atoms()
        );
    }

    #[test]
    fn pdb_writer_keeps_synthetic_system_segment_stable() {
        let universe = Universe::from_atoms(vec![Atom::new(0, "C", [1.0, 2.0, 3.0])]);
        let output = std::env::temp_dir().join(format!(
            "mdanalysis-rs-pdb-system-{}-{}.pdb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        universe.write_pdb(&output).expect("write PDB");
        let reparsed = Universe::from_pdb(&output).expect("read written PDB");
        let _ = std::fs::remove_file(output);
        assert_eq!(reparsed.topology.atoms[0].segid, "SYSTEM");
    }

    #[test]
    fn pdb_mixed_segment_fields_keep_blank_segments_blank() {
        let first = PdbAtom {
            serial: 1,
            name: "CA".to_owned(),
            alt_loc: None,
            residue_name: "ALA".to_owned(),
            chain_id: Some('A'),
            segid: Some("PROT".to_owned()),
            residue_sequence: 1,
            insertion_code: None,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            occupancy: None,
            temperature_factor: None,
            element: Some("C".to_owned()),
            charge: None,
            hetatm: false,
        };
        let mut second = first.clone();
        second.serial = 2;
        second.chain_id = Some('B');
        second.segid = None;
        second.x = 2.0;
        let input = PdbStructure {
            atoms: vec![first, second],
            frames: vec![vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            ..PdbStructure::default()
        }
        .to_pdb_string()
        .unwrap();

        let universe = Universe::from_pdb_str(&input).unwrap();
        assert_eq!(universe.topology.atoms[0].segid, "PROT");
        assert_eq!(universe.topology.atoms[1].segid, "");
        assert_eq!(universe.n_segments(), 2);
    }

    #[test]
    fn mol2_two_letter_elements_are_inferred_from_atom_types() {
        let mol2 = concat!(
            "@<TRIPOS>MOLECULE\n",
            "metals\n",
            "2 0 0 0 0\n",
            "SMALL\nUSER_CHARGES\n\n",
            "@<TRIPOS>ATOM\n",
            "1 Cr1 0 0 0 Cr.th 1 MET 0\n",
            "2 Co1 1 0 0 Co.oh 1 MET 0\n",
        );
        let universe = Universe::from_mol2_str(mol2).unwrap();
        assert_eq!(
            universe
                .topology
                .atoms
                .iter()
                .map(|atom| atom.element.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("CR"), Some("CO")]
        );
        assert!((universe.topology.atoms[0].mass - 51.996).abs() < 1e-6);
        assert!((universe.topology.atoms[1].mass - 58.933).abs() < 1e-6);
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

    #[test]
    fn amber_and_namdbin_universe_constructors_attach_coordinates() {
        let inpcrd = "title\n    1  2.5\n   1.0000000   2.0000000   3.0000000\n";
        let universe = Universe::from_inpcrd_str(inpcrd).unwrap();
        assert_eq!(universe.n_atoms(), 1);
        assert_eq!(universe.positions(), vec![[1.0, 2.0, 3.0]]);
        assert!((universe.current_frame().unwrap().time - 2.5).abs() < 1.0e-12);
        let namd = crate::amber::NamdBinFile {
            coordinates: crate::coordinates::CoordinateFile::new(vec![
                crate::coordinates::CoordinateFrame::new(vec![[4.0, 5.0, 6.0]]),
            ]),
        };
        let universe = Universe::from_namdbin_bytes(&namd.to_bytes().unwrap()).unwrap();
        assert_eq!(universe.positions(), vec![[4.0, 5.0, 6.0]]);
    }

    #[test]
    fn xdr_universe_constructors_attach_metadata_and_trr_vectors() {
        let mut coordinate =
            crate::coordinates::CoordinateFrame::new(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        coordinate.step = 17;
        coordinate.time = 2.5;
        coordinate.dimensions = Some([2.0, 3.0, 4.0, 90.0, 90.0, 90.0]);
        let coordinate_file = crate::coordinates::CoordinateFile::new(vec![coordinate.clone()]);

        let xtc = coordinate_file.to_xtc_bytes().unwrap();
        let universe = Universe::from_xtc_bytes(&xtc).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.current_frame().unwrap().step, 17);
        assert!((universe.current_frame().unwrap().time - 2.5).abs() < 1.0e-6);
        assert_eq!(
            universe.current_frame().unwrap().dimensions.unwrap()[..3],
            [2.0, 3.0, 4.0]
        );

        let mut trr_coordinate = coordinate;
        trr_coordinate.velocities = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let trr = crate::xdr::TrrFile {
            n_atoms: 2,
            coordinates: crate::coordinates::CoordinateFile::new(vec![trr_coordinate]),
            steps: vec![17],
            times: vec![2.5],
            forces: vec![Some(vec![[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])],
            lambdas: vec![0.0],
            double_precision: vec![false],
        };
        let trr_bytes = trr
            .to_bytes(crate::xdr::TrrWriteOptions::default())
            .unwrap();
        let universe = Universe::from_trr_bytes(&trr_bytes).unwrap();
        let frame = universe.current_frame().unwrap();
        assert_eq!(frame.velocities.as_ref().unwrap().len(), 2);
        let force = frame.forces.as_ref().unwrap()[1];
        assert!((force[0] - 1.3).abs() < 1.0e-6);
        assert!((force[1] - 1.4).abs() < 1.0e-6);
        assert!((force[2] - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn psf_xdr_constructors_validate_topology_atom_count() {
        let psf = concat!(
            "PSF\n\n",
            "       2 !NATOM\n",
            "       1 SEG      1 ALA      N        NH1          0.000000      14.007000        0\n",
            "       2 SEG      1 ALA      CA       CT1          0.000000      12.011000        0\n",
        );
        let coordinate = crate::coordinates::CoordinateFile::new(vec![
            crate::coordinates::CoordinateFrame::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ]);
        let xtc = coordinate.to_xtc_bytes().unwrap();
        let universe = Universe::from_psf_and_xtc_bytes(psf, &xtc).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.positions()[1], [1.0, 0.0, 0.0]);

        let one_atom = crate::coordinates::CoordinateFile::new(vec![
            crate::coordinates::CoordinateFrame::new(vec![[0.0, 0.0, 0.0]]),
        ])
        .to_xtc_bytes()
        .unwrap();
        let error = Universe::from_psf_and_xtc_bytes(psf, &one_atom).unwrap_err();
        assert!(
            matches!(error, crate::Error::InvalidInput(message) if message.contains("trajectory contains 1 atoms"))
        );
    }

    #[test]
    fn atom_group_tracks_current_trajectory_frame() {
        let mut universe = sample();
        universe
            .add_frame(Frame::new(vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(
            universe.trajectory.next_frame().unwrap().positions[0],
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            universe.trajectory.next_frame().unwrap().positions[0],
            [5.0, 0.0, 0.0]
        );
        assert_eq!(universe.atoms().positions()[0], [5.0, 0.0, 0.0]);
    }

    #[test]
    fn trajectory_next_alias_tracks_current_frame_and_end() {
        let mut trajectory = Trajectory::new(vec![
            Frame::new(vec![[1.0, 0.0, 0.0]]),
            Frame::new(vec![[2.0, 0.0, 0.0]]),
        ]);
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 1.0);
        assert_eq!(trajectory.next().unwrap().positions[0][0], 1.0);
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 1.0);
        assert_eq!(trajectory.next().unwrap().positions[0][0], 2.0);
        assert!(trajectory.next().is_none());
        trajectory.rewind();
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 1.0);
    }

    #[test]
    fn universe_atoms_include_current_frame_velocities_and_forces() {
        let mut universe = Universe::from_atoms(vec![
            Atom::new(0, "N", [0.0, 0.0, 0.0]),
            Atom::new(1, "CA", [1.0, 0.0, 0.0]),
        ]);
        let mut frame = Frame::new(vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]);
        frame.velocities = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        frame.forces = Some(vec![[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]);
        universe.trajectory = Trajectory::new(vec![frame]);

        let atoms = universe.atoms();
        assert_eq!(atoms.positions(), vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]);
        assert_eq!(
            atoms.velocities(),
            Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        );
        assert_eq!(atoms.forces(), Some(vec![[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]));
    }

    #[test]
    fn trajectory_reports_atom_count_and_universe_iteration_aliases() {
        let mut universe = Universe::from_atoms(vec![Atom::new(0, "X", [0.0, 0.0, 0.0])]);
        universe
            .add_frame(Frame::new(vec![[1.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(universe.trajectory.n_atoms(), 1);
        assert_eq!(universe.next().unwrap().positions[0][0], 0.0);
        assert_eq!(universe.next_frame().unwrap().positions[0][0], 1.0);
        assert!(universe.next().is_none());
        universe.rewind();
        assert_eq!(universe.current_frame().unwrap().positions[0][0], 0.0);
    }

    #[test]
    fn trajectory_supports_collection_access_without_changing_cursor() {
        let mut trajectory = Trajectory::new(vec![
            Frame::new(vec![[1.0, 0.0, 0.0]]),
            Frame::new(vec![[2.0, 0.0, 0.0]]),
        ]);
        assert_eq!(trajectory.len(), 2);
        assert!(!trajectory.is_empty());
        assert_eq!(trajectory[1].positions[0][0], 2.0);
        trajectory[1].positions[0][0] = 3.0;
        assert_eq!(trajectory.iter().count(), 2);
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 1.0);
        assert_eq!(trajectory.next().unwrap().positions[0][0], 1.0);
        assert_eq!(trajectory[1].positions[0][0], 3.0);
    }

    #[test]
    fn trajectory_and_universe_expose_mutable_current_frames() {
        let mut trajectory = Trajectory::new(vec![
            Frame::new(vec![[1.0, 0.0, 0.0]]),
            Frame::new(vec![[2.0, 0.0, 0.0]]),
        ]);
        trajectory.current_frame_mut().unwrap().positions[0][0] = 4.0;
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 4.0);
        trajectory.next().unwrap();
        trajectory.current_frame_mut().unwrap().positions[0][0] = 5.0;
        assert_eq!(trajectory.current_frame().unwrap().positions[0][0], 5.0);

        let mut universe = sample();
        universe.current_frame_mut().unwrap().positions[0][0] = 6.0;
        universe.positions_mut().unwrap()[1][0] = 7.0;
        assert_eq!(universe.positions(), vec![[6.0, 0.0, 0.0], [7.0, 0.0, 0.0]]);
    }

    #[test]
    fn trajectory_slice_preserves_frame_metadata_and_resets_cursor() {
        let mut first = Frame::new(vec![[1.0, 0.0, 0.0]]);
        first.step = 11;
        first.time = 1.5;
        let mut second = Frame::new(vec![[2.0, 0.0, 0.0]]);
        second.step = 22;
        second.time = 2.5;
        let mut third = Frame::new(vec![[3.0, 0.0, 0.0]]);
        third.step = 33;
        third.time = 3.5;
        let trajectory = Trajectory::new(vec![first, second, third]);

        let sliced = trajectory.slice(1..=2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.current, 0);
        assert_eq!(sliced[0].positions[0][0], 2.0);
        assert_eq!(sliced[0].step, 22);
        assert_eq!(sliced[1].time, 3.5);

        assert_eq!(trajectory.slice(..2).len(), 2);
        assert_eq!(trajectory.slice(1..).len(), 2);
        assert_eq!(trajectory.slice(..=1).len(), 2);
        assert_eq!(trajectory.slice(1..=1)[0].positions[0][0], 2.0);
    }

    #[test]
    fn atom_group_supports_mutable_and_owned_iteration() {
        let mut group = AtomGroup::new(vec![
            Atom::new(0, "N", [0.0, 0.0, 0.0]),
            Atom::new(1, "CA", [1.0, 0.0, 0.0]),
        ]);
        group[0].position[0] = 2.0;
        for atom in &mut group {
            atom.position[1] = 3.0;
        }
        assert_eq!(group.positions(), vec![[2.0, 3.0, 0.0], [1.0, 3.0, 0.0]]);
        let names: Vec<_> = group.clone().into_iter().map(|atom| atom.name).collect();
        assert_eq!(names, vec!["N", "CA"]);
    }

    #[test]
    fn frame_supports_atom_access_iteration_and_metadata_preserving_slices() {
        let mut frame = Frame::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        frame.velocities = Some(vec![
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
        ]);
        frame.forces = Some(vec![
            [20.0, 20.0, 20.0],
            [21.0, 21.0, 21.0],
            [22.0, 22.0, 22.0],
            [23.0, 23.0, 23.0],
        ]);
        frame.dimensions = Some([4.0, 5.0, 6.0, 90.0, 90.0, 90.0]);
        frame.time = 2.5;
        frame.step = 17;
        frame.data.insert("lambda".to_owned(), vec![0.25]);

        assert_eq!(frame.len(), 4);
        assert!(!frame.is_empty());
        assert_eq!(frame.get(1), Some(&[1.0, 1.0, 1.0]));
        assert_eq!(frame.get(4), None);
        assert_eq!(frame[2], [2.0, 2.0, 2.0]);
        frame[2] = [20.0, 20.0, 20.0];
        assert_eq!(frame.iter().count(), 4);
        frame.iter_mut().next().unwrap()[0] = -1.0;
        assert_eq!(frame[0][0], -1.0);
        assert_eq!((&frame).into_iter().count(), frame.len());

        let sliced = frame.slice(1..3);
        assert_eq!(sliced.positions, vec![[1.0, 1.0, 1.0], [20.0, 20.0, 20.0]]);
        assert_eq!(
            sliced.velocities,
            Some(vec![[11.0, 11.0, 11.0], [12.0, 12.0, 12.0]])
        );
        assert_eq!(
            sliced.forces,
            Some(vec![[21.0, 21.0, 21.0], [22.0, 22.0, 22.0]])
        );
        assert_eq!(sliced.dimensions, frame.dimensions);
        assert_eq!(sliced.time, frame.time);
        assert_eq!(sliced.step, frame.step);
        assert_eq!(sliced.data, frame.data);
    }
}
