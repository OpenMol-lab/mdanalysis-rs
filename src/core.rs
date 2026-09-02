//! Core topology and trajectory objects.

use crate::coordinates::CoordinateFile;
use crate::dcd::DcdFile;
use crate::formats::Structure;
use crate::pdb::{PdbAtom, PdbBond, PdbStructure, read_pdb};
use crate::psf::{PsfStructure, read_psf};
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
        let element = atom.element.clone();
        let position = atom.position();
        let chain_id = atom.chain_id.map(|id| id.to_string()).unwrap_or_default();
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
            segid: if chain_id.is_empty() {
                "SYSTEM".to_string()
            } else {
                chain_id.clone()
            },
            segment_index: 0,
            chain_id,
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

impl IntoIterator for Trajectory {
    type Item = Frame;
    type IntoIter = std::vec::IntoIter<Frame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
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

    pub fn get(&self, index: usize) -> Option<&Atom> {
        self.atoms.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Atom> {
        self.atoms.iter()
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self::new(self.atoms[range].to_vec())
    }
}

impl std::ops::Index<usize> for AtomGroup {
    type Output = Atom;

    fn index(&self, index: usize) -> &Self::Output {
        &self.atoms[index]
    }
}

impl<'a> IntoIterator for &'a AtomGroup {
    type Item = &'a Atom;
    type IntoIter = std::slice::Iter<'a, Atom>;

    fn into_iter(self) -> Self::IntoIter {
        self.atoms.iter()
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
        let atoms: Vec<Atom> = pdb_atoms.into_iter().map(Atom::from).collect();
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
        Self::from_coordinate_file(CoordinateFile::read_xyz(std::fs::File::open(path)?)?)
    }

    /// Construct a universe directly from an XYZ document held in memory.
    pub fn from_xyz_str(input: &str) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::from_xyz_str(input)?)
    }

    /// Construct a universe from a Gromacs GRO file. Coordinates retain the
    /// nanometre units used by GRO; callers can convert with [`crate::units`].
    pub fn from_gro(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::read_gro(std::fs::File::open(path)?)?)
    }

    /// Construct a universe directly from a GRO document held in memory.
    pub fn from_gro_str(input: &str) -> crate::Result<Self> {
        Self::from_coordinate_file(CoordinateFile::from_gro_str(input)?)
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
        Self::from_psf_and_coordinate_file(
            read_psf(psf_path)?,
            CoordinateFile::read_xyz(std::fs::File::open(xyz_path)?)?,
        )
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
        Self::from_psf_and_coordinate_file(
            read_psf(psf_path)?,
            CoordinateFile::read_gro(std::fs::File::open(gro_path)?)?,
        )
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

    fn from_psf_and_dcd_file(psf: PsfStructure, dcd: DcdFile) -> crate::Result<Self> {
        let mut universe = Self::from_psf_structure(psf)?;
        universe.attach_dcd(dcd)?;
        Ok(universe)
    }

    fn from_pdb_and_dcd_structures(pdb: PdbStructure, dcd: DcdFile) -> crate::Result<Self> {
        let mut universe = Self::from_pdb_structure(pdb)?;
        universe.attach_dcd(dcd)?;
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
                residue_sequence: atom.resid,
                insertion_code: None,
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

    pub fn atoms(&self) -> AtomGroup {
        let positions = self.positions();
        let mut atoms = self.topology.atoms.clone();
        for (atom, position) in atoms.iter_mut().zip(positions) {
            atom.position = position;
        }
        AtomGroup::new(atoms)
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

    pub fn n_frames(&self) -> usize {
        self.trajectory.n_frames()
    }

    pub fn current_frame(&self) -> Option<&Frame> {
        if self.trajectory.current == 0 {
            self.trajectory.frames.first()
        } else {
            self.trajectory.frames.get(self.trajectory.current - 1)
        }
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
}
