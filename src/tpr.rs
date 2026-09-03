//! Native support for GROMACS portable run input (TPR/TPX) files.
//!
//! TPR is a versioned XDR format whose layout changes with GROMACS releases.
//! Parsing is delegated to [`minitpr`], which supports TPX versions 103 and
//! newer (GROMACS 5.1+).  Legacy GROMACS files from the v58, v73, v83, and v100
//! formats are parsed by the compatibility reader in this module.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use std::convert::TryFrom;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Header metadata from a parsed TPR file.
pub type TprHeader = minitpr::TprHeader;
/// Precision used by a parsed TPR file.
pub type TprPrecision = minitpr::Precision;
/// Simulation box vectors from a parsed TPR file.
pub type TprSimBox = minitpr::SimBox;
/// One atom record from a parsed TPR topology.
pub type TprAtom = minitpr::Atom;
/// One bond record from a parsed TPR topology.
pub type TprBond = minitpr::Bond;
/// Parsed atom and bond topology.
pub type TprTopology = minitpr::TprTopology;
/// Conventional data alias for [`TprFile`].
pub type TprData = TprFile;
/// Conventional structure alias for [`TprFile`].
pub type TprStructure = TprFile;

/// A parsed GROMACS TPR file.
#[derive(Clone, Debug)]
pub struct TprFile {
    pub header: TprHeader,
    pub system_name: String,
    pub simbox: Option<TprSimBox>,
    pub topology: TprTopology,
    /// The topology's initial coordinates as a one-frame coordinate file.
    pub coordinates: CoordinateFile,
}

impl TprFile {
    /// Parse a TPR file from a filesystem path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, TprError> {
        let path = path.as_ref();
        match minitpr::TprFile::parse(path) {
            Ok(parsed) => Ok(Self::from_minitpr(parsed)),
            Err(minitpr::errors::ParseTprError::UnsupportedVersion(58 | 73 | 83 | 100)) => {
                let bytes = std::fs::read(path)?;
                parse_legacy(&bytes).map(Self::from_minitpr)
            }
            Err(error) => Err(TprError::from(error)),
        }
    }

    /// Alias for [`TprFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TprError> {
        Self::read_file(path)
    }

    /// Parse a TPR file from a byte stream.
    ///
    /// `minitpr` exposes a path-based parser.  The bytes are therefore written
    /// to a process-unique temporary file before parsing and removed whether
    /// parsing succeeds or fails.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, TprError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse TPR bytes held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TprError> {
        if let Some(version) = detect_version(bytes)?
            && version < 103
        {
            return parse_legacy(bytes).map(Self::from_minitpr);
        }
        let path = temporary_path();
        let parsed = (|| -> Result<minitpr::TprFile, TprError> {
            {
                let mut file = File::create(&path)?;
                file.write_all(bytes)?;
            }
            minitpr::TprFile::parse(&path).map_err(TprError::from)
        })();
        let _ = std::fs::remove_file(&path);
        parsed.map(Self::from_minitpr)
    }

    /// Number of atoms in the topology.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.topology.atoms.len()
    }

    /// Number of bonds in the topology.
    #[must_use]
    pub fn n_bonds(&self) -> usize {
        self.topology.bonds.len()
    }

    /// Number of coordinate frames represented by this TPR.
    #[must_use]
    pub const fn n_frames(&self) -> usize {
        1
    }

    /// Return the initial coordinate frame.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    /// Convert the parsed topology and initial coordinates into a universe.
    pub fn to_universe(&self) -> crate::Result<Universe> {
        if self.topology.atoms.is_empty() {
            return Err(crate::Error::InvalidInput(
                "TPR file contains no atoms".to_owned(),
            ));
        }
        let atoms = self
            .topology
            .atoms
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let mut atom = Atom::new(
                    index,
                    source.atom_name.clone(),
                    source.position.unwrap_or([0.0; 3]),
                );
                atom.mass = source.mass;
                atom.charge = source.charge;
                atom.resid = source.residue_number;
                atom.resname = source.residue_name.clone();
                atom.element = source.element.map(|element| element.symbol().to_owned());
                atom.velocity = source.velocity;
                atom.force = source.force;
                atom
            })
            .collect::<Vec<_>>();
        let mut topology = Topology::new(atoms);
        for source in &self.topology.bonds {
            topology.add_bond(Bond::new(source.atom1, source.atom2));
        }
        let source = self
            .coordinates
            .frames
            .first()
            .ok_or_else(|| crate::Error::InvalidInput("TPR file has no frame".to_owned()))?;
        let mut frame = Frame::new(source.positions.clone());
        frame.velocities = source.velocities.clone();
        frame.forces = source
            .positions
            .iter()
            .enumerate()
            .map(|(index, _)| self.topology.atoms.get(index).and_then(|atom| atom.force))
            .collect::<Option<Vec<_>>>();
        frame.dimensions = source.dimensions;
        Ok(Universe {
            topology,
            trajectory: Trajectory::new(vec![frame]),
        })
    }

    fn from_minitpr(parsed: minitpr::TprFile) -> Self {
        let coordinates = coordinate_file(&parsed);
        Self {
            header: parsed.header,
            system_name: parsed.system_name,
            simbox: parsed.simbox,
            topology: parsed.topology,
            coordinates,
        }
    }
}

/// Read a GROMACS TPR file from a path.
pub fn read_tpr(path: impl AsRef<Path>) -> Result<TprFile, TprError> {
    TprFile::read_file(path)
}

impl CoordinateFile {
    /// Read a TPR's initial coordinate frame from a byte stream.
    pub fn read_tpr<R: Read>(reader: R) -> Result<Self, TprError> {
        Ok(TprFile::read(reader)?.coordinates)
    }

    /// Parse TPR bytes and return the initial coordinate frame.
    pub fn from_tpr_bytes(bytes: &[u8]) -> Result<Self, TprError> {
        Ok(TprFile::from_bytes(bytes)?.coordinates)
    }

    /// Read a TPR's initial coordinate frame from a path.
    pub fn read_tpr_file(path: impl AsRef<Path>) -> Result<Self, TprError> {
        Ok(TprFile::read_file(path)?.coordinates)
    }
}

impl Universe {
    /// Construct a universe from a GROMACS TPR file.
    pub fn from_tpr(path: impl AsRef<Path>) -> crate::Result<Self> {
        TprFile::read_file(path)?.to_universe()
    }

    /// Construct a universe from parsed TPR data.
    pub fn from_tpr_file(file: TprFile) -> crate::Result<Self> {
        file.to_universe()
    }

    /// Construct a universe from TPR bytes.
    pub fn from_tpr_bytes(bytes: &[u8]) -> crate::Result<Self> {
        TprFile::from_bytes(bytes)?.to_universe()
    }
}

fn coordinate_file(parsed: &minitpr::TprFile) -> CoordinateFile {
    let positions = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.position.unwrap_or([0.0; 3]))
        .collect::<Vec<_>>();
    let velocities = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.velocity)
        .collect::<Option<Vec<_>>>();
    let dimensions = parsed
        .simbox
        .as_ref()
        .map(|simbox| triclinic_box(simbox.simbox));
    let mut frame = CoordinateFrame::new(positions);
    frame.velocities = velocities;
    frame.dimensions = dimensions;
    frame.names = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.atom_name.clone())
        .collect();
    frame.residue_names = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.residue_name.clone())
        .collect();
    frame.residue_ids = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.residue_number)
        .collect();
    frame.atom_ids = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.atom_number.max(1) as usize)
        .collect();
    frame.title = parsed.system_name.clone();
    CoordinateFile::new(vec![frame])
}

fn detect_version(bytes: &[u8]) -> Result<Option<i32>, TprError> {
    if bytes.len() < 12 {
        return Ok(None);
    }
    let outer = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let inner = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let padded = inner
        .checked_add(3)
        .and_then(|value| value.checked_div(4))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| TprError::Parse("TPR header length overflow".to_owned()))?;
    let end = 8usize
        .checked_add(padded)
        .ok_or_else(|| TprError::Parse("TPR header length overflow".to_owned()))?;
    let version_end = 8usize
        .checked_add(inner)
        .ok_or_else(|| TprError::Parse("TPR header length overflow".to_owned()))?;
    let version_value_end = end
        .checked_add(8)
        .ok_or_else(|| TprError::Parse("TPR header length overflow".to_owned()))?;
    if outer < inner.saturating_add(1)
        || version_end > bytes.len()
        || version_value_end > bytes.len()
    {
        return Ok(None);
    }
    let version = std::str::from_utf8(&bytes[8..version_end]).ok();
    if !version.is_some_and(|value| value.starts_with("VERSION")) {
        return Ok(None);
    }
    Ok(Some(i32::from_be_bytes(
        bytes[end + 4..end + 8].try_into().unwrap(),
    )))
}

fn parse_legacy(bytes: &[u8]) -> Result<minitpr::TprFile, TprError> {
    let mut reader = LegacyReader::new(bytes);
    let version_string = reader.string()?;
    if !version_string.starts_with("VERSION") {
        return Err(TprError::Parse(
            "input does not look like a TPR file".to_owned(),
        ));
    }
    let precision_value = reader.i32()?;
    let precision = match precision_value {
        4 => minitpr::Precision::Single,
        8 => minitpr::Precision::Double,
        value => {
            return Err(TprError::Parse(format!(
                "unsupported TPR precision {value}"
            )));
        }
    };
    let version = reader.i32()?;
    if !matches!(version, 58 | 73 | 83 | 100) {
        return Err(TprError::UnsupportedVersion(version));
    }
    let generation = if version >= 26 { reader.i32()? } else { 0 };
    if (77..=79).contains(&version) {
        reader.i32()?;
        reader.string()?;
    }
    let file_tag = if version >= 81 {
        reader.string()?
    } else {
        "release".to_owned()
    };
    let n_atoms = reader.i32()?;
    if n_atoms < 0 {
        return Err(TprError::Parse("negative TPR atom count".to_owned()));
    }
    let n_coupling_groups = if version >= 28 { reader.i32()? } else { 0 };
    if n_coupling_groups < 0 {
        return Err(TprError::Parse(
            "negative TPR coupling-group count".to_owned(),
        ));
    }
    if version < 62 {
        reader.i32()?;
        reader.real(precision)?;
    }
    let fep_state = if version >= 79 { reader.i32()? } else { 0 };
    let lambda = reader.real(precision)?;
    let flags = [
        reader.i32()? != 0,
        reader.i32()? != 0,
        reader.i32()? != 0,
        reader.i32()? != 0,
        reader.i32()? != 0,
        reader.i32()? != 0,
    ];
    let header = minitpr::TprHeader {
        gromacs_version: version_string,
        precision,
        tpr_version: version,
        tpr_generation: generation,
        file_tag,
        n_atoms,
        n_coupling_groups,
        fep_state,
        lambda,
        has_input_record: flags[0],
        has_topology: flags[1],
        has_positions: flags[2],
        has_velocities: flags[3],
        has_forces: flags[4],
        has_box: flags[5],
        body_size: None,
    };

    let simbox = if header.has_box {
        let mut values = [[[0.0; 3]; 3]; 3];
        for matrix in &mut values {
            for row in matrix {
                for value in row {
                    *value = reader.real(precision)?;
                }
            }
        }
        Some(minitpr::SimBox {
            simbox: values[0],
            simbox_rel: values[1],
            simbox_v: values[2],
        })
    } else {
        None
    };
    if n_coupling_groups > 0 {
        let count = i64::from(n_coupling_groups) * if version < 69 { 2 } else { 1 };
        reader.skip_reals(precision, count)?;
    }
    let symbols = parse_legacy_symbols(&mut reader)?;
    let system_name = symbols
        .get(reader.usize("system name index")?)
        .ok_or_else(|| TprError::Parse("system name index is outside the symbol table".to_owned()))?
        .clone();
    let (mut atoms, bonds, atnr) = parse_legacy_topology(
        &mut reader,
        &symbols,
        version,
        precision,
        header.has_topology,
    )?;
    skip_legacy_topology_tail(&mut reader, version, precision, atnr)?;
    if header.has_positions {
        for atom in atoms.iter_mut() {
            atom.position = Some(reader.vector(precision)?);
        }
    }
    if header.has_velocities {
        for atom in atoms.iter_mut() {
            atom.velocity = Some(reader.vector(precision)?);
        }
    }
    if header.has_forces {
        for atom in atoms.iter_mut() {
            atom.force = Some(reader.vector(precision)?);
        }
    }
    if atoms.len() != n_atoms as usize {
        return Err(TprError::Parse(format!(
            "TPR atom count mismatch: header {n_atoms}, topology {}",
            atoms.len()
        )));
    }
    if bonds
        .iter()
        .any(|bond| bond.atom1 >= atoms.len() || bond.atom2 >= atoms.len())
    {
        return Err(TprError::Parse(
            "TPR topology contains a bond outside the atom table".to_owned(),
        ));
    }
    Ok(minitpr::TprFile {
        header,
        system_name,
        simbox,
        topology: minitpr::TprTopology { atoms, bonds },
    })
}

fn parse_legacy_symbols(reader: &mut LegacyReader<'_>) -> Result<Vec<String>, TprError> {
    let count = reader.count("symbol count")?;
    let mut symbols = Vec::new();
    for _ in 0..count {
        symbols.push(reader.string()?);
    }
    Ok(symbols)
}

fn parse_legacy_topology(
    reader: &mut LegacyReader<'_>,
    symbols: &[String],
    version: i32,
    precision: minitpr::Precision,
    has_topology: bool,
) -> Result<(Vec<minitpr::Atom>, Vec<minitpr::Bond>, i32), TprError> {
    if !has_topology {
        return Err(TprError::Parse(
            "TPR file does not contain topology".to_owned(),
        ));
    }
    let atnr = reader.i32()?;
    let ntypes = reader.count("force-field type count")?;
    let mut function_types = Vec::new();
    for _ in 0..ntypes {
        let mut value = reader.i32()?;
        for &(threshold, function) in FT_UPDATES {
            if version < threshold && value >= function as i32 {
                value += 1;
            }
        }
        function_types.push(value);
    }
    let _reppow = if version >= 66 { reader.f64()? } else { 12.0 };
    let _fudge = reader.real(precision)?;
    for function in function_types {
        skip_legacy_iparam(reader, function, version, precision)?;
    }
    let molecule_type_count = reader.count("molecule type count")?;
    let mut molecule_types = Vec::new();
    for _ in 0..molecule_type_count {
        molecule_types.push(parse_legacy_molecule_type(
            reader, symbols, version, precision,
        )?);
    }
    let molecule_block_count = reader.count("molecule block count")?;
    let mut blocks = Vec::new();
    for _ in 0..molecule_block_count {
        let molecule_type = reader.usize("molecule type index")?;
        let molecules = reader.count("molecule count")?;
        let _atoms_per_molecule = reader.i32()?;
        for _ in 0..2 {
            let count = reader.count("position restraint count")?;
            reader.skip_reals(
                precision,
                i64::try_from(count)
                    .map_err(|_| TprError::Parse("position restraint count overflow".to_owned()))?
                    * 3,
            )?;
        }
        blocks.push((molecule_type, molecules));
    }
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut atom_counter = 0usize;
    let mut residue_counter = 0i32;
    for (type_index, molecule_count) in blocks {
        let molecule = molecule_types.get(type_index).ok_or_else(|| {
            TprError::Parse("molecule block references an unknown molecule type".to_owned())
        })?;
        for _ in 0..molecule_count {
            let atom_start = atom_counter;
            let mut previous_residue = None;
            for source in &molecule.atoms {
                if previous_residue != Some(source.residue_index) {
                    residue_counter += 1;
                    previous_residue = Some(source.residue_index);
                }
                atoms.push(minitpr::Atom {
                    atom_name: source.name.clone(),
                    atom_number: i32::try_from(atom_counter).map_err(|_| {
                        TprError::Parse("TPR atom index exceeds i32 range".to_owned())
                    })?,
                    residue_name: source.residue_name.clone(),
                    residue_number: residue_counter,
                    mass: source.mass,
                    charge: source.charge,
                    element: element_from_atomic_number(source.atomic_number),
                    position: None,
                    velocity: None,
                    force: None,
                });
                atom_counter += 1;
            }
            for &(first, second) in &molecule.bonds {
                let first = atom_start
                    .checked_add(first)
                    .ok_or_else(|| TprError::Parse("TPR bond index overflow".to_owned()))?;
                let second = atom_start
                    .checked_add(second)
                    .ok_or_else(|| TprError::Parse("TPR bond index overflow".to_owned()))?;
                bonds.push(minitpr::Bond {
                    atom1: first,
                    atom2: second,
                });
            }
        }
    }
    Ok((atoms, bonds, atnr))
}

struct LegacyMolecule {
    atoms: Vec<LegacyAtom>,
    bonds: Vec<(usize, usize)>,
}

struct LegacyAtom {
    name: String,
    residue_name: String,
    residue_index: i32,
    mass: f64,
    charge: f64,
    atomic_number: i32,
}

fn parse_legacy_molecule_type(
    reader: &mut LegacyReader<'_>,
    symbols: &[String],
    version: i32,
    precision: minitpr::Precision,
) -> Result<LegacyMolecule, TprError> {
    let _molecule_name = symbol(symbols, reader.usize("molecule name index")?)?;
    let atom_count = reader.count("molecule atom count")?;
    let residue_count = reader.count("molecule residue count")?;
    let mut atoms = Vec::new();
    for _ in 0..atom_count {
        let mass = reader.real(precision)?;
        let charge = reader.real(precision)?;
        reader.skip_reals(precision, 2)?;
        reader.legacy_ushort()?;
        reader.legacy_ushort()?;
        reader.i32()?;
        let residue_index = reader.i32()?;
        let atomic_number = reader.i32()?;
        atoms.push(LegacyAtom {
            name: String::new(),
            residue_name: String::new(),
            residue_index,
            mass,
            charge,
            atomic_number,
        });
    }
    for atom in &mut atoms {
        let idx = reader.usize("atom name index")?;
        atom.name = symbol(symbols, idx)?.to_owned();
    }
    let atom_type_count = atom_count
        .checked_mul(2)
        .ok_or_else(|| TprError::Parse("atom type count overflow".to_owned()))?;
    for _ in 0..atom_type_count {
        symbol(symbols, reader.usize("atom type index")?)?;
    }
    let mut residues = Vec::new();
    for _ in 0..residue_count {
        let name = symbol(symbols, reader.usize("residue name index")?)?.to_owned();
        if version >= 63 {
            reader.i32()?;
            reader.legacy_uchar()?;
        }
        residues.push(name);
    }
    for (index, atom) in atoms.iter_mut().enumerate() {
        atom.residue_name = residues
            .get(atom.residue_index as usize)
            .ok_or_else(|| TprError::Parse(format!("atom {index} has invalid residue index")))?
            .clone();
    }
    let mut bonds = Vec::new();
    for function in 0..F_NRE {
        if legacy_function_unavailable(version, function) {
            continue;
        }
        let count = reader.count("interaction list length")?;
        if is_bond_function(function) {
            for _ in 0..count / 3 {
                reader.i32()?;
                let first = usize::try_from(reader.i32()?).map_err(|_| {
                    TprError::Parse("negative atom index in TPR bond list".to_owned())
                })?;
                let second = usize::try_from(reader.i32()?).map_err(|_| {
                    TprError::Parse("negative atom index in TPR bond list".to_owned())
                })?;
                bonds.push((first, second));
            }
            reader.skip_i32s(i64::try_from(count % 3).map_err(|_| {
                TprError::Parse("bond interaction list length overflow".to_owned())
            })?)?;
        } else if function == F_SETTLE {
            if count == 2 {
                reader.i32()?;
                let base = usize::try_from(reader.i32()?).map_err(|_| {
                    TprError::Parse("negative atom index in TPR SETTLE list".to_owned())
                })?;
                bonds.push((
                    base,
                    base.checked_add(1).ok_or_else(|| {
                        TprError::Parse("TPR SETTLE atom index overflow".to_owned())
                    })?,
                ));
                bonds.push((
                    base,
                    base.checked_add(2).ok_or_else(|| {
                        TprError::Parse("TPR SETTLE atom index overflow".to_owned())
                    })?,
                ));
            } else {
                for _ in 0..count / 4 {
                    reader.i32()?;
                    let base = usize::try_from(reader.i32()?).map_err(|_| {
                        TprError::Parse("negative atom index in TPR SETTLE list".to_owned())
                    })?;
                    let first = usize::try_from(reader.i32()?).map_err(|_| {
                        TprError::Parse("negative atom index in TPR SETTLE list".to_owned())
                    })?;
                    let second = usize::try_from(reader.i32()?).map_err(|_| {
                        TprError::Parse("negative atom index in TPR SETTLE list".to_owned())
                    })?;
                    bonds.push((base, first));
                    bonds.push((base, second));
                }
                reader.skip_i32s(i64::try_from(count % 4).map_err(|_| {
                    TprError::Parse("SETTLE interaction list length overflow".to_owned())
                })?)?;
            }
        } else {
            reader
                .skip_i32s(i64::try_from(count).map_err(|_| {
                    TprError::Parse("interaction list length overflow".to_owned())
                })?)?;
        }
    }
    let block_count = reader.count("charge-group block length")?;
    reader.skip_i32s(
        i64::try_from(block_count)
            .map_err(|_| TprError::Parse("charge-group block length overflow".to_owned()))?
            + 1,
    )?;
    let block_count = reader.count("exclusion block length")?;
    let excluded_count = reader.count("excluded atom count")?;
    reader.skip_i32s(
        i64::try_from(block_count)
            .map_err(|_| TprError::Parse("exclusion block length overflow".to_owned()))?
            + 1,
    )?;
    reader.skip_i32s(
        i64::try_from(excluded_count)
            .map_err(|_| TprError::Parse("excluded atom count overflow".to_owned()))?,
    )?;
    Ok(LegacyMolecule { atoms, bonds })
}

fn symbol(symbols: &[String], index: usize) -> Result<&str, TprError> {
    symbols
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| TprError::Parse(format!("symbol index {index} is outside the symbol table")))
}

fn element_from_atomic_number(number: i32) -> Option<minitpr::Element> {
    if number <= 0 {
        return None;
    }
    minitpr::Element::list().get(number as usize - 1).copied()
}

fn skip_legacy_topology_tail(
    reader: &mut LegacyReader<'_>,
    version: i32,
    precision: minitpr::Precision,
    atnr: i32,
) -> Result<(), TprError> {
    reader.i32()?;
    if version < 116 {
        reader.skip_i32s(
            i64::from(atnr.max(0))
                * 3
                * if precision == minitpr::Precision::Double {
                    2
                } else {
                    1
                },
        )?;
    }
    if version < 129 {
        reader.skip_i32s(i64::from(atnr.max(0)) + 1)?;
    }
    if (59..=83).contains(&version) {
        reader.skip_i32s(
            i64::from(atnr.max(0))
                * if precision == minitpr::Precision::Double {
                    4
                } else {
                    2
                },
        )?;
    }
    if (84..116).contains(&version) {
        reader.skip_i32s(i64::from(atnr.max(0)) * 2)?;
    }
    if version > 58 {
        let ngrid = reader.count("dihedral grid count")?;
        let spacing = reader.count("dihedral grid spacing")?;
        let ngrid = i64::try_from(ngrid)
            .map_err(|_| TprError::Parse("dihedral grid count overflow".to_owned()))?;
        let spacing = i64::try_from(spacing)
            .map_err(|_| TprError::Parse("dihedral grid spacing overflow".to_owned()))?;
        let total = ngrid
            .checked_mul(spacing)
            .and_then(|value| value.checked_mul(spacing))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| TprError::Parse("dihedral grid size overflow".to_owned()))?;
        reader.skip_reals(precision, total)?;
    }
    for _ in 0..10 {
        let count = reader.count("atom group size")?;
        reader.skip_i32s(
            i64::try_from(count)
                .map_err(|_| TprError::Parse("atom group size overflow".to_owned()))?,
        )?;
    }
    let names = reader.count("atom group name count")?;
    reader.skip_i32s(
        i64::try_from(names)
            .map_err(|_| TprError::Parse("atom group name count overflow".to_owned()))?,
    )?;
    for _ in 0..10 {
        let count = reader.count("atom group number count")?;
        reader.skip_legacy_uchars(
            i64::try_from(count)
                .map_err(|_| TprError::Parse("atom group number count overflow".to_owned()))?,
        )?;
    }
    Ok(())
}

fn is_bond_function(function: usize) -> bool {
    matches!(
        function,
        F_BONDS
            | F_G96BONDS
            | F_MORSE
            | F_CUBICBONDS
            | F_CONNBONDS
            | F_HARMONIC
            | F_FENEBONDS
            | F_TABBONDS
            | F_TABBONDSNC
            | F_RESTRBONDS
            | F_CONSTR
            | F_CONSTRNC
    )
}

fn legacy_function_unavailable(version: i32, function: usize) -> bool {
    FT_UPDATES
        .iter()
        .any(|&(threshold, index)| version < threshold && function == index)
}

const F_NRE: usize = 95;
const F_BONDS: usize = 0;
const F_G96BONDS: usize = 1;
const F_MORSE: usize = 2;
const F_CUBICBONDS: usize = 3;
const F_CONNBONDS: usize = 4;
const F_HARMONIC: usize = 5;
const F_FENEBONDS: usize = 6;
const F_TABBONDS: usize = 7;
const F_TABBONDSNC: usize = 8;
const F_RESTRBONDS: usize = 9;
const F_ANGLES: usize = 10;
const F_G96ANGLES: usize = 11;
const F_RESTRANGLES: usize = 12;
const F_LINEAR_ANGLES: usize = 13;
const F_CROSS_BOND_BONDS: usize = 14;
const F_CROSS_BOND_ANGLES: usize = 15;
const F_UREY_BRADLEY: usize = 16;
const F_QUARTIC_ANGLES: usize = 17;
const F_TABANGLES: usize = 18;
const F_PDIHS: usize = 19;
const F_RBDIHS: usize = 20;
const F_RESTRDIHS: usize = 21;
const F_CBTDIHS: usize = 22;
const F_FOURDIHS: usize = 23;
const F_IDIHS: usize = 24;
const F_PIDIHS: usize = 25;
const F_TABDIHS: usize = 26;
const F_CMAP: usize = 27;
const F_GB12: usize = 28;
const F_GB13: usize = 29;
const F_GB14: usize = 30;
const F_LJ14: usize = 33;
const F_LJC14_Q: usize = 35;
const F_LJC_PAIRS_NB: usize = 36;
const F_LJ: usize = 37;
const F_BHAM: usize = 38;
const F_BHAM_LR: usize = 40;
const F_RF_EXCL: usize = 44;
const F_COUL_RECIP: usize = 45;
const F_LJ_RECIP: usize = 46;
const F_DPD: usize = 47;
const F_POLARIZATION: usize = 48;
const F_WATER_POL: usize = 49;
const F_THOLE_POL: usize = 50;
const F_ANHARM_POL: usize = 51;
const F_POSRES: usize = 52;
const F_FBPOSRES: usize = 53;
const F_DISRES: usize = 54;
const F_DISRESVIOL: usize = 55;
const F_ORIRES: usize = 56;
const F_ORIRESDEV: usize = 57;
const F_ANGRES: usize = 58;
const F_ANGRESZ: usize = 59;
const F_DIHRES: usize = 60;
const F_DIHRESVIOL: usize = 61;
const F_CONSTR: usize = 62;
const F_CONSTRNC: usize = 63;
const F_SETTLE: usize = 64;
const F_VSITE1: usize = 65;
const F_VSITE2: usize = 66;
const F_VSITE2FD: usize = 67;
const F_VSITE3: usize = 68;
const F_VSITE3FD: usize = 69;
const F_VSITE3FAD: usize = 70;
const F_VSITE3OUT: usize = 71;
const F_VSITE4FD: usize = 72;
const F_VSITE4FDN: usize = 73;
const F_VSITEN: usize = 74;
const F_COM_PULL: usize = 75;
const F_DENSITYFITTING: usize = 76;
const F_EQM: usize = 77;
const F_ENNPOT: usize = 78;
const F_ECONSERVED: usize = 82;
const F_VTEMP_NOLONGERUSED: usize = 84;
const F_PDISPCORR: usize = 85;
const F_DHDL_CON: usize = 87;
const F_DVDL_COUL: usize = 90;
const F_DVDL_VDW: usize = 91;
const F_DVDL_BONDED: usize = 92;
const F_DVDL_RESTRAINT: usize = 93;
const F_DVDL_TEMPERATURE: usize = 94;

const FT_UPDATES: &[(i32, usize)] = &[
    (20, F_CUBICBONDS),
    (20, F_CONNBONDS),
    (20, F_HARMONIC),
    (34, F_FENEBONDS),
    (43, F_TABBONDS),
    (43, F_TABBONDSNC),
    (70, F_RESTRBONDS),
    (98, F_RESTRANGLES),
    (76, F_LINEAR_ANGLES),
    (30, F_CROSS_BOND_BONDS),
    (30, F_CROSS_BOND_ANGLES),
    (30, F_UREY_BRADLEY),
    (34, F_QUARTIC_ANGLES),
    (43, F_TABANGLES),
    (98, F_RESTRDIHS),
    (98, F_CBTDIHS),
    (26, F_FOURDIHS),
    (26, F_PIDIHS),
    (43, F_TABDIHS),
    (65, F_CMAP),
    (60, F_GB12),
    (61, F_GB13),
    (61, F_GB14),
    (72, 31),
    (72, 32),
    (41, F_LJC14_Q),
    (41, F_LJC_PAIRS_NB),
    (32, F_BHAM_LR),
    (32, F_RF_EXCL),
    (32, F_COUL_RECIP),
    (93, F_LJ_RECIP),
    (46, F_DPD),
    (30, F_POLARIZATION),
    (36, F_THOLE_POL),
    (90, F_FBPOSRES),
    (22, F_DISRESVIOL),
    (22, F_ORIRES),
    (22, F_ORIRESDEV),
    (26, F_DIHRES),
    (26, F_DIHRESVIOL),
    (49, F_VSITE4FDN),
    (50, F_VSITEN),
    (46, F_COM_PULL),
    (20, F_EQM),
    (46, F_ECONSERVED),
    (69, F_VTEMP_NOLONGERUSED),
    (66, F_PDISPCORR),
    (54, F_DHDL_CON),
    (76, F_ANHARM_POL),
    (79, F_DVDL_COUL),
    (79, F_DVDL_VDW),
    (79, F_DVDL_BONDED),
    (79, F_DVDL_RESTRAINT),
    (79, F_DVDL_TEMPERATURE),
    (117, F_DENSITYFITTING),
    (121, F_VSITE1),
    (118, F_VSITE2FD),
    (137, F_ENNPOT),
];

fn skip_legacy_iparam(
    reader: &mut LegacyReader<'_>,
    function: i32,
    version: i32,
    precision: minitpr::Precision,
) -> Result<(), TprError> {
    let real = |reader: &mut LegacyReader<'_>, count: i64| reader.skip_reals(precision, count);
    let count = match usize::try_from(function) {
        Ok(value) => value,
        Err(_) => {
            return Err(TprError::Parse(format!(
                "negative interaction function {function}"
            )));
        }
    };
    match count {
        F_BONDS | F_G96BONDS | F_HARMONIC | F_ANGLES | F_G96ANGLES | F_IDIHS => real(reader, 4),
        F_RESTRANGLES => real(reader, if version >= 134 { 4 } else { 2 }),
        F_LINEAR_ANGLES => real(reader, 4),
        F_FENEBONDS => real(reader, 2),
        F_RESTRBONDS => real(reader, 8),
        F_TABBONDS | F_TABBONDSNC | F_TABANGLES | F_TABDIHS => {
            real(reader, 1)?;
            reader.i32()?;
            real(reader, 1)
        }
        F_CROSS_BOND_BONDS => real(reader, 3),
        F_CROSS_BOND_ANGLES => real(reader, 4),
        F_UREY_BRADLEY => real(reader, if version >= 79 { 8 } else { 4 }),
        F_QUARTIC_ANGLES => real(reader, 6),
        F_BHAM => real(reader, 3),
        F_MORSE => real(reader, if version >= 79 { 6 } else { 3 }),
        F_CUBICBONDS => real(reader, 3),
        F_CONNBONDS | F_VSITE1 => Ok(()),
        F_POLARIZATION => real(reader, 1),
        F_ANHARM_POL => real(reader, 3),
        F_WATER_POL => real(reader, 6),
        F_THOLE_POL => real(reader, if version < 127 { 4 } else { 3 }),
        F_LJ => real(reader, 2),
        F_LJ14 => real(reader, 4),
        F_LJC14_Q => real(reader, 5),
        F_LJC_PAIRS_NB => real(reader, 4),
        F_PIDIHS | F_ANGRES | F_ANGRESZ | F_PDIHS => {
            real(reader, 4)?;
            reader.i32()?;
            Ok(())
        }
        F_RESTRDIHS => real(reader, if version >= 134 { 4 } else { 2 }),
        F_DISRES => {
            reader.skip_i32s(2)?;
            real(reader, 4)
        }
        F_ORIRES => {
            reader.skip_i32s(3)?;
            real(reader, 3)
        }
        F_DIHRES => {
            if version < 72 {
                reader.skip_i32s(2)?;
            }
            real(reader, if version >= 72 { 6 } else { 3 })
        }
        F_POSRES => real(reader, 12),
        F_FBPOSRES => {
            reader.i32()?;
            real(reader, 5)
        }
        F_CBTDIHS => real(reader, if version >= 134 { 12 } else { 6 }),
        F_RBDIHS | F_FOURDIHS => real(reader, 12),
        F_CONSTR | F_CONSTRNC => real(reader, 2),
        F_SETTLE => real(reader, 2),
        F_VSITE2 | F_VSITE2FD => real(reader, 1),
        F_VSITE3 | F_VSITE3FD | F_VSITE3FAD => real(reader, 2),
        F_VSITE3OUT | F_VSITE4FD | F_VSITE4FDN => real(reader, 3),
        F_VSITEN => {
            reader.i32()?;
            real(reader, 1)
        }
        F_GB12 | F_GB13 | F_GB14 => {
            if version < 68 {
                real(reader, 4)?;
            }
            real(reader, 5)
        }
        F_CMAP => reader.skip_i32s(2),
        31 | 32 | 34 | 39 | 41 | 42 | 43 | 55 | 57 | 61 | 75..=95 => Ok(()),
        _ => Err(TprError::Parse(format!(
            "unsupported legacy interaction function {function}"
        ))),
    }
}

struct LegacyReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> LegacyReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], TprError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| TprError::Parse("TPR offset overflow".to_owned()))?;
        if end > self.bytes.len() {
            return Err(TprError::Parse("unexpected end of TPR file".to_owned()));
        }
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32, TprError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, TprError> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, TprError> {
        Ok(i64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, TprError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, TprError> {
        Ok(f64::from_bits(self.i64()? as u64))
    }

    fn real(&mut self, precision: minitpr::Precision) -> Result<f64, TprError> {
        match precision {
            minitpr::Precision::Single => Ok(self.f32()? as f64),
            minitpr::Precision::Double => self.f64(),
        }
    }

    fn vector(&mut self, precision: minitpr::Precision) -> Result<[f64; 3], TprError> {
        Ok([
            self.real(precision)?,
            self.real(precision)?,
            self.real(precision)?,
        ])
    }

    fn skip(&mut self, count: usize) -> Result<(), TprError> {
        self.take(count).map(|_| ())
    }

    fn skip_i32s(&mut self, count: i64) -> Result<(), TprError> {
        let count = usize::try_from(count)
            .map_err(|_| TprError::Parse("negative TPR skip count".to_owned()))?;
        self.skip(
            count
                .checked_mul(4)
                .ok_or_else(|| TprError::Parse("TPR skip overflow".to_owned()))?,
        )
    }

    fn skip_reals(&mut self, precision: minitpr::Precision, count: i64) -> Result<(), TprError> {
        let count = usize::try_from(count)
            .map_err(|_| TprError::Parse("negative TPR real count".to_owned()))?;
        let width = if precision == minitpr::Precision::Double {
            8
        } else {
            4
        };
        self.skip(
            count
                .checked_mul(width)
                .ok_or_else(|| TprError::Parse("TPR skip overflow".to_owned()))?,
        )
    }

    fn skip_legacy_uchars(&mut self, count: i64) -> Result<(), TprError> {
        self.skip_i32s(count)
    }

    fn legacy_ushort(&mut self) -> Result<u32, TprError> {
        self.u32()
    }

    fn legacy_uchar(&mut self) -> Result<u32, TprError> {
        self.u32()
    }

    fn count(&mut self, name: &str) -> Result<usize, TprError> {
        self.nonnegative_usize(name)
    }

    fn usize(&mut self, name: &str) -> Result<usize, TprError> {
        self.nonnegative_usize(name)
    }

    fn nonnegative_usize(&mut self, name: &str) -> Result<usize, TprError> {
        let value = self.i32()?;
        usize::try_from(value).map_err(|_| TprError::Parse(format!("negative {name}")))
    }

    fn string(&mut self) -> Result<String, TprError> {
        let _outer = self.u32()?;
        let length = self.count("TPR string length")?;
        let bytes = self.take(length)?;
        let padded = length
            .checked_add(3)
            .and_then(|value| value.checked_div(4))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| TprError::Parse("TPR string length overflow".to_owned()))?;
        self.skip(padded - length)?;
        let end = bytes
            .iter()
            .position(|&value| value == 0)
            .unwrap_or(bytes.len());
        std::str::from_utf8(&bytes[..end])
            .map(str::to_owned)
            .map_err(|error| TprError::Parse(format!("invalid TPR string: {error}")))
    }
}

fn temporary_path() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("mdanalysis-rs-tpr-{}-{id}.tpr", std::process::id()))
}

/// Errors produced while reading a TPR file.
#[derive(Debug)]
pub enum TprError {
    Io(io::Error),
    UnsupportedVersion(i32),
    Parse(String),
}

impl fmt::Display for TprError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "TPR I/O error: {error}"),
            Self::UnsupportedVersion(version) => write!(
                formatter,
                "unsupported TPR version {version}; supported versions are 58, 73, 83, 100, and 103 or newer"
            ),
            Self::Parse(message) => write!(formatter, "TPR parse error: {message}"),
        }
    }
}

impl std::error::Error for TprError {}

impl From<io::Error> for TprError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<minitpr::errors::ParseTprError> for TprError {
    fn from(error: minitpr::errors::ParseTprError) -> Self {
        use minitpr::errors::ParseTprError;
        match error {
            ParseTprError::UnsupportedVersion(version) => Self::UnsupportedVersion(version),
            ParseTprError::CouldNotRead(error) => Self::Io(error),
            other => Self::Parse(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/tprs")
            .join(name)
    }

    #[test]
    fn reads_modern_fixture_and_builds_universe() {
        let file = read_tpr(fixture("2lyz_gmx_2024.tpr")).expect("modern TPR should parse");
        assert_eq!(file.header.tpr_version, 133);
        assert_eq!(file.n_atoms(), 2263);
        assert_eq!(file.n_bonds(), 2186);
        assert_eq!(file.n_frames(), 1);
        assert_eq!(file.system_name, "HEN EGG WHITE LYSOZYME");
        let frame = file.frame(0).expect("initial TPR frame");
        assert_eq!(frame.n_atoms(), 2263);
        assert_eq!(frame.names[0], "N");
        assert_eq!(frame.residue_names[0], "LYSH");
        assert_eq!(frame.residue_ids[0], 1);
        assert!(frame.velocities.is_some());
        let dimensions = frame.dimensions.expect("TPR box");
        assert!((dimensions[0] - 7.91).abs() < 1e-5);
        assert!((dimensions[2] - 3.79).abs() < 1e-5);

        let universe = file.to_universe().expect("TPR universe");
        assert_eq!(universe.n_atoms(), 2263);
        assert_eq!(universe.n_frames(), 1);
        assert_eq!(universe.topology.bonds.len(), 2186);
        assert_eq!(universe.topology.atoms[0].name, "N");
        assert_eq!(universe.topology.atoms[0].resname, "LYSH");
        assert_eq!(
            universe.current_frame().unwrap().positions[0],
            frame.positions[0]
        );
    }

    #[test]
    fn bytes_and_coordinate_constructors_match_path_reader() {
        let bytes = std::fs::read(fixture("2lyz_gmx_2024.tpr")).expect("fixture bytes");
        let from_bytes = TprFile::from_bytes(&bytes).expect("TPR bytes should parse");
        let coordinates = CoordinateFile::from_tpr_bytes(&bytes).expect("TPR coordinates");
        assert_eq!(coordinates, from_bytes.coordinates);
        assert_eq!(coordinates.n_frames(), 1);
        assert_eq!(coordinates.n_atoms(), 2263);
    }

    #[test]
    fn reads_legacy_v58_fixture() {
        let file = read_tpr(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../mdanalysis/testsuite/MDAnalysisTests/data/adk_oplsaa.tpr"),
        )
        .expect("legacy TPR should parse");
        assert_eq!(file.header.tpr_version, 58);
        assert_eq!(file.n_atoms(), 47681);
        assert_eq!(file.n_frames(), 1);
        assert_eq!(file.frame(0).unwrap().n_atoms(), 47681);
        assert_eq!(file.frame(0).unwrap().names[0], "N");
    }

    #[test]
    fn reads_other_supported_legacy_versions() {
        for (name, version) in [
            ("281_small.tpr", 100),
            ("cobrotoxin.tpr", 73),
            ("../tprs/2lyz_gmx_4.0.2.tpr", 58),
            ("../tprs/2lyz_gmx_4.0.3.tpr", 58),
            ("../tprs/2lyz_gmx_4.0.4.tpr", 58),
            ("../tprs/2lyz_gmx_4.0.5.tpr", 58),
            ("../tprs/2lyz_gmx_4.0.6.tpr", 58),
            ("../tprs/2lyz_gmx_4.0.7.tpr", 58),
            ("../tprs/virtual_sites/extra-interactions-4.0.7.tpr", 58),
            ("../tprs/2lyz_gmx_4.5.tpr", 73),
            ("../tprs/2lyz_gmx_4.5.1.tpr", 73),
            ("2lyz_gmx_4.5.2.tpr", 73),
            ("../tprs/2lyz_gmx_4.5.3.tpr", 73),
            ("../tprs/2lyz_gmx_4.5.4.tpr", 73),
            ("../tprs/2lyz_gmx_4.5.5.tpr", 73),
            ("ab42_gmx_4.6.tpr", 83),
            ("../tprs/ab42_gmx_4.6.1.tpr", 83),
            ("../tprs/2lyz_gmx_5.0.2.tpr", 100),
            ("../tprs/2lyz_gmx_5.0.4.tpr", 100),
            ("2lyz_gmx_5.0.5.tpr", 100),
            ("drew_gmx_4.5.5.double.tpr", 73),
        ] {
            let path = if matches!(name, "281_small.tpr" | "cobrotoxin.tpr") {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../mdanalysis/testsuite/MDAnalysisTests/data")
                    .join(name)
            } else {
                fixture(name)
            };
            let file =
                read_tpr(path).unwrap_or_else(|error| panic!("{name} should parse: {error}"));
            assert_eq!(file.header.tpr_version, version);
            assert!(file.n_atoms() > 0);
            let frame = file.frame(0).expect("legacy TPR frame");
            assert_eq!(frame.n_atoms(), file.n_atoms());
            assert!(!frame.names[0].is_empty());
            if name == "drew_gmx_4.5.5.double.tpr" {
                assert!((frame.positions[0][0] - 1.487).abs() < 1e-12);
                assert!((frame.positions[0][1] - 4.026).abs() < 1e-12);
                assert!((frame.positions[0][2] - 7.954).abs() < 1e-12);
            }
        }
    }
}
