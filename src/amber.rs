//! Amber restart (INPCRD/RESTRT) and NAMD binary coordinate readers.
//!
//! Amber restart files are text records with fixed-width coordinate values;
//! optional unit-cell and velocity records are retained when present.  NAMD's
//! `coor`/`namdbin` format contains a native-endian atom count followed by
//! double-precision Cartesian coordinates.  The public containers use the
//! same [`crate::coordinates::CoordinateFrame`] representation as the other
//! trajectory readers.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use bzip2::read::BzDecoder;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// One atom record from an Amber PRMTOP/PARM7/TOP topology.
#[derive(Clone, Debug, PartialEq)]
pub struct AmberTopAtom {
    pub index: usize,
    pub name: String,
    pub atom_type: String,
    pub type_index: i32,
    pub charge: f64,
    pub mass: f64,
    pub element: Option<String>,
    pub residue_index: usize,
    pub resid: i32,
    pub resname: String,
    pub segid: String,
    pub chain_id: String,
}

/// A bond listed in an Amber topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmberBond {
    pub atom1: usize,
    pub atom2: usize,
}

/// An angle listed in an Amber topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmberAngle {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
}

/// A conventional dihedral listed in an Amber topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AmberDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

/// An improper dihedral listed in an Amber topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmberImproper {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

/// Parsed Amber PRMTOP/PARM7/TOP topology data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AmberTopFile {
    pub title: String,
    pub pointers: Vec<i64>,
    pub atoms: Vec<AmberTopAtom>,
    pub bonds: Vec<AmberBond>,
    pub angles: Vec<AmberAngle>,
    pub dihedrals: Vec<AmberDihedral>,
    pub impropers: Vec<AmberImproper>,
    /// Chain/segment identifiers in residue order.
    pub residue_chain_ids: Vec<String>,
}

/// Conventional aliases for Amber topology naming.
pub type PrmtopFile = AmberTopFile;
pub type AmberTopology = AmberTopFile;

/// A parsed Amber restart file.
#[derive(Clone, Debug, PartialEq)]
pub struct InpcrdFile {
    pub title: String,
    pub time: Option<f64>,
    pub coordinates: CoordinateFile,
}

/// A parsed NAMD binary coordinate file.
#[derive(Clone, Debug, PartialEq)]
pub struct NamdBinFile {
    pub coordinates: CoordinateFile,
}

/// Errors produced by Amber restart or NAMD binary operations.
#[derive(Debug)]
pub enum AmberError {
    Io(io::Error),
    Parse {
        format: &'static str,
        message: String,
    },
    InvalidStructure(String),
}

impl fmt::Display for AmberError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { format, message } => write!(formatter, "{format} parse error: {message}"),
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid coordinate structure: {message}")
            }
        }
    }
}

impl std::error::Error for AmberError {}

impl From<io::Error> for AmberError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CoordinateError> for AmberError {
    fn from(error: CoordinateError) -> Self {
        Self::InvalidStructure(error.to_string())
    }
}

impl AmberTopFile {
    /// Parse an Amber topology from UTF-8 text.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, AmberError> {
        parse_amber_top(input)
    }

    /// Parse an Amber topology from bytes.  bzip2-compressed PRMTOP input is
    /// accepted when the byte stream starts with the standard `BZh` marker.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AmberError> {
        if bytes.starts_with(b"BZh") {
            let mut decoder = BzDecoder::new(bytes);
            let mut decoded = String::new();
            decoder.read_to_string(&mut decoded)?;
            Self::from_str(&decoded)
        } else {
            let text = std::str::from_utf8(bytes).map_err(|error| {
                parse_error("PRMTOP", format!("topology is not valid UTF-8: {error}"))
            })?;
            Self::from_str(text)
        }
    }

    /// Read an Amber topology from a reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, AmberError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Read an Amber topology from a filesystem path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, AmberError> {
        Self::read(File::open(path)?)
    }

    /// Number of atoms in the topology.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Number of residues in the topology.
    #[must_use]
    pub fn n_residues(&self) -> usize {
        self.atoms
            .iter()
            .map(|atom| atom.residue_index)
            .max()
            .map_or(0, |index| index + 1)
    }

    /// Number of segments represented by residue chain identifiers.
    #[must_use]
    pub fn n_segments(&self) -> usize {
        let mut ids = Vec::new();
        for atom in &self.atoms {
            if !ids.iter().any(|id| id == &atom.segid) {
                ids.push(atom.segid.as_str());
            }
        }
        ids.len()
    }
}

/// Read an Amber PRMTOP/PARM7/TOP file.
pub fn read_amber_top(path: impl AsRef<Path>) -> Result<AmberTopFile, AmberError> {
    AmberTopFile::read_file(path)
}

/// Alias for [`read_amber_top`].
pub fn read_prmtop(path: impl AsRef<Path>) -> Result<PrmtopFile, AmberError> {
    read_amber_top(path)
}

/// Alias for [`read_amber_top`].
pub fn read_top(path: impl AsRef<Path>) -> Result<AmberTopFile, AmberError> {
    read_amber_top(path)
}

fn parse_amber_top(input: &str) -> Result<AmberTopFile, AmberError> {
    let lines: Vec<&str> = input.lines().collect();
    if lines
        .first()
        .is_none_or(|line| !line.trim_start().starts_with("%VE"))
    {
        return Err(parse_error("PRMTOP", "%VE version header is missing"));
    }
    let sections = parse_top_sections(&lines)?;
    let title_values = parse_section_strings(
        sections
            .get("TITLE")
            .ok_or_else(|| parse_error("PRMTOP", "TITLE section is missing"))?,
    )?;
    let title = title_values.join("").trim().to_owned();
    let pointers = parse_section_ints(
        sections
            .get("POINTERS")
            .ok_or_else(|| parse_error("PRMTOP", "POINTERS section is missing"))?,
    )?;
    if pointers.len() < 12 {
        return Err(parse_error(
            "PRMTOP",
            format!("POINTERS contains {}; expected at least 12", pointers.len()),
        ));
    }
    let n_atoms = positive_count(pointers[0], "atom count")?;
    let n_residues = positive_or_zero_count(pointers[11], "residue count")?;
    let names = required_strings(&sections, "ATOM_NAME")?;
    let masses = required_floats(&sections, "MASS")?;
    let charges = required_floats(&sections, "CHARGE")?;
    let types = required_strings(&sections, "AMBER_ATOM_TYPE")?;
    let type_indices = required_ints(&sections, "ATOM_TYPE_INDEX")?;
    for (label, length) in [
        ("ATOM_NAME", names.len()),
        ("MASS", masses.len()),
        ("CHARGE", charges.len()),
        ("AMBER_ATOM_TYPE", types.len()),
        ("ATOM_TYPE_INDEX", type_indices.len()),
    ] {
        if length != n_atoms {
            return Err(parse_error(
                "PRMTOP",
                format!("{label} contains {length} values; expected {n_atoms}"),
            ));
        }
    }
    let resnames = required_strings(&sections, "RESIDUE_LABEL")?;
    let residue_pointers = required_ints(&sections, "RESIDUE_POINTER")?;
    if resnames.len() != n_residues || residue_pointers.len() != n_residues {
        return Err(parse_error(
            "PRMTOP",
            format!(
                "residue metadata has {} names and {} pointers; expected {n_residues}",
                resnames.len(),
                residue_pointers.len()
            ),
        ));
    }
    let mut starts = Vec::with_capacity(n_residues + 1);
    for pointer in residue_pointers {
        let start = usize::try_from(pointer - 1)
            .map_err(|_| parse_error("PRMTOP", "RESIDUE_POINTER contains a non-positive value"))?;
        starts.push(start);
    }
    starts.push(n_atoms);
    if starts
        .windows(2)
        .any(|window| window[0] > window[1] || window[1] > n_atoms)
    {
        return Err(parse_error(
            "PRMTOP",
            "RESIDUE_POINTER values are out of order",
        ));
    }
    let residue_chain_ids = sections
        .get("RESIDUE_CHAINID")
        .map(parse_section_strings)
        .transpose()?
        .unwrap_or_default();
    let chain_ids = if residue_chain_ids.len() == n_residues {
        residue_chain_ids
    } else if residue_chain_ids.is_empty() {
        vec!["SYSTEM".to_owned(); n_residues]
    } else {
        // RESIDUE_CHAINID is a non-standard optional section.  Amber files
        // produced by older tools occasionally contain a partial list; in
        // that case retain the topology and use the conventional SYSTEM
        // segment rather than rejecting otherwise valid atom records.
        vec!["SYSTEM".to_owned(); n_residues]
    };
    let atomic_numbers = sections
        .get("ATOMIC_NUMBER")
        .map(parse_section_ints)
        .transpose()?
        .unwrap_or_default();
    if !atomic_numbers.is_empty() && atomic_numbers.len() != n_atoms {
        return Err(parse_error(
            "PRMTOP",
            "ATOMIC_NUMBER length does not match atoms",
        ));
    }
    let mut atoms = Vec::with_capacity(n_atoms);
    let mut segment_ids = Vec::<String>::new();
    for index in 0..n_atoms {
        let residue_index = starts
            .windows(2)
            .position(|window| index >= window[0] && index < window[1])
            .ok_or_else(|| parse_error("PRMTOP", "atom is not covered by residue pointers"))?;
        let segid = if chain_ids[residue_index].is_empty() {
            "SYSTEM".to_owned()
        } else {
            chain_ids[residue_index].clone()
        };
        if !segment_ids.iter().any(|id| id == &segid) {
            segment_ids.push(segid.clone());
        }
        let element = atomic_numbers
            .get(index)
            .and_then(|number| amber_atomic_symbol(*number));
        atoms.push(AmberTopAtom {
            index,
            name: names[index].clone(),
            atom_type: types[index].clone(),
            type_index: i32::try_from(type_indices[index]).unwrap_or(i32::MAX),
            charge: charges[index] / 18.2223,
            mass: masses[index],
            element,
            residue_index,
            resid: i32::try_from(residue_index + 1).unwrap_or(i32::MAX),
            resname: resnames[residue_index].clone(),
            segid: segid.clone(),
            chain_id: if chain_ids[residue_index].is_empty() {
                String::new()
            } else {
                chain_ids[residue_index].clone()
            },
        });
    }
    let bonds = parse_bond_sections(
        &sections,
        "BONDS_INC_HYDROGEN",
        "BONDS_WITHOUT_HYDROGEN",
        n_atoms,
    )?;
    let angles = parse_angle_sections(
        &sections,
        "ANGLES_INC_HYDROGEN",
        "ANGLES_WITHOUT_HYDROGEN",
        n_atoms,
    )?;
    let (dihedrals, impropers) = parse_dihedral_sections(
        &sections,
        "DIHEDRALS_INC_HYDROGEN",
        "DIHEDRALS_WITHOUT_HYDROGEN",
        n_atoms,
    )?;
    Ok(AmberTopFile {
        title,
        pointers,
        atoms,
        bonds,
        angles,
        dihedrals,
        impropers,
        residue_chain_ids: chain_ids,
    })
}

#[derive(Clone, Debug)]
struct TopSection {
    format: String,
    lines: Vec<String>,
}

fn parse_top_sections(lines: &[&str]) -> Result<HashMap<String, TopSection>, AmberError> {
    let mut sections = HashMap::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim_end();
        if !line.starts_with("%FLAG") {
            index += 1;
            continue;
        }
        let name = line[5..].trim();
        if name.is_empty() {
            return Err(parse_error("PRMTOP", "empty %FLAG name"));
        }
        index += 1;
        let start = index;
        while index < lines.len() && !lines[index].trim_start().starts_with("%FLAG") {
            index += 1;
        }
        let section_lines = &lines[start..index];
        let format = section_lines
            .iter()
            .find_map(|line| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("%FORMAT(")
                    .and_then(|format| format.strip_suffix(')'))
                    .map(str::to_owned)
            })
            .ok_or_else(|| parse_error("PRMTOP", format!("%FORMAT missing for {name}")))?;
        let data_lines = section_lines
            .iter()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("%FORMAT") && !trimmed.starts_with("%COMMENT")
            })
            .map(|line| (*line).to_owned())
            .collect();
        sections.insert(
            name.to_owned(),
            TopSection {
                format,
                lines: data_lines,
            },
        );
    }
    Ok(sections)
}

#[derive(Clone, Copy)]
enum TopFieldKind {
    Integer,
    Real,
    Text,
}

fn parse_format(format: &str) -> Result<(usize, usize, TopFieldKind), AmberError> {
    let kind_position = format
        .char_indices()
        .find(|(_, character)| {
            matches!(
                character,
                'I' | 'i' | 'E' | 'e' | 'F' | 'f' | 'D' | 'd' | 'A' | 'a'
            )
        })
        .map(|(index, _)| index)
        .ok_or_else(|| parse_error("PRMTOP", format!("unsupported %FORMAT({format})")))?;
    let count = if kind_position == 0 {
        1
    } else {
        format[..kind_position].parse::<usize>().map_err(|_| {
            parse_error(
                "PRMTOP",
                format!("invalid repeat count in %FORMAT({format})"),
            )
        })?
    };
    let kind_char = format.as_bytes()[kind_position] as char;
    let width_start = kind_position + 1;
    let width_end = format[width_start..]
        .find(|character: char| !character.is_ascii_digit())
        .map_or(format.len(), |index| width_start + index);
    let width = format[width_start..width_end]
        .parse::<usize>()
        .map_err(|_| {
            parse_error(
                "PRMTOP",
                format!("invalid field width in %FORMAT({format})"),
            )
        })?;
    if count == 0 || width == 0 {
        return Err(parse_error("PRMTOP", format!("invalid %FORMAT({format})")));
    }
    let kind = match kind_char.to_ascii_uppercase() {
        'I' => TopFieldKind::Integer,
        'E' | 'F' | 'D' => TopFieldKind::Real,
        'A' => TopFieldKind::Text,
        _ => unreachable!(),
    };
    Ok((count, width, kind))
}

fn parse_section_values<T>(
    section: &TopSection,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Vec<T>, AmberError> {
    let (count, width, kind) = parse_format(&section.format)?;
    let mut values = Vec::new();
    for line in &section.lines {
        let bytes = line.as_bytes();
        for field in 0..count {
            let start = field.saturating_mul(width);
            let end = start.saturating_add(width).min(bytes.len());
            if start >= bytes.len() {
                continue;
            }
            let value = std::str::from_utf8(&bytes[start..end]).unwrap_or("").trim();
            if value.is_empty() {
                continue;
            }
            let parsed = parse(value).ok_or_else(|| {
                let kind_name = match kind {
                    TopFieldKind::Integer => "integer",
                    TopFieldKind::Real => "real",
                    TopFieldKind::Text => "text",
                };
                parse_error(
                    "PRMTOP",
                    format!(
                        "invalid {kind_name} value {value:?} in %FORMAT({})",
                        section.format
                    ),
                )
            })?;
            values.push(parsed);
        }
    }
    Ok(values)
}

fn parse_section_ints(section: &TopSection) -> Result<Vec<i64>, AmberError> {
    let (kind, format) = (parse_format(&section.format)?.2, section.format.as_str());
    if !matches!(kind, TopFieldKind::Integer) {
        return Err(parse_error(
            "PRMTOP",
            format!("%FORMAT({format}) is not integer"),
        ));
    }
    parse_section_values(section, |value| value.parse::<i64>().ok())
}

fn parse_section_floats(section: &TopSection) -> Result<Vec<f64>, AmberError> {
    let (kind, format) = (parse_format(&section.format)?.2, section.format.as_str());
    if !matches!(kind, TopFieldKind::Real) {
        return Err(parse_error(
            "PRMTOP",
            format!("%FORMAT({format}) is not real"),
        ));
    }
    parse_section_values(section, |value| {
        value.replace(['D', 'd'], "E").parse::<f64>().ok()
    })
}

fn parse_section_strings(section: &TopSection) -> Result<Vec<String>, AmberError> {
    let (kind, format) = (parse_format(&section.format)?.2, section.format.as_str());
    if !matches!(kind, TopFieldKind::Text) {
        return Err(parse_error(
            "PRMTOP",
            format!("%FORMAT({format}) is not text"),
        ));
    }
    parse_section_values(section, |value| Some(value.to_owned()))
}

fn required_ints(
    sections: &HashMap<String, TopSection>,
    name: &str,
) -> Result<Vec<i64>, AmberError> {
    let section = sections
        .get(name)
        .ok_or_else(|| parse_error("PRMTOP", format!("{name} section is missing")))?;
    parse_section_ints(section)
}

fn required_floats(
    sections: &HashMap<String, TopSection>,
    name: &str,
) -> Result<Vec<f64>, AmberError> {
    let section = sections
        .get(name)
        .ok_or_else(|| parse_error("PRMTOP", format!("{name} section is missing")))?;
    parse_section_floats(section)
}

fn required_strings(
    sections: &HashMap<String, TopSection>,
    name: &str,
) -> Result<Vec<String>, AmberError> {
    let section = sections
        .get(name)
        .ok_or_else(|| parse_error("PRMTOP", format!("{name} section is missing")))?;
    parse_section_strings(section)
}

fn parse_bond_sections(
    sections: &HashMap<String, TopSection>,
    first: &str,
    second: &str,
    n_atoms: usize,
) -> Result<Vec<AmberBond>, AmberError> {
    let mut result = Vec::new();
    for name in [first, second] {
        let Some(section) = sections.get(name) else {
            continue;
        };
        let values = parse_section_ints(section)?;
        if values.len() % 3 != 0 {
            return Err(parse_error(
                "PRMTOP",
                format!("{name} is not a multiple of three"),
            ));
        }
        for chunk in values.as_chunks::<3>().0 {
            let atom1 = decode_atom_index(chunk[0], n_atoms)?;
            let atom2 = decode_atom_index(chunk[1], n_atoms)?;
            if atom1 != atom2
                && !result.iter().any(|bond: &AmberBond| {
                    (bond.atom1 == atom1 && bond.atom2 == atom2)
                        || (bond.atom1 == atom2 && bond.atom2 == atom1)
                })
            {
                result.push(AmberBond { atom1, atom2 });
            }
        }
    }
    Ok(result)
}

fn parse_angle_sections(
    sections: &HashMap<String, TopSection>,
    first: &str,
    second: &str,
    n_atoms: usize,
) -> Result<Vec<AmberAngle>, AmberError> {
    let mut result = Vec::new();
    for name in [first, second] {
        let Some(section) = sections.get(name) else {
            continue;
        };
        let values = parse_section_ints(section)?;
        if values.len() % 4 != 0 {
            return Err(parse_error(
                "PRMTOP",
                format!("{name} is not a multiple of four"),
            ));
        }
        for chunk in values.as_chunks::<4>().0 {
            result.push(AmberAngle {
                atom1: decode_atom_index(chunk[0], n_atoms)?,
                atom2: decode_atom_index(chunk[1], n_atoms)?,
                atom3: decode_atom_index(chunk[2], n_atoms)?,
            });
        }
    }
    Ok(result)
}

fn parse_dihedral_sections(
    sections: &HashMap<String, TopSection>,
    first: &str,
    second: &str,
    n_atoms: usize,
) -> Result<(Vec<AmberDihedral>, Vec<AmberImproper>), AmberError> {
    let mut dihedrals = BTreeSet::new();
    let mut impropers = Vec::new();
    for name in [first, second] {
        let Some(section) = sections.get(name) else {
            continue;
        };
        let values = parse_section_ints(section)?;
        if values.len() % 5 != 0 {
            return Err(parse_error(
                "PRMTOP",
                format!("{name} is not a multiple of five"),
            ));
        }
        for chunk in values.as_chunks::<5>().0 {
            let atom1 = decode_atom_index(chunk[0], n_atoms)?;
            let atom2 = decode_atom_index(chunk[1], n_atoms)?;
            let atom3 = decode_atom_index(chunk[2].abs(), n_atoms)?;
            let atom4 = decode_atom_index(chunk[3].abs(), n_atoms)?;
            if chunk[3] < 0 {
                impropers.push(AmberImproper {
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                });
            } else {
                dihedrals.insert(AmberDihedral {
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                });
            }
        }
    }
    Ok((dihedrals.into_iter().collect(), impropers))
}

fn decode_atom_index(value: i64, n_atoms: usize) -> Result<usize, AmberError> {
    if value < 0 {
        return Err(parse_error("PRMTOP", "bonded atom index is negative"));
    }
    let index = usize::try_from(value / 3)
        .map_err(|_| parse_error("PRMTOP", "bonded atom index overflows usize"))?;
    if index >= n_atoms {
        return Err(parse_error(
            "PRMTOP",
            format!("bonded atom index {index} is out of range"),
        ));
    }
    Ok(index)
}

fn positive_count(value: i64, name: &str) -> Result<usize, AmberError> {
    if value <= 0 {
        return Err(parse_error("PRMTOP", format!("{name} must be positive")));
    }
    usize::try_from(value).map_err(|_| parse_error("PRMTOP", format!("{name} overflows usize")))
}

fn positive_or_zero_count(value: i64, name: &str) -> Result<usize, AmberError> {
    if value < 0 {
        return Err(parse_error(
            "PRMTOP",
            format!("{name} must be non-negative"),
        ));
    }
    usize::try_from(value).map_err(|_| parse_error("PRMTOP", format!("{name} overflows usize")))
}

fn amber_atomic_symbol(number: i64) -> Option<String> {
    let symbol = match number {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        21 => "Sc",
        22 => "Ti",
        23 => "V",
        24 => "Cr",
        25 => "Mn",
        26 => "Fe",
        27 => "Co",
        28 => "Ni",
        29 => "Cu",
        30 => "Zn",
        31 => "Ga",
        32 => "Ge",
        33 => "As",
        34 => "Se",
        35 => "Br",
        36 => "Kr",
        37 => "Rb",
        38 => "Sr",
        39 => "Y",
        40 => "Zr",
        41 => "Nb",
        42 => "Mo",
        43 => "Tc",
        44 => "Ru",
        45 => "Rh",
        46 => "Pd",
        47 => "Ag",
        48 => "Cd",
        49 => "In",
        50 => "Sn",
        51 => "Sb",
        52 => "Te",
        53 => "I",
        54 => "Xe",
        55 => "Cs",
        56 => "Ba",
        _ => return None,
    };
    Some(symbol.to_owned())
}

impl Universe {
    /// Construct a universe from an Amber PRMTOP/PARM7/TOP topology.
    pub fn from_prmtop(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_amber_top_file(read_prmtop(path)?)
    }

    /// Construct a universe from an Amber topology held in memory.
    pub fn from_prmtop_str(input: &str) -> crate::Result<Self> {
        Self::from_amber_top_file(AmberTopFile::from_str(input)?)
    }

    /// Construct a universe from an Amber topology and restart coordinates.
    pub fn from_prmtop_and_inpcrd(
        topology_path: impl AsRef<Path>,
        coordinates_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop(topology_path)?;
        let coordinates = InpcrdFile::read(File::open(coordinates_path)?)?.coordinates;
        attach_amber_coordinates(&mut universe, coordinates)?;
        Ok(universe)
    }

    /// Construct a universe from an Amber topology and restart text.
    pub fn from_prmtop_and_inpcrd_str(topology: &str, coordinates: &str) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop_str(topology)?;
        let coordinates = InpcrdFile::from_str(coordinates)?.coordinates;
        attach_amber_coordinates(&mut universe, coordinates)?;
        Ok(universe)
    }

    /// Construct a universe directly from parsed Amber topology data.
    pub fn from_amber_top_file(file: AmberTopFile) -> crate::Result<Self> {
        let atoms = file
            .atoms
            .iter()
            .map(|source| {
                let mut atom = Atom::new(source.index, source.name.clone(), [0.0; 3]);
                atom.atom_type = Some(source.atom_type.clone());
                atom.element = source.element.clone();
                atom.mass = source.mass;
                atom.charge = source.charge;
                atom.resid = source.resid;
                atom.resname = source.resname.clone();
                atom.segid = source.segid.clone();
                atom.chain_id = source.chain_id.clone();
                atom
            })
            .collect::<Vec<_>>();
        let mut topology = Topology::new(atoms);
        for source in &file.bonds {
            topology.add_bond(Bond::new(source.atom1, source.atom2));
        }
        Ok(Self::new(topology))
    }
}

fn attach_amber_coordinates(
    universe: &mut Universe,
    coordinates: CoordinateFile,
) -> crate::Result<()> {
    if coordinates.frames.is_empty() {
        return Err(crate::Error::InvalidInput(
            "Amber coordinate file contains no frames".to_owned(),
        ));
    }
    if coordinates
        .frames
        .iter()
        .any(|frame| frame.n_atoms() != universe.n_atoms())
    {
        return Err(crate::Error::InvalidInput(format!(
            "Amber coordinate file contains {} atoms, topology contains {}",
            coordinates.n_atoms(),
            universe.n_atoms()
        )));
    }
    universe.trajectory = Trajectory::new(
        coordinates
            .frames
            .into_iter()
            .enumerate()
            .map(|(step, source)| {
                let mut frame = Frame::new(source.positions);
                frame.velocities = source.velocities;
                frame.dimensions = source.dimensions;
                frame.time = source.time;
                frame.step = step;
                frame
            })
            .collect(),
    );
    Ok(())
}

impl InpcrdFile {
    /// Read an Amber restart from any text reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, AmberError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Parse an Amber restart held in memory.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, AmberError> {
        parse_inpcrd(input)
    }

    /// Write this restart in the conventional fixed-width Amber layout.
    pub fn write<W: Write>(&self, writer: W) -> Result<(), AmberError> {
        write_inpcrd_document(self, writer)
    }

    /// Serialize this restart to text.
    pub fn to_string(&self) -> Result<String, AmberError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output).map_err(|error| {
            AmberError::InvalidStructure(format!("restart output is not UTF-8: {error}"))
        })
    }
}

impl NamdBinFile {
    /// Read a NAMD binary coordinate file from any byte reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, AmberError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a NAMD binary coordinate file held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AmberError> {
        if bytes.len() < 4 {
            return Err(parse_error(
                "NAMDBIN",
                "file is shorter than atom-count header",
            ));
        }
        let little_count = i32::from_le_bytes(bytes[..4].try_into().expect("four-byte slice"));
        let (count, little_endian) = if little_count > 0
            && expected_binary_size(little_count as usize) == Some(bytes.len())
        {
            (little_count as usize, true)
        } else {
            let big_count = i32::from_be_bytes(bytes[..4].try_into().expect("four-byte slice"));
            if big_count <= 0 || expected_binary_size(big_count as usize) != Some(bytes.len()) {
                return Err(parse_error("NAMDBIN", "atom count or file size is invalid"));
            }
            (big_count as usize, false)
        };
        let mut positions = Vec::with_capacity(count);
        let mut offset = 4;
        for _ in 0..count {
            let mut position = [0.0; 3];
            for value in &mut position {
                let bytes: [u8; 8] = bytes[offset..offset + 8]
                    .try_into()
                    .map_err(|_| parse_error("NAMDBIN", "coordinate payload is truncated"))?;
                *value = if little_endian {
                    f64::from_le_bytes(bytes)
                } else {
                    f64::from_be_bytes(bytes)
                };
                if !value.is_finite() {
                    return Err(parse_error("NAMDBIN", "coordinates must be finite"));
                }
                offset += 8;
            }
            positions.push(position);
        }
        Ok(Self {
            coordinates: CoordinateFile::new(vec![CoordinateFrame::new(positions)]),
        })
    }

    /// Write the coordinate frame in little-endian NAMD format.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), AmberError> {
        let frame =
            self.coordinates.frames.first().ok_or_else(|| {
                AmberError::InvalidStructure("NAMDBIN requires one frame".to_owned())
            })?;
        if frame.n_atoms() == 0 {
            return Err(AmberError::InvalidStructure(
                "NAMDBIN requires at least one atom".to_owned(),
            ));
        }
        if self.coordinates.frames.len() != 1
            || frame
                .positions
                .iter()
                .flat_map(|position| position.iter())
                .any(|value| !value.is_finite())
        {
            return Err(AmberError::InvalidStructure(
                "NAMDBIN contains one finite coordinate frame".to_owned(),
            ));
        }
        let count = i32::try_from(frame.n_atoms())
            .map_err(|_| AmberError::InvalidStructure("too many atoms".to_owned()))?;
        writer.write_all(&count.to_le_bytes())?;
        for position in &frame.positions {
            for value in position {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Serialize to little-endian bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AmberError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes)?;
        Ok(bytes)
    }
}

/// Read an Amber restart from a path.
pub fn read_inpcrd<P: AsRef<Path>>(path: P) -> Result<InpcrdFile, AmberError> {
    InpcrdFile::read(File::open(path)?)
}

/// Write an Amber restart to a path.
pub fn write_inpcrd<P: AsRef<Path>>(path: P, file: &InpcrdFile) -> Result<(), AmberError> {
    file.write(File::create(path)?)
}

/// Read a NAMD binary coordinate file from a path.
pub fn read_namdbin<P: AsRef<Path>>(path: P) -> Result<NamdBinFile, AmberError> {
    NamdBinFile::read(File::open(path)?)
}

/// Write a NAMD binary coordinate file to a path.
pub fn write_namdbin<P: AsRef<Path>>(path: P, file: &NamdBinFile) -> Result<(), AmberError> {
    file.write(File::create(path)?)
}

impl CoordinateFile {
    /// Read Amber restart coordinates, preserving title/time and optional data.
    pub fn read_inpcrd<R: Read>(reader: R) -> Result<Self, AmberError> {
        Ok(InpcrdFile::read(reader)?.coordinates)
    }

    /// Write this one-frame coordinate file in Amber restart format.
    pub fn write_inpcrd<W: Write>(
        &self,
        writer: W,
        title: impl Into<String>,
        time: Option<f64>,
    ) -> Result<(), AmberError> {
        let file = InpcrdFile {
            title: title.into(),
            time,
            coordinates: self.clone(),
        };
        file.write(writer)
    }

    /// Read NAMD binary coordinates from bytes.
    pub fn from_namdbin_bytes(bytes: &[u8]) -> Result<Self, AmberError> {
        Ok(NamdBinFile::from_bytes(bytes)?.coordinates)
    }

    /// Write this one-frame coordinate file in NAMD binary format.
    pub fn write_namdbin<W: Write>(&self, writer: W) -> Result<(), AmberError> {
        NamdBinFile {
            coordinates: self.clone(),
        }
        .write(writer)
    }
}

fn parse_inpcrd(input: &str) -> Result<InpcrdFile, AmberError> {
    let mut lines = input.lines();
    let title = lines.next().unwrap_or_default().trim_end().to_owned();
    let header = lines
        .next()
        .ok_or_else(|| parse_error("INPCRD", "missing atom-count line"))?;
    let mut header_values = header.split_whitespace();
    let atom_count = header_values
        .next()
        .ok_or_else(|| parse_error("INPCRD", "missing atom count"))?
        .parse::<usize>()
        .map_err(|_| parse_error("INPCRD", "atom count is not a non-negative integer"))?;
    if atom_count == 0 {
        return Err(parse_error("INPCRD", "atom count must be positive"));
    }
    let time = header_values
        .next()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| parse_error("INPCRD", "time is not a valid number"))
        })
        .transpose()?;
    let rest = lines.collect::<Vec<_>>().join("\n");
    let values = rest
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| parse_error("INPCRD", "coordinate payload contains an invalid number"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coordinate_values = atom_count
        .checked_mul(3)
        .ok_or_else(|| parse_error("INPCRD", "atom count overflows coordinate size"))?;
    if values.len() < coordinate_values {
        return Err(parse_error(
            "INPCRD",
            format!(
                "file contains {} coordinate values; expected {coordinate_values}",
                values.len()
            ),
        ));
    }
    let mut positions = Vec::with_capacity(atom_count);
    for chunk in values[..coordinate_values].chunks(3) {
        if chunk.iter().any(|value| !value.is_finite()) {
            return Err(parse_error("INPCRD", "coordinates must be finite"));
        }
        positions.push([chunk[0], chunk[1], chunk[2]]);
    }
    let remaining = &values[coordinate_values..];
    let with_box_and_velocities = coordinate_values.checked_add(6);
    let (dimensions, velocity_values) =
        if remaining.len() == 6 || with_box_and_velocities == Some(values.len()) {
            let box_values = remaining
                .get(..6)
                .ok_or_else(|| parse_error("INPCRD", "unit-cell payload is truncated"))?;
            let dimensions = box_values.try_into().expect("six-value slice");
            (Some(dimensions), &remaining[6..])
        } else {
            (None, remaining)
        };
    let velocities = if velocity_values.is_empty() {
        None
    } else {
        if velocity_values.len() != coordinate_values {
            return Err(parse_error(
                "INPCRD",
                "trailing payload is neither a unit cell nor a complete velocity array",
            ));
        }
        Some(
            velocity_values
                .chunks(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect(),
        )
    };
    let mut frame = CoordinateFrame::new(positions);
    frame.title.clone_from(&title);
    frame.time = time.unwrap_or(0.0);
    frame.dimensions = dimensions;
    frame.velocities = velocities;
    Ok(InpcrdFile {
        title,
        time,
        coordinates: CoordinateFile::new(vec![frame]),
    })
}

fn write_inpcrd_document<W: Write>(file: &InpcrdFile, mut writer: W) -> Result<(), AmberError> {
    let frame = file
        .coordinates
        .frames
        .first()
        .ok_or_else(|| AmberError::InvalidStructure("INPCRD requires one frame".to_owned()))?;
    if file.coordinates.frames.len() != 1 || frame.n_atoms() == 0 {
        return Err(AmberError::InvalidStructure(
            "INPCRD requires exactly one non-empty frame".to_owned(),
        ));
    }
    if frame
        .positions
        .iter()
        .flat_map(|position| position.iter())
        .any(|value| !value.is_finite())
    {
        return Err(AmberError::InvalidStructure(
            "coordinates must be finite".to_owned(),
        ));
    }
    if let Some(velocities) = &frame.velocities
        && (velocities.len() != frame.n_atoms()
            || velocities
                .iter()
                .flat_map(|velocity| velocity.iter())
                .any(|value| !value.is_finite()))
    {
        return Err(AmberError::InvalidStructure(
            "velocities must match the coordinate count and be finite".to_owned(),
        ));
    }
    writeln!(writer, "{}", file.title)?;
    if let Some(time) = file.time {
        if !time.is_finite() {
            return Err(AmberError::InvalidStructure(
                "time must be finite".to_owned(),
            ));
        }
        writeln!(writer, "{:>6} {:>15.7E}", frame.n_atoms(), time)?;
    } else {
        writeln!(writer, "{:>6}", frame.n_atoms())?;
    }
    let mut values = frame
        .positions
        .iter()
        .flat_map(|position| position.iter().copied());
    write_fixed_values(&mut writer, &mut values)?;
    if let Some(dimensions) = frame.dimensions {
        if dimensions
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AmberError::InvalidStructure(
                "unit-cell values must be finite and positive".to_owned(),
            ));
        }
        let mut values = dimensions.into_iter();
        write_fixed_values(&mut writer, &mut values)?;
    }
    if let Some(velocities) = &frame.velocities {
        let mut values = velocities
            .iter()
            .flat_map(|velocity| velocity.iter().copied());
        write_fixed_values(&mut writer, &mut values)?;
    }
    Ok(())
}

fn write_fixed_values<W: Write, I: Iterator<Item = f64>>(
    writer: &mut W,
    values: &mut I,
) -> Result<(), AmberError> {
    let mut count = 0;
    for value in values {
        if !value.is_finite() {
            return Err(AmberError::InvalidStructure(
                "numeric values must be finite".to_owned(),
            ));
        }
        write!(writer, "{:12.7}", value)?;
        count += 1;
        if count == 6 {
            writer.write_all(b"\n")?;
            count = 0;
        }
    }
    if count != 0 {
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn expected_binary_size(count: usize) -> Option<usize> {
    count.checked_mul(24)?.checked_add(4)
}

fn parse_error(format: &'static str, message: impl Into<String>) -> AmberError {
    AmberError::Parse {
        format,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/Amber")
            .join(name)
    }

    #[test]
    fn inpcrd_reads_fixture_and_round_trips() {
        let input =
            include_str!("../../mdanalysis/testsuite/MDAnalysisTests/data/Amber/test.inpcrd");
        let file = InpcrdFile::from_str(input).unwrap();
        assert_eq!(file.coordinates.n_atoms(), 5);
        assert_eq!(file.title, "ACE");
        assert_eq!(file.time, Some(30.0));
        assert!((file.coordinates.frames[0].positions[0][0] - 6.6528795).abs() < 1.0e-7);
        let serialized = file.to_string().unwrap();
        let parsed = InpcrdFile::from_str(&serialized).unwrap();
        assert_eq!(parsed.coordinates.n_atoms(), 5);
        assert!((parsed.coordinates.frames[0].positions[4][2] + 7.9729560).abs() < 1.0e-6);
    }

    #[test]
    fn namdbin_reads_fixture_and_round_trips() {
        let bytes = include_bytes!("../../mdanalysis/testsuite/MDAnalysisTests/data/adk_open.coor");
        let file = NamdBinFile::from_bytes(bytes).unwrap();
        assert_eq!(file.coordinates.n_atoms(), 3341);
        assert!((file.coordinates.frames[0].positions[0][0] - 1.0).abs() < 100.0);
        let output = file.to_bytes().unwrap();
        let parsed = NamdBinFile::from_bytes(&output).unwrap();
        assert_eq!(parsed.coordinates, file.coordinates);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        assert!(InpcrdFile::from_str("title\n2\n0 0 0\n").is_err());
        assert!(NamdBinFile::from_bytes(&[1, 0, 0, 0]).is_err());
    }

    #[test]
    fn prmtop_reads_core_sections_and_connectivity() {
        let file = AmberTopFile::read_file(fixture("ache.prmtop")).unwrap();
        assert_eq!(file.n_atoms(), 252);
        assert_eq!(file.n_residues(), 14);
        assert_eq!(file.bonds.len(), 259);
        assert_eq!(file.angles.len(), 456);
        assert_eq!(file.dihedrals.len(), 673);
        assert_eq!(file.impropers.len(), 66);
        assert_eq!(file.atoms[0].name, "N");
        assert_eq!(file.atoms[0].element, None);
        assert!((file.atoms[0].charge - 2.57663322 / 18.2223).abs() < 1.0e-8);
        assert!(file.bonds.contains(&AmberBond { atom1: 0, atom2: 4 }));
        assert!(file.angles.contains(&AmberAngle {
            atom1: 0,
            atom2: 4,
            atom3: 6
        }));
    }

    #[test]
    fn prmtop_reads_bzip2_chain_ids_and_elements() {
        let file = AmberTopFile::read_file(fixture("ache_chainid.prmtop.bz2")).unwrap();
        assert_eq!(file.n_atoms(), 677);
        assert_eq!(file.n_residues(), 38);
        assert_eq!(file.n_segments(), 3);
        assert_eq!(file.atoms[0].chain_id, "A");
        assert_eq!(file.atoms[250].element.as_deref(), Some("O"));
        assert_eq!(file.atoms[500].element.as_deref(), Some("H"));
    }

    #[test]
    fn prmtop_universe_constructor_attaches_restart() {
        let topology = AmberTopFile::read_file(fixture("ace_mbondi3.parm7")).unwrap();
        let universe = Universe::from_amber_top_file(topology).unwrap();
        assert_eq!(universe.n_atoms(), 6);
        assert_eq!(universe.n_residues(), 1);
        assert_eq!(universe.n_segments(), 1);
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("H"));
        assert_eq!(universe.topology.bonds.len(), 5);
    }

    #[test]
    fn prmtop_universe_preserves_missing_elements() {
        let topology = AmberTopFile::read_file(fixture("ache.prmtop")).unwrap();
        let universe = Universe::from_amber_top_file(topology).unwrap();
        assert!(
            universe
                .topology
                .atoms
                .iter()
                .all(|atom| atom.element.is_none())
        );
    }

    #[test]
    fn parses_large_parm7_and_legacy_topology() {
        let parm7 = AmberTopFile::read_file(fixture("tz2.truncoct.parm7.bz2")).unwrap();
        assert_eq!(parm7.n_atoms(), 5827);
        assert_eq!(parm7.n_residues(), 1882);
        assert_eq!(parm7.bonds.len(), 5834);
        assert_eq!(parm7.angles.len(), 402);
        assert_eq!(parm7.dihedrals.len(), 602);
        assert_eq!(parm7.impropers.len(), 55);

        let top = AmberTopFile::read_file(fixture("anti.top")).unwrap();
        assert_eq!(top.n_atoms(), 8923);
        assert_eq!(top.n_residues(), 2861);
        assert_eq!(top.bonds.len(), 8947);
        assert_eq!(top.angles.len(), 756);
        assert_eq!(top.dihedrals.len(), 1128);
        assert_eq!(top.impropers.len(), 72);
    }

    #[test]
    fn ignores_partial_residue_chain_id_section() {
        let bytes = std::fs::read(fixture("ache_chainid.error5.prmtop.bz2")).unwrap();
        let file = AmberTopFile::from_bytes(&bytes).unwrap();
        assert_eq!(file.n_residues(), 38);
        assert!(file.atoms.iter().all(|atom| atom.segid == "SYSTEM"));
    }

    #[test]
    fn rejects_malformed_numeric_section_values() {
        let section = TopSection {
            format: "2I4".to_owned(),
            lines: vec!["   1xxxx".to_owned()],
        };
        let error = parse_section_ints(&section).unwrap_err();
        assert!(error.to_string().contains("invalid integer value"));
    }

    #[test]
    fn accepts_trailing_blank_fields_in_numeric_sections() {
        let section = TopSection {
            format: "2I4".to_owned(),
            lines: vec!["   1   ".to_owned()],
        };
        assert_eq!(parse_section_ints(&section).unwrap(), vec![1]);
    }
}
