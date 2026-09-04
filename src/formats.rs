//! Readers and writers for common text molecular formats.
//!
//! The types in this module intentionally contain only fields shared by the
//! formats supported here.  Format-specific details (for example a MOL2 bond
//! type) are retained where they can be represented without making the
//! structure format-dependent.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

/// A format-independent atom record.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatAtom {
    /// Serial number as written by the source format.
    pub serial: usize,
    /// Atom name (for example `CA`).
    pub name: String,
    /// Optional force-field atom type (used by MOL2).
    pub atom_type: Option<String>,
    /// Residue name.
    pub residue_name: String,
    /// Residue number.
    pub residue_id: i32,
    /// Optional chain identifier.
    pub chain_id: Option<String>,
    /// Optional segment/substructure identifier.
    pub segment_id: Option<String>,
    /// Cartesian coordinates in the units used by the source file.
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Partial charge, when supplied by the format.
    pub charge: Option<f64>,
    /// Atomic radius, when supplied by the format (PQR).
    pub radius: Option<f64>,
}

impl Default for FormatAtom {
    fn default() -> Self {
        Self {
            serial: 1,
            name: "X".to_owned(),
            atom_type: None,
            residue_name: "UNK".to_owned(),
            residue_id: 1,
            chain_id: None,
            segment_id: None,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            charge: None,
            radius: None,
        }
    }
}

impl FormatAtom {
    /// Construct an atom with the minimum required metadata.
    #[must_use]
    pub fn new(serial: usize, name: impl Into<String>, position: [f64; 3]) -> Self {
        Self {
            serial,
            name: name.into(),
            x: position[0],
            y: position[1],
            z: position[2],
            ..Self::default()
        }
    }

    /// Return the Cartesian position as an array.
    #[must_use]
    pub const fn position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Alias for callers using the conventional `resid` spelling.
    #[must_use]
    pub const fn resid(&self) -> i32 {
        self.residue_id
    }

    /// Alias for callers using the conventional `resname` spelling.
    #[must_use]
    pub fn resname(&self) -> &str {
        &self.residue_name
    }
}

/// A bond in a MOL2 structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatBond {
    /// Serial number in the MOL2 bond section.
    pub serial: usize,
    /// Source atom serial number.
    pub atom1: usize,
    /// Destination atom serial number.
    pub atom2: usize,
    /// MOL2 bond type (`1`, `ar`, `am`, and so on).
    pub bond_type: String,
}

impl FormatBond {
    #[must_use]
    pub fn new(serial: usize, atom1: usize, atom2: usize, bond_type: impl Into<String>) -> Self {
        Self {
            serial,
            atom1,
            atom2,
            bond_type: bond_type.into(),
        }
    }
}

/// A molecular structure read from one of the supported text formats.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Structure {
    /// Atom records in file order.
    pub atoms: Vec<FormatAtom>,
    /// Bond records (MOL2 only; empty for PQR/CRD).
    pub bonds: Vec<FormatBond>,
    /// Optional title (MOL2 molecule title or CRD title).
    pub title: String,
    /// Optional unit-cell dimensions as `[a, b, c, alpha, beta, gamma]`.
    /// MOL2 stores these in its `@<TRIPOS>CRYSIN` section.
    pub dimensions: Option<[f64; 6]>,
}

impl Structure {
    /// Parse a PQR document from a string.
    pub fn from_pqr_str(input: &str) -> Result<Self, FormatError> {
        parse_pqr(input)
    }

    /// Read a PQR document from a reader.
    pub fn read_pqr<R: Read>(reader: R) -> Result<Self, FormatError> {
        read_text(reader, parse_pqr)
    }

    /// Write this structure as PQR to a writer.
    pub fn write_pqr<W: Write>(&self, writer: W) -> Result<(), FormatError> {
        write_pqr_document(self, writer)
    }

    /// Serialize this structure as PQR.
    pub fn to_pqr_string(&self) -> Result<String, FormatError> {
        let mut bytes = Vec::new();
        self.write_pqr(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Parse a MOL2 document from a string.
    pub fn from_mol2_str(input: &str) -> Result<Self, FormatError> {
        parse_mol2(input)
    }

    /// Read a MOL2 document from a reader.
    pub fn read_mol2<R: Read>(reader: R) -> Result<Self, FormatError> {
        read_text(reader, parse_mol2)
    }

    /// Write this structure as MOL2 to a writer.
    pub fn write_mol2<W: Write>(&self, writer: W) -> Result<(), FormatError> {
        write_mol2_document(self, writer)
    }

    /// Serialize this structure as MOL2.
    pub fn to_mol2_string(&self) -> Result<String, FormatError> {
        let mut bytes = Vec::new();
        self.write_mol2(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Parse a CRD/CARD document from a string.
    pub fn from_crd_str(input: &str) -> Result<Self, FormatError> {
        parse_crd(input)
    }

    /// Read a CRD/CARD document from a reader.
    pub fn read_crd<R: Read>(reader: R) -> Result<Self, FormatError> {
        read_text(reader, parse_crd)
    }

    /// Write this structure as a standard or extended CRD document.
    pub fn write_crd<W: Write>(&self, writer: W) -> Result<(), FormatError> {
        write_crd_document(self, writer)
    }

    /// Serialize this structure as a CRD document.
    pub fn to_crd_string(&self) -> Result<String, FormatError> {
        let mut bytes = Vec::new();
        self.write_crd(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Errors produced while parsing or writing text molecular formats.
#[derive(Debug)]
pub enum FormatError {
    Io(io::Error),
    Parse {
        format: &'static str,
        line: usize,
        message: String,
    },
    InvalidStructure(String),
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse {
                format,
                line,
                message,
            } => write!(formatter, "{format} parse error on line {line}: {message}"),
            Self::InvalidStructure(message) => write!(formatter, "invalid structure: {message}"),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<io::Error> for FormatError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Read a PQR structure from a filesystem path.
pub fn read_pqr<P: AsRef<Path>>(path: P) -> Result<Structure, FormatError> {
    Structure::from_pqr_str(&crate::io_utils::read_text_file(path.as_ref())?)
}

/// Write a PQR structure to a filesystem path.
pub fn write_pqr<P: AsRef<Path>>(path: P, structure: &Structure) -> Result<(), FormatError> {
    structure.write_pqr(File::create(path)?)
}

/// Read a MOL2 structure from a filesystem path.
pub fn read_mol2<P: AsRef<Path>>(path: P) -> Result<Structure, FormatError> {
    Structure::from_mol2_str(&crate::io_utils::read_text_file(path.as_ref())?)
}

/// Write a MOL2 structure to a filesystem path.
pub fn write_mol2<P: AsRef<Path>>(path: P, structure: &Structure) -> Result<(), FormatError> {
    structure.write_mol2(File::create(path)?)
}

/// Read a CRD/CARD structure from a filesystem path.
pub fn read_crd<P: AsRef<Path>>(path: P) -> Result<Structure, FormatError> {
    Structure::from_crd_str(&crate::io_utils::read_text_file(path.as_ref())?)
}

/// Write a CRD/CARD structure to a filesystem path.
pub fn write_crd<P: AsRef<Path>>(path: P, structure: &Structure) -> Result<(), FormatError> {
    structure.write_crd(File::create(path)?)
}

fn read_text<R: Read>(
    reader: R,
    parser: fn(&str) -> Result<Structure, FormatError>,
) -> Result<Structure, FormatError> {
    let mut input = String::new();
    BufReader::new(reader).read_to_string(&mut input)?;
    parser(&input)
}

fn parse_pqr(input: &str) -> Result<Structure, FormatError> {
    let mut structure = Structure::default();
    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let record = trimmed.split_whitespace().next().unwrap_or_default();
        if !matches!(record, "ATOM" | "HETATM") {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < 10 {
            return Err(parse_error(
                "PQR",
                line_number,
                "atom record requires serial, names, coordinates, charge, and radius",
            ));
        }
        let serial = parse_token::<usize>(tokens[1], "atom serial", "PQR", line_number)?;
        let name = nonempty(tokens[2], "atom name", "PQR", line_number)?;
        let residue_name = nonempty(tokens[3], "residue name", "PQR", line_number)?;
        let (chain_id, residue_index) = if let Ok(residue_id) = tokens[4].parse::<i32>() {
            (None, residue_id)
        } else {
            if tokens.len() < 11 {
                return Err(parse_error("PQR", line_number, "missing residue number"));
            }
            (
                Some(tokens[4].to_owned()),
                parse_token(tokens[5], "residue number", "PQR", line_number)?,
            )
        };
        let coordinate_start = if chain_id.is_some() { 6 } else { 5 };
        let x = parse_token(tokens[coordinate_start], "x coordinate", "PQR", line_number)?;
        let y = parse_token(
            tokens[coordinate_start + 1],
            "y coordinate",
            "PQR",
            line_number,
        )?;
        let z = parse_token(
            tokens[coordinate_start + 2],
            "z coordinate",
            "PQR",
            line_number,
        )?;
        let charge = parse_token(tokens[coordinate_start + 3], "charge", "PQR", line_number)?;
        let radius = parse_token(tokens[coordinate_start + 4], "radius", "PQR", line_number)?;
        structure.atoms.push(FormatAtom {
            serial,
            name,
            residue_name,
            residue_id: residue_index,
            chain_id,
            segment_id: tokens
                .get(coordinate_start + 5)
                .map(|value| (*value).to_owned()),
            atom_type: None,
            x,
            y,
            z,
            charge: Some(charge),
            radius: Some(radius),
        });
    }
    Ok(structure)
}

fn write_pqr_document<W: Write>(structure: &Structure, mut writer: W) -> Result<(), FormatError> {
    validate_atoms(structure, "PQR")?;
    for atom in &structure.atoms {
        let chain = atom.chain_id.as_deref().unwrap_or("");
        let segment = atom.segment_id.as_deref().unwrap_or("");
        let charge = atom.charge.ok_or_else(|| {
            FormatError::InvalidStructure(format!("atom {} has no charge", atom.serial))
        })?;
        let radius = atom.radius.ok_or_else(|| {
            FormatError::InvalidStructure(format!("atom {} has no radius", atom.serial))
        })?;
        if chain.is_empty() && segment.is_empty() {
            writeln!(
                writer,
                "ATOM {:>5} {:<4} {:<4} {:>5} {:>10.5} {:>10.5} {:>10.5} {:>9.5} {:>8.5}",
                atom.serial,
                atom.name,
                atom.residue_name,
                atom.residue_id,
                atom.x,
                atom.y,
                atom.z,
                charge,
                radius
            )?;
        } else if chain.is_empty() {
            writeln!(
                writer,
                "ATOM {:>5} {:<4} {:<4} {:>5} {:>10.5} {:>10.5} {:>10.5} {:>9.5} {:>8.5} {}",
                atom.serial,
                atom.name,
                atom.residue_name,
                atom.residue_id,
                atom.x,
                atom.y,
                atom.z,
                charge,
                radius,
                segment
            )?;
        } else if segment.is_empty() {
            writeln!(
                writer,
                "ATOM {:>5} {:<4} {:<4} {:<2} {:>5} {:>10.5} {:>10.5} {:>10.5} {:>9.5} {:>8.5}",
                atom.serial,
                atom.name,
                atom.residue_name,
                chain,
                atom.residue_id,
                atom.x,
                atom.y,
                atom.z,
                charge,
                radius
            )?;
        } else {
            writeln!(
                writer,
                "ATOM {:>5} {:<4} {:<4} {:<2} {:>5} {:>10.5} {:>10.5} {:>10.5} {:>9.5} {:>8.5} {}",
                atom.serial,
                atom.name,
                atom.residue_name,
                chain,
                atom.residue_id,
                atom.x,
                atom.y,
                atom.z,
                charge,
                radius,
                segment
            )?;
        }
    }
    writer.write_all(b"END\n")?;
    Ok(())
}

fn parse_mol2(input: &str) -> Result<Structure, FormatError> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Section {
        None,
        Molecule,
        Atom,
        Bond,
        CrysIn,
        Other,
    }
    let mut section = Section::None;
    let mut structure = Structure::default();
    let mut expected_atoms: Option<usize> = None;
    let mut expected_bonds: Option<usize> = None;
    let mut seen_molecule = false;
    let mut molecule_data_line = 0;

    for (index, raw_line) in input.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(section_name) = line.strip_prefix("@<TRIPOS>") {
            section = match section_name.trim().to_ascii_uppercase().as_str() {
                "MOLECULE" => Section::Molecule,
                "ATOM" => Section::Atom,
                "BOND" => Section::Bond,
                "CRYSIN" => Section::CrysIn,
                _ => Section::Other,
            };
            continue;
        }
        match section {
            Section::Molecule => {
                if !seen_molecule {
                    structure.title = line.to_owned();
                    seen_molecule = true;
                    molecule_data_line = line_number;
                } else if expected_atoms.is_none() {
                    let tokens: Vec<&str> = line.split_whitespace().collect();
                    if tokens.len() < 2 {
                        return Err(parse_error(
                            "MOL2",
                            line_number,
                            "molecule counts require atoms and bonds",
                        ));
                    }
                    expected_atoms =
                        Some(parse_token(tokens[0], "atom count", "MOL2", line_number)?);
                    expected_bonds =
                        Some(parse_token(tokens[1], "bond count", "MOL2", line_number)?);
                }
            }
            Section::Atom => {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() < 6 {
                    return Err(parse_error(
                        "MOL2",
                        line_number,
                        "atom record requires at least six fields",
                    ));
                }
                let serial = parse_token(tokens[0], "atom id", "MOL2", line_number)?;
                let name = nonempty(tokens[1], "atom name", "MOL2", line_number)?;
                let x = parse_token(tokens[2], "x coordinate", "MOL2", line_number)?;
                let y = parse_token(tokens[3], "y coordinate", "MOL2", line_number)?;
                let z = parse_token(tokens[4], "z coordinate", "MOL2", line_number)?;
                let atom_type = Some(tokens[5].to_owned());
                let residue_id = match tokens.get(6) {
                    Some(value) => parse_token(value, "substructure id", "MOL2", line_number)?,
                    None => 1,
                };
                let residue_name = tokens
                    .get(7)
                    .map_or_else(|| "UNK".to_owned(), |value| (*value).to_owned());
                let charge = tokens
                    .get(8)
                    .map(|value| parse_token(value, "charge", "MOL2", line_number))
                    .transpose()?;
                structure.atoms.push(FormatAtom {
                    serial,
                    name,
                    atom_type,
                    residue_name,
                    residue_id,
                    chain_id: None,
                    segment_id: None,
                    x,
                    y,
                    z,
                    charge,
                    radius: None,
                });
            }
            Section::Bond => {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() < 4 {
                    return Err(parse_error(
                        "MOL2",
                        line_number,
                        "bond record requires id, two atom ids, and type",
                    ));
                }
                structure.bonds.push(FormatBond {
                    serial: parse_token(tokens[0], "bond id", "MOL2", line_number)?,
                    atom1: parse_token(tokens[1], "first atom id", "MOL2", line_number)?,
                    atom2: parse_token(tokens[2], "second atom id", "MOL2", line_number)?,
                    bond_type: tokens[3].to_owned(),
                });
            }
            Section::CrysIn => {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() < 6 {
                    return Err(parse_error(
                        "MOL2",
                        line_number,
                        "CRYSIN record requires six unit-cell values",
                    ));
                }
                let mut dimensions = [0.0_f64; 6];
                for (index, value) in dimensions.iter_mut().enumerate() {
                    *value = parse_token(tokens[index], "unit-cell value", "MOL2", line_number)?;
                }
                if dimensions
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(parse_error(
                        "MOL2",
                        line_number,
                        "unit-cell values must be finite and positive",
                    ));
                }
                structure.dimensions = Some(dimensions);
            }
            Section::None | Section::Other => {}
        }
    }
    if !seen_molecule {
        return Err(parse_error("MOL2", 1, "missing @<TRIPOS>MOLECULE section"));
    }
    if let Some(expected) = expected_atoms
        && expected != structure.atoms.len()
    {
        return Err(parse_error(
            "MOL2",
            molecule_data_line,
            format!(
                "declared {expected} atoms but found {}",
                structure.atoms.len()
            ),
        ));
    }
    if let Some(expected) = expected_bonds
        && expected != structure.bonds.len()
    {
        return Err(parse_error(
            "MOL2",
            molecule_data_line,
            format!(
                "declared {expected} bonds but found {}",
                structure.bonds.len()
            ),
        ));
    }
    let atom_ids: std::collections::HashSet<usize> =
        structure.atoms.iter().map(|atom| atom.serial).collect();
    for bond in &structure.bonds {
        if !atom_ids.contains(&bond.atom1) || !atom_ids.contains(&bond.atom2) {
            return Err(FormatError::InvalidStructure(format!(
                "MOL2 bond {} references an unknown atom",
                bond.serial
            )));
        }
    }
    Ok(structure)
}

fn write_mol2_document<W: Write>(structure: &Structure, mut writer: W) -> Result<(), FormatError> {
    validate_atoms(structure, "MOL2")?;
    let title = if structure.title.is_empty() {
        "mdanalysis-rs"
    } else {
        &structure.title
    };
    writeln!(writer, "@<TRIPOS>MOLECULE")?;
    writeln!(writer, "{title}")?;
    writeln!(
        writer,
        "{} {} 0 0 0",
        structure.atoms.len(),
        structure.bonds.len()
    )?;
    writeln!(writer, "SMALL")?;
    writeln!(writer, "USER_CHARGES")?;
    writeln!(writer)?;
    writeln!(writer, "@<TRIPOS>ATOM")?;
    for (index, atom) in structure.atoms.iter().enumerate() {
        let serial = if atom.serial == 0 {
            index + 1
        } else {
            atom.serial
        };
        let atom_type = atom.atom_type.as_deref().unwrap_or("C.3");
        let subst_id = atom.residue_id;
        let subst_name = if atom.residue_name.is_empty() {
            "UNK"
        } else {
            &atom.residue_name
        };
        let charge = atom.charge.unwrap_or(0.0);
        writeln!(
            writer,
            "{serial:>7} {:<8} {:>10.4} {:>10.4} {:>10.4} {:<8} {subst_id:>4} {:<8} {charge:>10.4}",
            atom.name, atom.x, atom.y, atom.z, atom_type, subst_name
        )?;
    }
    if !structure.bonds.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "@<TRIPOS>BOND")?;
        for (index, bond) in structure.bonds.iter().enumerate() {
            let serial = if bond.serial == 0 {
                index + 1
            } else {
                bond.serial
            };
            writeln!(
                writer,
                "{serial:>6} {:-6} {:-6} {}",
                bond.atom1, bond.atom2, bond.bond_type
            )?;
        }
    }
    if let Some(dimensions) = structure.dimensions {
        writeln!(writer)?;
        writeln!(writer, "@<TRIPOS>CRYSIN")?;
        writeln!(
            writer,
            "{:.4} {:.4} {:.4} {:.4} {:.4} {:.4} 1 1",
            dimensions[0],
            dimensions[1],
            dimensions[2],
            dimensions[3],
            dimensions[4],
            dimensions[5]
        )?;
    }
    Ok(())
}

fn parse_crd(input: &str) -> Result<Structure, FormatError> {
    let mut structure = Structure::default();
    let mut expected = None;
    let mut saw_count = false;
    for (index, raw_line) in input.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('*') {
            if structure.title.is_empty() && line.starts_with('*') {
                structure.title = line.trim_start_matches('*').trim().to_owned();
            }
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if !saw_count && let Ok(count) = tokens[0].parse::<usize>() {
            // Extended CRD files append the literal `EXT` to the atom count.
            // The token is a layout marker rather than a second count.
            if tokens.len() == 1 || (tokens.len() == 2 && tokens[1].eq_ignore_ascii_case("EXT")) {
                expected = Some(count);
                saw_count = true;
                continue;
            }
        }
        if tokens.len() < 8 {
            return Err(parse_error(
                "CRD",
                line_number,
                "atom record requires at least eight fields",
            ));
        }
        let serial = parse_token(tokens[0], "atom serial", "CRD", line_number)?;
        let (segment_id, residue_id, residue_name, name, coordinate_start) =
            if tokens[1].parse::<i32>().is_ok() {
                (
                    tokens.get(7).map(|value| (*value).to_owned()),
                    parse_token(tokens[1], "residue number", "CRD", line_number)?,
                    tokens[2].to_owned(),
                    tokens[3].to_owned(),
                    4,
                )
            } else if tokens[2].parse::<i32>().is_ok() {
                (
                    Some(tokens[1].to_owned()),
                    parse_token(tokens[2], "residue number", "CRD", line_number)?,
                    tokens[3].to_owned(),
                    tokens[4].to_owned(),
                    5,
                )
            } else {
                return Err(parse_error(
                    "CRD",
                    line_number,
                    "cannot identify residue number",
                ));
            };
        let x = parse_token(tokens[coordinate_start], "x coordinate", "CRD", line_number)?;
        let y = parse_token(
            tokens[coordinate_start + 1],
            "y coordinate",
            "CRD",
            line_number,
        )?;
        let z = parse_token(
            tokens[coordinate_start + 2],
            "z coordinate",
            "CRD",
            line_number,
        )?;
        structure.atoms.push(FormatAtom {
            serial,
            name: nonempty(&name, "atom name", "CRD", line_number)?,
            atom_type: None,
            residue_name: nonempty(&residue_name, "residue name", "CRD", line_number)?,
            residue_id,
            chain_id: None,
            segment_id,
            x,
            y,
            z,
            charge: None,
            radius: None,
        });
    }
    if let Some(expected) = expected
        && expected != structure.atoms.len()
    {
        return Err(parse_error(
            "CRD",
            1,
            format!(
                "declared {expected} atoms but found {}",
                structure.atoms.len()
            ),
        ));
    }
    Ok(structure)
}

fn write_crd_document<W: Write>(structure: &Structure, mut writer: W) -> Result<(), FormatError> {
    validate_atoms(structure, "CRD")?;
    let extended = structure.atoms.len() > 99_999;
    if !structure.title.trim().is_empty() {
        writeln!(writer, "* {}", structure.title.trim())?;
    }
    if extended {
        writeln!(writer, "{:10} EXT", structure.atoms.len())?;
        for (index, atom) in structure.atoms.iter().enumerate() {
            let serial = if atom.serial == 0 {
                index + 1
            } else {
                atom.serial
            };
            let segment = atom.segment_id.as_deref().unwrap_or("");
            writeln!(
                writer,
                "{serial:10}{:10}  {:<8.8}  {:<8.8}{:20.10}{:20.10}{:20.10}  {:<8.8}  {:<8}{:20.10}",
                atom.residue_id,
                atom.residue_name,
                atom.name,
                atom.x,
                atom.y,
                atom.z,
                segment,
                atom.residue_id,
                0.0_f64,
            )?;
        }
    } else {
        writeln!(writer, "{:5}", structure.atoms.len())?;
        for (index, atom) in structure.atoms.iter().enumerate() {
            let serial = if atom.serial == 0 {
                index + 1
            } else {
                atom.serial
            };
            let segment = atom.segment_id.as_deref().unwrap_or("");
            writeln!(
                writer,
                "{serial:5}{:5} {:<4.4} {:<4.4}{:10.5}{:10.5}{:10.5} {:<4.4} {:<4}{:10.5}",
                atom.residue_id,
                atom.residue_name,
                atom.name,
                atom.x,
                atom.y,
                atom.z,
                segment,
                atom.residue_id,
                0.0_f64,
            )?;
        }
    }
    Ok(())
}

fn validate_atoms(structure: &Structure, format: &'static str) -> Result<(), FormatError> {
    for (index, atom) in structure.atoms.iter().enumerate() {
        if atom.name.trim().is_empty() {
            return Err(FormatError::InvalidStructure(format!(
                "{format} atom {} has an empty name",
                index + 1
            )));
        }
        if !atom.x.is_finite() || !atom.y.is_finite() || !atom.z.is_finite() {
            return Err(FormatError::InvalidStructure(format!(
                "{format} atom {} has non-finite coordinates",
                index + 1
            )));
        }
    }
    Ok(())
}

fn parse_error(format: &'static str, line: usize, message: impl Into<String>) -> FormatError {
    FormatError::Parse {
        format,
        line,
        message: message.into(),
    }
}

fn parse_token<T: std::str::FromStr>(
    value: &str,
    field: &str,
    format: &'static str,
    line: usize,
) -> Result<T, FormatError>
where
    T::Err: fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| parse_error(format, line, format!("invalid {field} {value:?}: {error}")))
}

fn nonempty(
    value: &str,
    field: &str,
    format: &'static str,
    line: usize,
) -> Result<String, FormatError> {
    (!value.trim().is_empty())
        .then(|| value.to_owned())
        .ok_or_else(|| parse_error(format, line, format!("missing {field}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pqr_round_trip_with_and_without_chain() {
        let input = "REMARK test\nATOM 1 N ALA A 1 1.0 2.0 3.0 -0.3 1.5\nATOM 2 CA ALA 1 2.0 3.0 4.0 0.1 1.7\nEND\n";
        let structure = Structure::from_pqr_str(input).expect("valid PQR");
        assert_eq!(structure.atoms.len(), 2);
        assert_eq!(structure.atoms[0].chain_id.as_deref(), Some("A"));
        assert_eq!(structure.atoms[1].chain_id, None);
        let output = structure.to_pqr_string().expect("write PQR");
        let reparsed = Structure::from_pqr_str(&output).expect("round trip");
        assert_eq!(reparsed.atoms, structure.atoms);
    }

    #[test]
    fn malformed_pqr_reports_line() {
        let error = Structure::from_pqr_str("ATOM 1 N ALA A 1 1.0 bad 3.0 -0.3 1.5").unwrap_err();
        assert!(matches!(
            error,
            FormatError::Parse {
                format: "PQR",
                line: 1,
                ..
            }
        ));
    }

    #[test]
    fn mol2_round_trip_preserves_bonds() {
        let input = "@<TRIPOS>MOLECULE\nwater\n3 2 0 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n1 O 0 0 0 O.2 1 HOH -0.8\n2 H1 1 0 0 H 1 HOH 0.4\n3 H2 0 1 0 H 1 HOH 0.4\n@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n";
        let structure = Structure::from_mol2_str(input).expect("valid MOL2");
        assert_eq!(structure.atoms.len(), 3);
        assert_eq!(structure.bonds.len(), 2);
        let output = structure.to_mol2_string().expect("write MOL2");
        let reparsed = Structure::from_mol2_str(&output).expect("round trip");
        assert_eq!(reparsed.bonds, structure.bonds);
        assert_eq!(reparsed.atoms[0].charge, Some(-0.8));
    }

    #[test]
    fn mol2_crysin_round_trip_preserves_dimensions() {
        let input = concat!(
            "@<TRIPOS>MOLECULE\n",
            "cell\n",
            "1 0 0 0 0\n",
            "SMALL\n",
            "USER_CHARGES\n",
            "@<TRIPOS>ATOM\n",
            "1 C 0 0 0 C.3 1 ALA 0\n",
            "@<TRIPOS>CRYSIN\n",
            "40 50 60 90 90 90 1 1\n",
        );
        let structure = Structure::from_mol2_str(input).expect("valid MOL2");
        assert_eq!(
            structure.dimensions,
            Some([40.0, 50.0, 60.0, 90.0, 90.0, 90.0])
        );
        let reparsed =
            Structure::from_mol2_str(&structure.to_mol2_string().unwrap()).expect("round trip");
        assert_eq!(reparsed.dimensions, structure.dimensions);
    }

    #[test]
    fn crd_reads_card_layout_and_count() {
        let input =
            "* title\n    2\n    1 SEG 1 ALA N 1.0 2.0 3.0\n    2 SEG 1 ALA CA 2.0 3.0 4.0\n";
        let structure = Structure::from_crd_str(input).expect("valid CRD");
        assert_eq!(structure.atoms.len(), 2);
        assert_eq!(structure.title, "title");
        assert_eq!(structure.atoms[1].name, "CA");
    }

    #[test]
    fn crd_accepts_extended_count_marker() {
        let input = "* title\n    1 EXT\n    1 SEG 1 ALA N 1.0 2.0 3.0\n";
        let structure = Structure::from_crd_str(input).expect("valid extended CRD");
        assert_eq!(structure.atoms.len(), 1);
    }

    #[test]
    fn crd_standard_layout_and_writer_round_trip() {
        let input = concat!(
            "* title\n",
            "    2\n",
            "    1    1 MET  N    -11.92100  26.30700  10.41000 4AKE 1      0.00000\n",
            "    2    1 MET  CA   -10.92100  26.30700  10.41000 4AKE 1      0.00000\n",
        );
        let structure = Structure::from_crd_str(input).expect("standard CRD");
        assert_eq!(structure.atoms[0].segment_id.as_deref(), Some("4AKE"));
        assert_eq!(structure.atoms[1].position(), [-10.921, 26.307, 10.41]);
        let reparsed = Structure::from_crd_str(&structure.to_crd_string().unwrap()).unwrap();
        assert_eq!(reparsed.atoms.len(), 2);
        assert_eq!(reparsed.atoms[0].residue_name, "MET");
        assert_eq!(reparsed.atoms[0].segment_id.as_deref(), Some("4AKE"));
    }

    #[test]
    fn mol2_count_mismatch_is_rejected() {
        let input = "@<TRIPOS>MOLECULE\nx\n2 0 0 0 0\n@<TRIPOS>ATOM\n1 C 0 0 0 C.3 1 X 0\n";
        assert!(matches!(
            Structure::from_mol2_str(input),
            Err(FormatError::Parse { format: "MOL2", .. })
        ));
    }
}
