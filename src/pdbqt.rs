//! PDBQT (AutoDock) structure and coordinate support.
//!
//! PDBQT is a PDB-like single-frame format that stores an AutoDock atom type
//! and a partial charge in the columns normally occupied by PDB element and
//! charge fields.  Control records used to describe rotatable bonds (for
//! example `ROOT`, `BRANCH`, and `TORSDOF`) are intentionally ignored: the
//! topology represented by a PDBQT file is the atom list and does not include
//! covalent bonds.

use crate::pdb::PdbCryst1;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// An atom record from a PDBQT document.
#[derive(Clone, Debug, PartialEq)]
pub struct PdbqtAtom {
    /// Atom serial number.  PDBQT files do not require serials to be
    /// contiguous.
    pub serial: u32,
    pub name: String,
    pub alt_loc: Option<char>,
    pub residue_name: String,
    pub chain_id: Option<char>,
    pub residue_sequence: i32,
    pub insertion_code: Option<char>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub occupancy: f64,
    pub temperature_factor: f64,
    /// Gasteiger/AutoDock partial charge.
    pub charge: f64,
    /// AutoDock atom type (for example `C`, `OA`, or `HD`).
    pub atom_type: String,
    /// Whether this record was written as `HETATM` rather than `ATOM`.
    pub hetatm: bool,
}

impl Default for PdbqtAtom {
    fn default() -> Self {
        Self {
            serial: 1,
            name: "X".to_owned(),
            alt_loc: None,
            residue_name: "UNK".to_owned(),
            chain_id: None,
            residue_sequence: 1,
            insertion_code: None,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temperature_factor: 0.0,
            charge: 0.0,
            atom_type: String::new(),
            hetatm: false,
        }
    }
}

impl PdbqtAtom {
    /// Construct an atom with the required identifying fields and position.
    #[must_use]
    pub fn new(serial: u32, name: impl Into<String>, position: [f64; 3]) -> Self {
        Self {
            serial,
            name: name.into(),
            x: position[0],
            y: position[1],
            z: position[2],
            ..Self::default()
        }
    }

    /// Return the Cartesian coordinates as an array.
    #[must_use]
    pub const fn position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Alias using the conventional `resid` spelling.
    #[must_use]
    pub const fn resid(&self) -> i32 {
        self.residue_sequence
    }

    /// Alias using the conventional `resname` spelling.
    #[must_use]
    pub fn resname(&self) -> &str {
        &self.residue_name
    }
}

/// A single-frame PDBQT structure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PdbqtStructure {
    /// Atom records in file order.
    pub atoms: Vec<PdbqtAtom>,
    /// Optional CRYST1 unit-cell information.
    pub cryst1: Option<PdbCryst1>,
    /// Optional TITLE text.  PDBQT control records are not retained.
    pub title: String,
}

impl PdbqtStructure {
    /// Parse a PDBQT document from a string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, PdbqtError> {
        parse_pdbqt(input)
    }

    /// Parse a PDBQT document from a reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, PdbqtError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Read a PDBQT document from a filesystem path.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, PdbqtError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    /// Serialize this structure as a PDBQT document.
    pub fn to_pdbqt_string(&self) -> Result<String, PdbqtError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Write this structure to any writer in PDBQT layout.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), PdbqtError> {
        validate_structure(self)?;

        if !self.title.trim().is_empty() {
            let title = self.title.replace(['\r', '\n'], " ");
            writeln!(writer, "TITLE     {title}")?;
        }
        if let Some(cryst1) = &self.cryst1 {
            writeln!(writer, "{}", format_cryst1(cryst1))?;
        }
        for atom in &self.atoms {
            writeln!(writer, "{}", format_atom(atom))?;
        }
        writer.write_all(b"END\n")?;
        Ok(())
    }

    /// Write this structure to a filesystem path.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), PdbqtError> {
        self.write(File::create(path)?)
    }

    /// Number of atoms in this structure.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Return the coordinate frame represented by this single-frame file.
    #[must_use]
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(PdbqtAtom::position).collect()
    }
}

impl std::str::FromStr for PdbqtStructure {
    type Err = PdbqtError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Read a PDBQT structure from a filesystem path.
pub fn read_pdbqt<P: AsRef<Path>>(path: P) -> Result<PdbqtStructure, PdbqtError> {
    PdbqtStructure::read_file(path)
}

/// Write a PDBQT structure to a filesystem path.
pub fn write_pdbqt<P: AsRef<Path>>(path: P, structure: &PdbqtStructure) -> Result<(), PdbqtError> {
    structure.write_file(path)
}

/// Errors produced while parsing or writing PDBQT data.
#[derive(Debug)]
pub enum PdbqtError {
    Io(io::Error),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for PdbqtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "PDBQT parse error on line {line}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid PDBQT structure: {message}")
            }
        }
    }
}

impl std::error::Error for PdbqtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for PdbqtError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn parse_pdbqt(input: &str) -> Result<PdbqtStructure, PdbqtError> {
    let mut structure = PdbqtStructure::default();
    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let record = field(line, 0, 6).trim();
        match record {
            "ATOM" | "HETATM" => {
                structure
                    .atoms
                    .push(parse_atom(line, line_number, record == "HETATM")?);
            }
            "CRYST1" => {
                structure.cryst1 = Some(parse_cryst1(line, line_number)?);
            }
            "TITLE" => {
                structure.title = field(line, 10, line.len()).trim().to_owned();
            }
            "MODEL" => {
                return Err(PdbqtError::Parse {
                    line: line_number,
                    message: "PDBQT supports a single coordinate frame; MODEL is not supported"
                        .to_owned(),
                });
            }
            // `END` terminates a PDBQT document.  Do not use starts_with:
            // ENDBRANCH and ENDROOT are valid AutoDock control records.
            "END" => break,
            _ => {}
        }
    }
    Ok(structure)
}

fn parse_atom(line: &str, line_number: usize, hetatm: bool) -> Result<PdbqtAtom, PdbqtError> {
    let serial = parse_required::<u32>(line, 6, 11, line_number, "atom serial")?;
    let name = field(line, 12, 16).trim().to_owned();
    if name.is_empty() {
        return Err(parse_error(line_number, "atom name is empty"));
    }

    let residue_name = field(line, 17, 21).trim().to_owned();
    let residue_sequence = parse_required::<i32>(line, 22, 26, line_number, "residue sequence")?;
    let x = parse_required::<f64>(line, 30, 38, line_number, "x coordinate")?;
    let y = parse_required::<f64>(line, 38, 46, line_number, "y coordinate")?;
    let z = parse_required::<f64>(line, 46, 54, line_number, "z coordinate")?;
    let occupancy = parse_required::<f64>(line, 54, 60, line_number, "occupancy")?;
    let temperature_factor =
        parse_required::<f64>(line, 60, 66, line_number, "temperature factor")?;

    // PDBQT puts the charge at columns 71-76 and leaves column 77 as a
    // separator.  A few writers omit trailing blanks, so only the type field
    // is optional; charge remains required by the format.
    let charge = parse_required::<f64>(line, 70, 76, line_number, "partial charge")?;
    let atom_type = field(line, 77, 80).trim().to_owned();

    Ok(PdbqtAtom {
        serial,
        name,
        alt_loc: nonblank_char(field(line, 16, 17)),
        residue_name,
        chain_id: nonblank_char(field(line, 21, 22)),
        residue_sequence,
        insertion_code: nonblank_char(field(line, 26, 27)),
        x,
        y,
        z,
        occupancy,
        temperature_factor,
        charge,
        atom_type,
        hetatm,
    })
}

fn parse_cryst1(line: &str, line_number: usize) -> Result<PdbCryst1, PdbqtError> {
    Ok(PdbCryst1 {
        a: parse_required::<f64>(line, 6, 15, line_number, "CRYST1 a")?,
        b: parse_required::<f64>(line, 15, 24, line_number, "CRYST1 b")?,
        c: parse_required::<f64>(line, 24, 33, line_number, "CRYST1 c")?,
        alpha: parse_required::<f64>(line, 33, 40, line_number, "CRYST1 alpha")?,
        beta: parse_required::<f64>(line, 40, 47, line_number, "CRYST1 beta")?,
        gamma: parse_required::<f64>(line, 47, 54, line_number, "CRYST1 gamma")?,
        space_group: field(line, 55, 66).trim().to_owned(),
        z: parse_optional::<u32>(line, 66, 70, line_number, "CRYST1 Z")?,
    })
}

fn validate_structure(structure: &PdbqtStructure) -> Result<(), PdbqtError> {
    for (index, atom) in structure.atoms.iter().enumerate() {
        let coordinate = atom.position();
        if coordinate.iter().any(|value| !value.is_finite()) {
            return Err(PdbqtError::InvalidStructure(format!(
                "atom {} has a non-finite coordinate",
                index + 1
            )));
        }
        // These are the representable PDBQT coordinate limits used by
        // MDAnalysis' writer (8.3 fixed-width coordinates).
        if coordinate
            .iter()
            .any(|value| *value < -999.9995 || *value > 9999.9995)
        {
            return Err(PdbqtError::InvalidStructure(format!(
                "atom {} coordinate is outside PDBQT range [-999.9995, 9999.9995]",
                index + 1
            )));
        }
        if !atom.occupancy.is_finite()
            || !atom.temperature_factor.is_finite()
            || !atom.charge.is_finite()
        {
            return Err(PdbqtError::InvalidStructure(format!(
                "atom {} contains a non-finite scalar field",
                index + 1
            )));
        }
        if atom.residue_sequence < -999 || atom.residue_sequence > 9999 {
            return Err(PdbqtError::InvalidStructure(format!(
                "atom {} residue sequence {} does not fit the PDBQT field",
                index + 1,
                atom.residue_sequence
            )));
        }
        if atom.charge < -999.999 || atom.charge > 9999.999 {
            return Err(PdbqtError::InvalidStructure(format!(
                "atom {} partial charge {} does not fit the PDBQT field",
                index + 1,
                atom.charge
            )));
        }
    }
    Ok(())
}

fn format_atom(atom: &PdbqtAtom) -> String {
    let record = if atom.hetatm { "HETATM" } else { "ATOM  " };
    let mut name: String = atom.name.chars().take(4).collect();
    if name.chars().count() < 4 {
        name = format!(" {name:<3}");
    }
    let residue_name = fit_field(&atom.residue_name, 4, false);
    let chain_id = atom.chain_id.unwrap_or(' ');
    let alt_loc = atom.alt_loc.unwrap_or(' ');
    let insertion_code = atom.insertion_code.unwrap_or(' ');
    let atom_type = fit_field(&atom.atom_type, 2, false);

    format!(
        "{record}{:>5} {name}{alt_loc}{residue_name}{chain_id}{:>4}{insertion_code}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}    {:>6.3} {atom_type}",
        atom.serial,
        atom.residue_sequence,
        atom.x,
        atom.y,
        atom.z,
        atom.occupancy,
        atom.temperature_factor,
        atom.charge,
    )
}

fn format_cryst1(cryst1: &PdbCryst1) -> String {
    let space_group = fit_field(&cryst1.space_group, 11, false);
    let z = cryst1
        .z
        .map_or_else(|| "    ".to_owned(), |value| format!("{value:>4}"));
    format!(
        "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} {space_group}{z}",
        cryst1.a, cryst1.b, cryst1.c, cryst1.alpha, cryst1.beta, cryst1.gamma
    )
}

fn fit_field(value: &str, width: usize, right_align: bool) -> String {
    let value: String = value.chars().take(width).collect();
    if right_align {
        format!("{value:>width$}")
    } else {
        format!("{value:<width$}")
    }
}

fn field(line: &str, start: usize, end: usize) -> &str {
    let bytes = line.as_bytes();
    if start >= bytes.len() {
        return "";
    }
    let end = end.min(bytes.len());
    std::str::from_utf8(&bytes[start..end]).unwrap_or("")
}

fn nonblank_char(value: &str) -> Option<char> {
    value.chars().find(|character| !character.is_whitespace())
}

fn parse_required<T>(
    line: &str,
    start: usize,
    end: usize,
    line_number: usize,
    field_name: &str,
) -> Result<T, PdbqtError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let value = field(line, start, end).trim();
    if value.is_empty() {
        return Err(parse_error(line_number, format!("missing {field_name}")));
    }
    value.parse::<T>().map_err(|error| {
        parse_error(
            line_number,
            format!("invalid {field_name} {value:?}: {error}"),
        )
    })
}

fn parse_optional<T>(
    line: &str,
    start: usize,
    end: usize,
    line_number: usize,
    field_name: &str,
) -> Result<Option<T>, PdbqtError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let value = field(line, start, end).trim();
    if value.is_empty() {
        return Ok(None);
    }
    value.parse::<T>().map(Some).map_err(|error| {
        parse_error(
            line_number,
            format!("invalid {field_name} {value:?}: {error}"),
        )
    })
}

fn parse_error(line: usize, message: impl Into<String>) -> PdbqtError {
    PdbqtError::Parse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const ATOM: &str =
        "ATOM      1  N   PRO A   2      14.607 -11.551  25.793  1.00  0.00    -0.062 N\n";

    #[test]
    fn parses_fixed_columns_and_control_records() {
        let input = format!(
            "TITLE     ligand\nCRYST1   10.000   20.000   30.000  90.00  90.00 120.00 P 1          1\nROOT\n{ATOM}ENDROOT\nTORSDOF 0\nEND\n"
        );
        let structure = PdbqtStructure::from_str(&input).expect("valid PDBQT");
        assert_eq!(structure.title, "ligand");
        assert_eq!(structure.n_atoms(), 1);
        let atom = &structure.atoms[0];
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "N");
        assert_eq!(atom.residue_name, "PRO");
        assert_eq!(atom.chain_id, Some('A'));
        assert_eq!(atom.residue_sequence, 2);
        assert_eq!(atom.position(), [14.607, -11.551, 25.793]);
        assert_eq!(atom.charge, -0.062);
        assert_eq!(atom.atom_type, "N");
        assert_eq!(
            structure.cryst1.as_ref().map(|cell| cell.gamma),
            Some(120.0)
        );
    }

    #[test]
    fn round_trip_preserves_atom_fields_and_writer_layout() {
        let structure = PdbqtStructure::from_str(ATOM).expect("valid PDBQT");
        let output = structure.to_pdbqt_string().expect("write PDBQT");
        assert_eq!(
            output.lines().next().map(str::trim_end),
            Some(ATOM.trim_end())
        );
        let reparsed = PdbqtStructure::from_str(&output).expect("read written PDBQT");
        assert_eq!(reparsed.atoms, structure.atoms);
    }

    #[test]
    fn supports_hetatm_and_nonsequential_serials() {
        let input = concat!(
            "HETATM   42  OA  HOH B  17       1.000   2.000   3.000  0.50 10.00    -0.500 OA\n",
            "END\n",
        );
        let structure = PdbqtStructure::read(Cursor::new(input.as_bytes())).expect("read");
        assert!(structure.atoms[0].hetatm);
        assert_eq!(structure.atoms[0].serial, 42);
        assert_eq!(structure.atoms[0].chain_id, Some('B'));
    }

    #[test]
    fn rejects_malformed_atom_and_models() {
        let malformed = ATOM.replacen("-0.062", "broken", 1);
        assert!(matches!(
            PdbqtStructure::from_str(&malformed),
            Err(PdbqtError::Parse { line: 1, .. })
        ));
        assert!(matches!(
            PdbqtStructure::from_str("MODEL        1\n"),
            Err(PdbqtError::Parse { line: 1, .. })
        ));
    }

    #[test]
    fn rejects_unrepresentable_coordinates() {
        let mut structure = PdbqtStructure::from_str(ATOM).expect("valid");
        structure.atoms[0].x = 10_000.0;
        assert!(matches!(
            structure.to_pdbqt_string(),
            Err(PdbqtError::InvalidStructure(_))
        ));
    }
}
