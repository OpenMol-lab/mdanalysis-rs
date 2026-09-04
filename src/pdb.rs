//! PDB coordinate-file parsing and writing.
//!
//! The parser intentionally focuses on the records needed for a molecular
//! structure: `ATOM`, `HETATM`, `MODEL`/`ENDMDL`, and `CRYST1`.  Other records
//! are ignored, which makes it possible to read files containing headers,
//! remarks, and connectivity records without having to model those records.

use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// An atom from a PDB structure.
///
/// The coordinates are the coordinates of the first frame in the containing
/// [`PdbStructure`].  Additional model coordinates are available through
/// [`PdbStructure::frames`].
#[derive(Debug, Clone, PartialEq)]
pub struct PdbAtom {
    pub serial: u32,
    pub name: String,
    pub alt_loc: Option<char>,
    pub residue_name: String,
    pub chain_id: Option<char>,
    /// Optional segment identifier stored in columns 73-76.
    pub segid: Option<String>,
    pub residue_sequence: i32,
    pub insertion_code: Option<char>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub occupancy: Option<f64>,
    pub temperature_factor: Option<f64>,
    pub element: Option<String>,
    pub charge: Option<String>,
    pub hetatm: bool,
}

impl PdbAtom {
    /// Return the atom coordinates as an array.
    pub fn position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    #[must_use]
    pub const fn resid(&self) -> i32 {
        self.residue_sequence
    }

    #[must_use]
    pub fn resname(&self) -> &str {
        &self.residue_name
    }

    #[must_use]
    pub fn chain(&self) -> Option<char> {
        self.chain_id
    }

    #[must_use]
    pub const fn temp_factor(&self) -> Option<f64> {
        self.temperature_factor
    }

    fn with_position(&self, position: [f64; 3]) -> Self {
        let mut atom = self.clone();
        atom.x = position[0];
        atom.y = position[1];
        atom.z = position[2];
        atom
    }
}

/// A connectivity record from a PDB `CONECT` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdbBond {
    /// Serial number of the first atom.
    pub atom1: u32,
    /// Serial number of the bonded atom.
    pub atom2: u32,
}

impl PdbBond {
    #[must_use]
    pub const fn new(atom1: u32, atom2: u32) -> Self {
        Self { atom1, atom2 }
    }
}

/// Unit-cell information from a `CRYST1` record.
#[derive(Debug, Clone, PartialEq)]
pub struct PdbCryst1 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub space_group: String,
    pub z: Option<u32>,
}

/// Alias using the more general name commonly used by callers.
pub type PdbCrystal = PdbCryst1;

/// A PDB structure and its coordinate frames.
///
/// `atoms` contains metadata and coordinates for the first frame.  Each
/// entry in `frames` contains one coordinate triplet per atom, in the same
/// order as `atoms`.  A file without `MODEL` records has one frame when it
/// contains at least one atom.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PdbStructure {
    pub atoms: Vec<PdbAtom>,
    pub frames: Vec<Vec<[f64; 3]>>,
    /// Per-frame atom records, including occupancy and temperature-factor
    /// values that may vary between MODEL records.
    pub frame_atoms: Vec<Vec<PdbAtom>>,
    pub cryst1: Option<PdbCryst1>,
    /// Connectivity records in atom-serial-number space.
    pub bonds: Vec<PdbBond>,
}

impl PdbStructure {
    /// Parse a PDB document from a UTF-8 string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, PdbError> {
        Self::parse_lines(input.lines())
    }

    /// Parse a PDB document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, PdbError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Read a PDB document from a file.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, PdbError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    /// Number of coordinate frames in this structure.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Return the coordinates for a frame, if it exists.
    pub fn frame(&self, index: usize) -> Option<&[[f64; 3]]> {
        self.frames.get(index).map(Vec::as_slice)
    }

    #[must_use]
    pub fn positions(&self) -> Option<&[[f64; 3]]> {
        self.frame(0)
    }

    /// Return atoms for a frame, combining its coordinates with that frame's
    /// metadata when available.
    pub fn atoms_for_frame(&self, index: usize) -> Option<Vec<PdbAtom>> {
        let frame = self.frames.get(index)?;
        if frame.len() != self.atoms.len() {
            return None;
        }
        let metadata = self
            .frame_atoms
            .get(index)
            .filter(|atoms| atoms.len() == frame.len())
            .map_or(self.atoms.as_slice(), Vec::as_slice);
        Some(
            metadata
                .iter()
                .zip(frame.iter().copied())
                .map(|(atom, position)| atom.with_position(position))
                .collect(),
        )
    }

    /// Serialize this structure to a PDB document.
    pub fn to_pdb_string(&self) -> Result<String, PdbError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        // All output is generated from ASCII-compatible fields.  If a caller
        // supplied non-ASCII metadata, replace it rather than failing a
        // serialization operation that otherwise succeeded.
        Ok(String::from_utf8_lossy(&output).into_owned())
    }

    /// Write this structure to any writer.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), PdbError> {
        if let Some(cryst1) = &self.cryst1 {
            writeln!(writer, "{}", format_cryst1(cryst1))?;
        }

        match self.frames.len() {
            0 => {
                if !self.atoms.is_empty() {
                    return Err(PdbError::InvalidStructure(
                        "atoms are present but no coordinate frame exists".to_string(),
                    ));
                }
            }
            1 => {
                validate_frame(self, 0)?;
                for atom in self.atoms_for_frame(0).expect("validated PDB atom count") {
                    writeln!(writer, "{}", format_atom(&atom))?;
                }
            }
            _ => {
                for (index, _frame) in self.frames.iter().enumerate() {
                    validate_frame(self, index)?;
                    writeln!(writer, "MODEL{:>9}", index + 1)?;
                    for atom in self
                        .atoms_for_frame(index)
                        .expect("validated PDB atom count")
                    {
                        writeln!(writer, "{}", format_atom(&atom))?;
                    }
                    writeln!(writer, "ENDMDL")?;
                }
            }
        }

        for bond in &self.bonds {
            writeln!(writer, "CONECT{:>5}{:>5}", bond.atom1, bond.atom2)?;
        }

        writer.write_all(b"END\n")?;
        Ok(())
    }

    /// Write this structure to a file.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), PdbError> {
        self.write(File::create(path)?)
    }

    fn parse_lines<'a, I>(lines: I) -> Result<Self, PdbError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut atoms = Vec::new();
        let mut model_atoms: Option<Vec<PdbAtom>> = None;
        let mut models = Vec::new();
        let mut saw_model = false;
        let mut cryst1 = None;
        let mut bonds = Vec::new();
        let mut previous_serial: Option<u32> = None;
        let mut previous_residue_sequence: Option<i32> = None;

        for (line_index, line) in lines.into_iter().enumerate() {
            let line_number = line_index + 1;
            let record = field(line, 0, 6).trim();
            match record {
                "ATOM" | "HETATM" => {
                    let mut atom = parse_atom(
                        line,
                        line_number,
                        record == "HETATM",
                        previous_serial.map(|serial| serial.saturating_add(1)),
                    )?;
                    if !is_xpdb_residue_line(line)
                        && previous_residue_sequence.is_some_and(|previous| {
                            atom.residue_sequence
                                .checked_sub(previous)
                                .is_some_and(|difference| difference < -5000)
                        })
                    {
                        atom.residue_sequence = atom
                            .residue_sequence
                            .checked_add(10000)
                            .ok_or_else(|| PdbError::Parse {
                                line: line_number,
                                message: "wrapped residue sequence overflows i32".to_owned(),
                            })?;
                    }
                    previous_residue_sequence = Some(atom.residue_sequence);
                    previous_serial = Some(atom.serial);
                    if let Some(current) = model_atoms.as_mut() {
                        current.push(atom);
                    } else if saw_model {
                        return Err(PdbError::Parse {
                            line: line_number,
                            message: "ATOM/HETATM record outside MODEL".to_string(),
                        });
                    } else {
                        atoms.push(atom);
                    }
                }
                "MODEL" => {
                    if let Some(current) = model_atoms.take() {
                        // A few real-world trajectories omit ENDMDL between
                        // consecutive MODEL records.  Treat the next MODEL
                        // as the implicit terminator when the current model
                        // contains atoms; an empty nested MODEL remains
                        // malformed.
                        if current.is_empty() {
                            return Err(PdbError::Parse {
                                line: line_number,
                                message: "nested MODEL record".to_string(),
                            });
                        }
                        models.push(current);
                    }
                    saw_model = true;
                    previous_serial = None;
                    previous_residue_sequence = None;
                    model_atoms = Some(Vec::new());
                }
                "ENDMDL" => {
                    let current = model_atoms.take().ok_or_else(|| PdbError::Parse {
                        line: line_number,
                        message: "ENDMDL without MODEL".to_string(),
                    })?;
                    models.push(current);
                }
                "CRYST1" => {
                    match parse_cryst1(line, line_number) {
                        Ok(value) if !is_placeholder_cryst1(&value) => cryst1 = Some(value),
                        Ok(_) => {}
                        // PDB files in the wild sometimes include a CRYST1
                        // record with incomplete or malformed fields.
                        // MDAnalysis treats such a record as absent unit-cell
                        // data, so keep parsing the coordinate records.
                        Err(_) => {}
                    }
                }
                "CONECT" => bonds.extend(parse_conect(line, line_number)?),
                _ => {}
            }
        }

        if let Some(current) = model_atoms.take() {
            // Accept a final MODEL whose ENDMDL was omitted, as is common in
            // hand-written or truncated coordinate files.
            models.push(current);
        }

        if !models.is_empty() {
            let expected = models[0].len();
            for (index, model) in models.iter().enumerate() {
                if model.len() != expected {
                    return Err(PdbError::InconsistentModel {
                        model: index + 1,
                        expected,
                        found: model.len(),
                    });
                }
            }
            let frame_atoms = models;
            atoms = frame_atoms[0].clone();
            let frames = frame_atoms
                .iter()
                .map(|model| model.iter().map(PdbAtom::position).collect())
                .collect();
            Ok(Self {
                atoms,
                frames,
                frame_atoms,
                cryst1,
                bonds,
            })
        } else {
            let frames = if atoms.is_empty() {
                Vec::new()
            } else {
                vec![atoms.iter().map(PdbAtom::position).collect()]
            };
            let frame_atoms = Vec::new();
            Ok(Self {
                atoms,
                frames,
                frame_atoms,
                cryst1,
                bonds,
            })
        }
    }
}

impl std::str::FromStr for PdbStructure {
    type Err = PdbError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::parse_lines(input.lines())
    }
}

/// Read a PDB structure from a filesystem path.
pub fn read_pdb<P: AsRef<Path>>(path: P) -> Result<PdbStructure, PdbError> {
    PdbStructure::read_file(path)
}

/// Write a PDB structure to a filesystem path.
pub fn write_pdb<P: AsRef<Path>>(path: P, structure: &PdbStructure) -> Result<(), PdbError> {
    structure.write_file(path)
}

fn validate_frame(structure: &PdbStructure, index: usize) -> Result<(), PdbError> {
    let found = structure.frames[index].len();
    if found != structure.atoms.len() {
        return Err(PdbError::InconsistentModel {
            model: index + 1,
            expected: structure.atoms.len(),
            found,
        });
    }
    for (atom_index, position) in structure.frames[index].iter().enumerate() {
        if position.iter().any(|value| !value.is_finite()) {
            return Err(PdbError::InvalidStructure(format!(
                "model {} atom {} has a non-finite coordinate",
                index + 1,
                atom_index + 1
            )));
        }
        if position
            .iter()
            .any(|value| *value < -999.9995 || *value > 9999.9995)
        {
            return Err(PdbError::InvalidStructure(format!(
                "model {} atom {} coordinate is outside PDB range [-999.9995, 9999.9995]",
                index + 1,
                atom_index + 1
            )));
        }
    }
    Ok(())
}

fn parse_atom(
    line: &str,
    line_number: usize,
    hetatm: bool,
    overflow_serial: Option<u32>,
) -> Result<PdbAtom, PdbError> {
    let serial = parse_serial(line, line_number, overflow_serial)?;
    let name = field(line, 12, 16).trim().to_string();
    if name.is_empty() {
        return Err(PdbError::Parse {
            line: line_number,
            message: "atom name is empty".to_string(),
        });
    }
    // Standard residue names occupy three characters, but common
    // non-standard records such as `TIP3` use the complete four-column slot
    // when no chain ID is present.
    let residue_name = field(line, 17, 21).trim().to_string();
    // Extended PDB (XPDB) files permit five-digit residue numbers.  The
    // fifth character occupies the insertion-code column, but is numeric in
    // that format; retain it instead of silently truncating the residue ID.
    let extended_residue = field(line, 22, 27);
    let is_xpdb_residue = is_xpdb_residue_line(line);
    let residue_sequence = if is_xpdb_residue {
        extended_residue
            .trim()
            .parse::<i32>()
            .map_err(|error| PdbError::Parse {
                line: line_number,
                message: format!("invalid residue sequence: {error}"),
            })?
    } else {
        parse_hybrid36_optional(line, 22, 26, line_number, "residue sequence")?.unwrap_or(1)
    };
    let x = parse_required::<f64>(line, 30, 38, line_number, "x coordinate")?;
    let y = parse_required::<f64>(line, 38, 46, line_number, "y coordinate")?;
    let z = parse_required::<f64>(line, 46, 54, line_number, "z coordinate")?;
    let occupancy = parse_optional::<f64>(line, 54, 60, line_number, "occupancy")?;
    let temperature_factor =
        parse_optional::<f64>(line, 60, 66, line_number, "temperature factor")?;
    let element = nonempty_string(field(line, 76, 78));
    let charge = nonempty_string(field(line, 78, 80));

    Ok(PdbAtom {
        serial,
        name,
        alt_loc: nonblank_char(field(line, 16, 17)),
        residue_name,
        chain_id: nonblank_char(field(line, 21, 22)),
        segid: nonempty_string(field(line, 72, 76)),
        residue_sequence,
        insertion_code: if is_xpdb_residue {
            None
        } else {
            nonblank_char(field(line, 26, 27))
        },
        x,
        y,
        z,
        occupancy,
        temperature_factor,
        element,
        charge,
        hetatm,
    })
}

fn is_xpdb_residue_line(line: &str) -> bool {
    let standard_residue = field(line, 22, 26);
    let extended_residue = field(line, 22, 27);
    extended_residue.trim().len() > standard_residue.trim().len()
        && extended_residue.trim().parse::<i32>().is_ok()
}

fn parse_cryst1(line: &str, line_number: usize) -> Result<PdbCryst1, PdbError> {
    Ok(PdbCryst1 {
        a: parse_required::<f64>(line, 6, 15, line_number, "CRYST1 a")?,
        b: parse_required::<f64>(line, 15, 24, line_number, "CRYST1 b")?,
        c: parse_required::<f64>(line, 24, 33, line_number, "CRYST1 c")?,
        alpha: parse_required::<f64>(line, 33, 40, line_number, "CRYST1 alpha")?,
        beta: parse_required::<f64>(line, 40, 47, line_number, "CRYST1 beta")?,
        gamma: parse_required::<f64>(line, 47, 54, line_number, "CRYST1 gamma")?,
        space_group: field(line, 55, 66).trim().to_string(),
        z: parse_optional::<u32>(line, 66, 70, line_number, "CRYST1 Z")?,
    })
}

fn is_placeholder_cryst1(cryst1: &PdbCryst1) -> bool {
    cryst1.a == 1.0
        && cryst1.b == 1.0
        && cryst1.c == 1.0
        && cryst1.alpha == 90.0
        && cryst1.beta == 90.0
        && cryst1.gamma == 90.0
}

fn parse_conect(line: &str, line_number: usize) -> Result<Vec<PdbBond>, PdbError> {
    let bytes = line.as_bytes();
    let mut values = Vec::new();
    // PDB CONECT records use five-column integer fields, but accepting
    // whitespace-separated fields also handles common hand-written files.
    if bytes.len() > 6 {
        let tail = &bytes[6..];
        let fixed_values = tail
            .chunks(5)
            .filter_map(|chunk| {
                let value = std::str::from_utf8(chunk).unwrap_or("").trim();
                (!value.is_empty()).then_some(value)
            })
            .map(|value| hybrid36_decode(5, value))
            .map(|value| {
                value.and_then(|value| {
                    u32::try_from(value).map_err(|_| "value is outside the u32 range".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>();
        if !tail.first().is_some_and(u8::is_ascii_whitespace) {
            if !tail.len().is_multiple_of(5) {
                return Err(PdbError::Parse {
                    line: line_number,
                    message: "CONECT fixed-width fields are incomplete".to_string(),
                });
            }
            values = fixed_values.map_err(|error| PdbError::Parse {
                line: line_number,
                message: format!("invalid CONECT atom serial: {error}"),
            })?;
        } else if let Ok(fixed_values) = fixed_values {
            values = fixed_values;
        }
    }
    if values.len() < 2 {
        values = line
            .split_whitespace()
            .skip(1)
            .map(|value| {
                hybrid36_decode(5, value)
                    .and_then(|value| {
                        u32::try_from(value).map_err(|_| {
                            format!("CONECT atom serial {value:?} is outside the u32 range")
                        })
                    })
                    .map_err(|error| PdbError::Parse {
                        line: line_number,
                        message: format!("invalid CONECT atom serial {value:?}: {error}"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    if values.len() < 2 {
        return Err(PdbError::Parse {
            line: line_number,
            message: "CONECT record requires an atom and at least one partner".to_string(),
        });
    }
    Ok(values[1..]
        .iter()
        .copied()
        .map(|atom2| PdbBond::new(values[0], atom2))
        .collect())
}

fn field(line: &str, start: usize, end: usize) -> &str {
    let bytes = line.as_bytes();
    if start >= bytes.len() {
        return "";
    }
    let end = end.min(bytes.len());
    // PDB data is ASCII by definition.  Treat malformed UTF-8 boundaries as
    // an absent field instead of panicking while slicing a caller-provided
    // string containing unrelated Unicode text.
    std::str::from_utf8(&bytes[start..end]).unwrap_or("")
}

fn nonblank_char(value: &str) -> Option<char> {
    value.chars().find(|character| !character.is_whitespace())
}

fn nonempty_string(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn parse_required<T>(
    line: &str,
    start: usize,
    end: usize,
    line_number: usize,
    field_name: &str,
) -> Result<T, PdbError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let value = field(line, start, end).trim();
    if value.is_empty() {
        return Err(PdbError::Parse {
            line: line_number,
            message: format!("missing {field_name}"),
        });
    }
    value.parse::<T>().map_err(|error| PdbError::Parse {
        line: line_number,
        message: format!("invalid {field_name} {value:?}: {error}"),
    })
}

fn parse_serial(
    line: &str,
    line_number: usize,
    overflow_serial: Option<u32>,
) -> Result<u32, PdbError> {
    let value = field(line, 6, 11);
    let trimmed = value.trim();
    if !trimmed.is_empty() && trimmed.chars().all(|character| character == '*') {
        return overflow_serial.ok_or_else(|| PdbError::Parse {
            line: line_number,
            message: "atom serial overflow has no preceding serial".to_string(),
        });
    }
    let decoded = hybrid36_decode(5, value).map_err(|error| PdbError::Parse {
        line: line_number,
        message: format!("invalid atom serial {value:?}: {error}"),
    })?;
    u32::try_from(decoded).map_err(|_| PdbError::Parse {
        line: line_number,
        message: format!("atom serial {value:?} is outside the u32 range"),
    })
}

fn parse_hybrid36_optional(
    line: &str,
    start: usize,
    end: usize,
    line_number: usize,
    field_name: &str,
) -> Result<Option<i32>, PdbError> {
    let value = field(line, start, end);
    if value.trim().is_empty() {
        return Ok(None);
    }
    let decoded = hybrid36_decode(end - start, value).map_err(|error| PdbError::Parse {
        line: line_number,
        message: format!("invalid {field_name} {value:?}: {error}"),
    })?;
    i32::try_from(decoded)
        .map(Some)
        .map_err(|_| PdbError::Parse {
            line: line_number,
            message: format!("{field_name} {value:?} is outside the i32 range"),
        })
}

/// Decode a PDB hybrid-36 integer field.
///
/// Decimal fields use their ordinary signed representation.  Uppercase and
/// lowercase base-36 fields extend the range above the decimal limit using
/// the offsets defined by the PDB hybrid-36 convention.
pub fn hybrid36_decode(width: usize, value: &str) -> Result<i64, String> {
    if width == 0 {
        return Err("field width must be positive".to_string());
    }
    let value = value.trim();
    if value.is_empty() {
        return Err("field is empty".to_string());
    }
    let first = value.as_bytes()[0];
    if first == b'+' || first == b'-' || first.is_ascii_digit() {
        return value
            .parse::<i64>()
            .map_err(|error| format!("invalid decimal value: {error}"));
    }
    let digits = value.as_bytes();
    let mut pure = 0_i64;
    for &digit in digits {
        let digit = match digit {
            b'0'..=b'9' => i64::from(digit - b'0'),
            b'A'..=b'Z' => i64::from(digit - b'A' + 10),
            b'a'..=b'z' => i64::from(digit - b'a' + 10),
            _ => return Err(format!("invalid base-36 digit {:?}", char::from(digit))),
        };
        pure = pure
            .checked_mul(36)
            .and_then(|value| value.checked_add(digit))
            .ok_or_else(|| "base-36 value overflows i64".to_string())?;
    }
    let radix = 36_i64
        .checked_pow((width - 1) as u32)
        .ok_or_else(|| "field width overflows i64".to_string())?;
    let decimal_limit = 10_i64
        .checked_pow(width as u32)
        .ok_or_else(|| "field width overflows i64".to_string())?;
    if first.is_ascii_uppercase() {
        pure.checked_sub(10 * radix)
            .and_then(|value| value.checked_add(decimal_limit))
            .ok_or_else(|| "hybrid-36 value overflows i64".to_string())
    } else if first.is_ascii_lowercase() {
        pure.checked_add(16 * radix)
            .and_then(|value| value.checked_add(decimal_limit))
            .ok_or_else(|| "hybrid-36 value overflows i64".to_string())
    } else {
        Err(format!("invalid hybrid-36 prefix {:?}", char::from(first)))
    }
}

fn parse_optional<T>(
    line: &str,
    start: usize,
    end: usize,
    line_number: usize,
    field_name: &str,
) -> Result<Option<T>, PdbError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let value = field(line, start, end).trim();
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<T>()
        .map(Some)
        .map_err(|error| PdbError::Parse {
            line: line_number,
            message: format!("invalid {field_name} {value:?}: {error}"),
        })
}

fn format_atom(atom: &PdbAtom) -> String {
    let record = if atom.hetatm { "HETATM" } else { "ATOM  " };
    let atom_name = format_atom_name(atom);
    let residue_name = fit_field(&atom.residue_name, 4, false);
    let chain_id = atom.chain_id.unwrap_or(' ');
    let segid = atom
        .segid
        .as_deref()
        .map_or_else(|| "    ".to_owned(), |value| fit_field(value, 4, false));
    let alt_loc = atom.alt_loc.unwrap_or(' ');
    let insertion_code = atom.insertion_code.unwrap_or(' ');
    let occupancy = atom
        .occupancy
        .map_or_else(|| "      ".to_string(), |value| format!("{value:6.2}"));
    let temperature_factor = atom
        .temperature_factor
        .map_or_else(|| "      ".to_string(), |value| format!("{value:6.2}"));
    let element = atom
        .element
        .as_deref()
        .map_or_else(|| "  ".to_string(), |value| fit_field(value, 2, true));
    let charge = atom
        .charge
        .as_deref()
        .map_or_else(|| "  ".to_string(), |value| fit_field(value, 2, true));

    format!(
        "{record}{:>5} {atom_name}{alt_loc}{residue_name}{chain_id}{:>4}{insertion_code}   {:>8.3}{:>8.3}{:>8.3}{occupancy}{temperature_factor}      {segid}{element}{charge}",
        atom.serial, atom.residue_sequence, atom.x, atom.y, atom.z
    )
}

fn format_atom_name(atom: &PdbAtom) -> String {
    let name: String = atom.name.chars().take(4).collect();
    if name.chars().count() >= 4 {
        return name;
    }

    // PDB uses a leading blank for names whose element is one character
    // (e.g. " CA "), while two-character elements are left-aligned ("FE  ").
    let element_len = atom
        .element
        .as_deref()
        .map_or(1, |element| element.trim().chars().count());
    if element_len <= 1 {
        format!(" {name:<3}")
    } else {
        format!("{name:<4}")
    }
}

fn format_cryst1(cryst1: &PdbCryst1) -> String {
    let space_group = fit_field(&cryst1.space_group, 11, false);
    let z = cryst1
        .z
        .map_or_else(|| "    ".to_string(), |value| format!("{value:>4}"));
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

/// Errors produced while parsing or writing PDB data.
#[derive(Debug)]
pub enum PdbError {
    Io(io::Error),
    Parse {
        line: usize,
        message: String,
    },
    InconsistentModel {
        model: usize,
        expected: usize,
        found: usize,
    },
    InvalidStructure(String),
}

impl fmt::Display for PdbError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "PDB parse error on line {line}: {message}")
            }
            Self::InconsistentModel {
                model,
                expected,
                found,
            } => write!(
                formatter,
                "PDB model {model} contains {found} atoms; expected {expected}"
            ),
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid PDB structure: {message}")
            }
        }
    }
}

impl std::error::Error for PdbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for PdbError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SINGLE_MODEL: &str = concat!(
        "CRYST1   10.000   20.000   30.000  90.00  90.00 120.00 P 1          1\n",
        "ATOM      1  N   ALA A   1      1.000   2.000   3.000  1.00 20.00           N  \n",
        "HETATM    2  O   HOH A   2      4.000   5.000   6.000  0.50 10.00           O  \n",
        "END\n",
    );

    #[test]
    fn parses_atoms_and_cryst1() {
        let structure = PdbStructure::from_str(SINGLE_MODEL).expect("valid PDB");
        assert_eq!(structure.atoms.len(), 2);
        assert_eq!(structure.num_frames(), 1);
        assert_eq!(structure.atoms[0].serial, 1);
        assert_eq!(structure.atoms[0].residue_name, "ALA");
        assert!(structure.atoms[1].hetatm);
        assert_eq!(structure.atoms[1].occupancy, Some(0.5));
        assert_eq!(structure.cryst1.as_ref().map(|cell| cell.a), Some(10.0));
        assert_eq!(structure.frame(0).expect("frame")[1], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn parses_multiple_models() {
        let input = concat!(
            "MODEL        1\n",
            "ATOM      1  CA  GLY A   1      1.000   2.000   3.000  1.00 10.00           C  \n",
            "ENDMDL\n",
            "MODEL        2\n",
            "ATOM      1  CA  GLY A   1      7.000   8.000   9.000  1.00 10.00           C  \n",
            "ENDMDL\n",
        );
        let structure = PdbStructure::from_str(input).expect("valid models");
        assert_eq!(structure.num_frames(), 2);
        assert_eq!(structure.frame(0).expect("first frame")[0], [1.0, 2.0, 3.0]);
        assert_eq!(
            structure.frame(1).expect("second frame")[0],
            [7.0, 8.0, 9.0]
        );
        assert_eq!(structure.atoms_for_frame(1).expect("atoms")[0].x, 7.0);
    }

    #[test]
    fn accepts_models_without_endmdl_before_next_model_or_eof() {
        let input = concat!(
            "MODEL        1\n",
            "ATOM      1  CA  GLY A   1      1.000   2.000   3.000  1.00 10.00           C  \n",
            "MODEL        2\n",
            "ATOM      1  CA  GLY A   1      7.000   8.000   9.000  0.50 20.00           C  \n",
        );
        let structure = PdbStructure::from_str(input).expect("valid models");
        assert_eq!(structure.num_frames(), 2);
        assert_eq!(structure.atoms_for_frame(1).unwrap().len(), 1);
        assert_eq!(
            structure.atoms_for_frame(1).unwrap()[0].occupancy,
            Some(0.5)
        );
    }

    #[test]
    fn rejects_empty_nested_models() {
        let input = concat!("MODEL        1\n", "MODEL        2\n", "ENDMDL\n");
        let error = PdbStructure::from_str(input).expect_err("nested model");
        assert!(
            matches!(error, PdbError::Parse { message, .. } if message == "nested MODEL record")
        );
    }

    #[test]
    fn parses_and_writes_conect_records() {
        let input = concat!(
            "ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n",
            "ATOM      2  O   ALA A   1       2.000   2.000   3.000  1.00 10.00           O  \n",
            "CONECT    1    2\n",
            "END\n",
        );
        let structure = PdbStructure::from_str(input).expect("valid connectivity");
        assert_eq!(structure.bonds, vec![PdbBond::new(1, 2)]);
        let reparsed = PdbStructure::from_str(&structure.to_pdb_string().unwrap()).unwrap();
        assert_eq!(reparsed.bonds, structure.bonds);

        let compact = input.replace("CONECT    1    2", "CONECT 1 2");
        assert_eq!(
            PdbStructure::from_str(&compact).unwrap().bonds,
            vec![PdbBond::new(1, 2)]
        );
    }

    #[test]
    fn parses_compact_conect_serials() {
        let cases = [
            ("CONECT1233212331", 12_332, vec![12_331]),
            ("CONECT123331233112334", 12_333, vec![12_331, 12_334]),
            ("CONECT123341233312335", 12_334, vec![12_333, 12_335]),
            ("CONECT123351233412336", 12_335, vec![12_334, 12_336]),
            (
                "CONECT12336123271233012335",
                12_336,
                vec![12_327, 12_330, 12_335],
            ),
            (
                "CONECT12337 7718 84081234012344",
                12_337,
                vec![7_718, 8_408, 12_340, 12_344],
            ),
            (
                "CONECT1233812339123401234112345",
                12_338,
                vec![12_339, 12_340, 12_341, 12_345],
            ),
        ];
        for (line, first, partners) in cases {
            let bonds = parse_conect(line, 1).expect("valid CONECT record");
            assert_eq!(bonds[0].atom1, first);
            assert_eq!(
                bonds.iter().map(|bond| bond.atom2).collect::<Vec<_>>(),
                partners
            );
        }
        assert!(parse_conect("CONECT12337 7718 84081234012344123", 1).is_err());
    }

    #[test]
    fn defaults_blank_residue_sequence_to_one() {
        let input = concat!(
            "ATOM      1  H2  TIP3           10.000  44.891  14.267  1.00  0.00      TIP3\n",
            "ATOM      2  OH2 TIP3           67.275  48.893  23.568  1.00  0.00      TIP3\n",
            "END\n",
        );
        let structure = PdbStructure::from_str(input).expect("blank resid is valid");
        assert_eq!(structure.atoms.len(), 2);
        assert!(structure.atoms.iter().all(|atom| atom.resid() == 1));
        assert!(structure.atoms.iter().all(|atom| atom.resname() == "TIP3"));
    }

    #[test]
    fn parses_five_digit_xpdb_residue_numbers() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/5digitResid.pdb");
        let structure = PdbStructure::read_file(path).expect("valid XPDB fixture");
        assert_eq!(structure.atoms.len(), 5);
        assert_eq!(structure.atoms[3].residue_sequence, 1000);
        assert_eq!(structure.atoms[4].residue_sequence, 10000);
        assert_eq!(structure.atoms[4].insertion_code, None);
        assert_eq!(structure.atoms[4].element, None);
        assert_eq!(structure.frames.len(), 1);
    }

    #[test]
    fn restores_wrapped_standard_residue_sequences() {
        let mut first =
            "ATOM      1  CA  GLY A   1      1.000   2.000   3.000  1.00 10.00           C  "
                .to_owned();
        first.replace_range(22..26, "9999");
        let mut second = first.clone();
        second.replace_range(6..11, "    2");
        second.replace_range(22..26, "   0");
        let mut third = second.clone();
        third.replace_range(6..11, "    3");
        third.replace_range(22..26, "   1");
        let input = format!("{first}\n{second}\n{third}\nEND\n");

        let structure = PdbStructure::from_str(&input).expect("valid wrapped PDB");
        assert_eq!(
            structure
                .atoms
                .iter()
                .map(|atom| atom.residue_sequence)
                .collect::<Vec<_>>(),
            vec![9999, 10000, 10001]
        );
    }

    #[test]
    fn writes_and_reads_four_character_residue_names() {
        let input = concat!(
            "ATOM      1  H2  TIP3           10.000  44.891  14.267  1.00  0.00      TIP3\n",
            "END\n",
        );
        let structure = PdbStructure::from_str(input).expect("valid four-character residue");
        assert_eq!(structure.atoms[0].resname(), "TIP3");
        let reparsed = PdbStructure::from_str(&structure.to_pdb_string().unwrap()).unwrap();
        assert_eq!(reparsed.atoms[0].resname(), "TIP3");
    }

    #[test]
    fn decodes_hybrid36_fields() {
        let values = [
            ("A0000", 100_000),
            ("MEGAN", 20_929_695),
            ("J0NNY", 15_247_214),
            ("DREW6", 6_417_862),
            ("ST3V3", 31_691_119),
            ("ADA8M", 719_798),
            ("a0000", 43_770_016),
            ("megan", 64_599_711),
            ("j0nny", 58_917_230),
            ("drew6", 50_087_878),
            ("st3v3", 75_361_135),
            ("ada8m", 44_389_814),
            ("    6", 6),
            ("   24", 24),
            ("  645", 645),
            (" 4951", 4951),
            ("10267", 10267),
        ];
        for (encoded, expected) in values {
            assert_eq!(hybrid36_decode(5, encoded).unwrap(), expected);
        }
    }

    #[test]
    fn parses_overflowed_atom_serials() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/xl_serial.pdb");
        let structure = PdbStructure::read_file(path).expect("valid overflow serial fixture");
        assert_eq!(
            structure
                .atoms
                .iter()
                .map(|atom| atom.serial)
                .collect::<Vec<_>>(),
            [99_998, 99_999, 100_000, 100_001]
        );
    }

    #[test]
    fn parses_hybrid36_atom_serials() {
        let input = concat!(
            "REMARK For testing hybrid-36 atom numbers\n",
            "CRYST1   80.000   80.017   80.017  90.00  90.00  90.00 P 1           1\n",
            "MODEL        1\n",
            "HETATM    1  H     2 L 400      20.168  00.034  40.428\n",
            "HETATMA0000  H     2 L 400      40.168  50.034  40.428\n",
            "HETATMA0001  H     2 L 400      30.453  60.495  50.132\n",
            "HETATMA0002  H     2 L 400      20.576  40.354  60.483\n",
            "HETATMA0003  H     2 L 400      10.208  30.067  70.045\n",
            "ENDMDL\n",
        );
        let structure = PdbStructure::from_str(input).expect("valid hybrid-36 fixture");
        assert_eq!(structure.atoms.len(), 5);
        assert_eq!(structure.atoms[0].serial, 1);
        assert_eq!(structure.atoms[1].serial, 100_000);
        assert_eq!(structure.atoms[4].serial, 100_003);
    }

    #[test]
    fn parses_hybrid36_residue_sequences() {
        let mut line =
            "ATOM      1  CA  GLY A   1      1.000   2.000   3.000  1.00 10.00           C  "
                .to_string();
        line.replace_range(22..26, "A000");
        let input = format!("{line}\nEND\n");
        let structure = PdbStructure::from_str(&input).expect("valid hybrid-36 residue");
        assert_eq!(structure.atoms[0].residue_sequence, 10_000);
    }

    #[test]
    fn writes_and_reads_back_models() {
        let structure = PdbStructure::from_str(SINGLE_MODEL).expect("valid PDB");
        let text = structure.to_pdb_string().expect("write PDB");
        assert!(text.contains("ATOM      1  N   ALA A   1"));
        let reparsed = PdbStructure::from_str(&text).expect("read written PDB");
        assert_eq!(reparsed.atoms, structure.atoms);
        assert_eq!(reparsed.frames, structure.frames);
        assert_eq!(reparsed.cryst1, structure.cryst1);
    }

    #[test]
    fn supports_reader_api() {
        let structure = PdbStructure::read(Cursor::new(SINGLE_MODEL.as_bytes())).expect("read");
        assert_eq!(structure.atoms.len(), 2);
    }

    #[test]
    fn supports_file_api() {
        let path = std::env::temp_dir().join(format!(
            "mdanalysis_rs_pdb_{}_{}.pdb",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let structure = PdbStructure::from_str(SINGLE_MODEL).expect("valid PDB");
        structure.write_file(&path).expect("write file");
        let loaded = PdbStructure::read_file(&path).expect("read file");
        std::fs::remove_file(&path).expect("remove temporary file");
        assert_eq!(loaded, structure);
    }

    #[test]
    fn rejects_inconsistent_models() {
        let input = concat!(
            "MODEL        1\n",
            "ATOM      1  CA  GLY A   1      1.000   2.000   3.000\n",
            "ENDMDL\n",
            "MODEL        2\n",
            "ENDMDL\n",
        );
        let error = PdbStructure::from_str(input).expect_err("inconsistent models");
        assert!(matches!(error, PdbError::InconsistentModel { .. }));
    }

    #[test]
    fn ignores_partially_blank_cryst1_records() {
        let input = concat!(
            "CRYST1                            90.00  90.00  90.00 P 1           1\n",
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000\n",
            "END\n",
        );
        let structure = PdbStructure::from_str(input).expect("partial CRYST1 is ignorable");
        assert_eq!(structure.atoms.len(), 1);
        assert_eq!(structure.cryst1, None);
    }

    #[test]
    fn ignores_unitary_placeholder_cryst1_records() {
        let input = concat!(
            "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n",
            "ATOM      1  CA  GLY A   1       1.000   2.000   3.000\n",
            "END\n",
        );
        let structure = PdbStructure::from_str(input).expect("unitary CRYST1 is a placeholder");
        assert_eq!(structure.cryst1, None);
    }

    #[test]
    fn rejects_coordinates_outside_fixed_width_range() {
        let mut structure = PdbStructure::from_str(SINGLE_MODEL).expect("valid PDB");
        structure.frames[0][0][0] = 10_000.0;
        assert!(matches!(
            structure.to_pdb_string(),
            Err(PdbError::InvalidStructure(_))
        ));
    }
}
