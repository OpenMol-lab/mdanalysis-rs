//! Tinker XYZ (`.txyz`) and ARC trajectory support.
//!
//! A Tinker frame starts with an atom count and title, optionally followed by
//! a periodic box (`a b c` or `a b c alpha beta gamma`), then atom records of
//! the form `id name x y z type neighbor...`.  ARC files concatenate those
//! frames.  The first frame supplies the topology; subsequent frames must
//! contain the same atom IDs; labels, types, and connectivity are retained
//! from the first frame for topology construction.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Atom metadata and coordinates from a Tinker XYZ frame.
#[derive(Clone, Debug, PartialEq)]
pub struct TxyzAtom {
    /// Tinker atom number.  IDs are retained even when the source is sparse.
    pub id: usize,
    /// Atom label, such as `C`, `CH2`, or `OW`.
    pub name: String,
    /// Tinker atom class/type token.  Tinker commonly uses integer strings,
    /// but preserving the token supports parameter files that use labels.
    pub atom_type: String,
    pub position: [f64; 3],
}

/// An undirected Tinker bond expressed in source atom IDs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TxyzBond {
    pub atom1: usize,
    pub atom2: usize,
}

impl TxyzBond {
    fn canonical(atom1: usize, atom2: usize) -> Self {
        if atom1 <= atom2 {
            Self { atom1, atom2 }
        } else {
            Self {
                atom1: atom2,
                atom2: atom1,
            }
        }
    }
}

/// A Tinker XYZ or ARC document, including first-frame topology and all
/// coordinate frames.
#[derive(Clone, Debug, PartialEq)]
pub struct TxyzFile {
    /// Title from the first frame header.
    pub title: String,
    /// Atom metadata and first-frame coordinates, sorted by atom ID.
    pub atoms: Vec<TxyzAtom>,
    /// Unique undirected bonds in source atom-ID space.
    pub bonds: Vec<TxyzBond>,
    /// All frames, including the first frame represented by [`Self::atoms`].
    pub coordinates: CoordinateFile,
}

/// Alias for callers that use the topology terminology of other formats.
pub type TxyzStructure = TxyzFile;
/// Alias for callers that prefer a data-file name.
pub type TxyzData = TxyzFile;
/// Alias for ARC trajectories.
pub type ArcFile = TxyzFile;

impl TxyzFile {
    /// Parse a Tinker XYZ or ARC document from a string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, TxyzError> {
        parse_txyz(input)
    }

    /// Read a Tinker document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, TxyzError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Read a Tinker document from a filesystem path.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, TxyzError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    /// Build a topology-free Tinker document from coordinate frames.
    ///
    /// Names and atom IDs are taken from the first frame when available;
    /// missing metadata receives `X`, sequential IDs, and atom type `1`.
    pub fn from_coordinates(coordinates: CoordinateFile) -> Result<Self, TxyzError> {
        let first = coordinates
            .frames
            .first()
            .ok_or_else(|| TxyzError::InvalidStructure("document has no frames".to_owned()))?;
        let count = first.positions.len();
        let ids = if first.atom_ids.len() == count {
            first.atom_ids.clone()
        } else {
            (1..=count).collect()
        };
        let atoms = first
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| TxyzAtom {
                id: ids[index],
                name: first
                    .names
                    .get(index)
                    .filter(|name| !name.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| "X".to_owned()),
                atom_type: "1".to_owned(),
                position: *position,
            })
            .collect();
        let file = Self {
            title: first.title.clone(),
            atoms,
            bonds: Vec::new(),
            coordinates,
        };
        validate_txyz(&file)?;
        Ok(file)
    }

    /// Serialize the document as Tinker XYZ/ARC text.
    pub fn to_string(&self) -> Result<String, TxyzError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output).map_err(|error| {
            TxyzError::InvalidStructure(format!("Tinker output is not UTF-8: {error}"))
        })
    }

    /// Write the document to any writer.
    pub fn write<W: Write>(&self, writer: W) -> Result<(), TxyzError> {
        validate_txyz(self)?;
        write_txyz_document(self, writer)
    }

    /// Write the document to a filesystem path.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), TxyzError> {
        self.write(File::create(path)?)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.frames.len()
    }

    /// Return the first frame's unit-cell dimensions, if present.
    #[must_use]
    pub fn dimensions(&self) -> Option<[f64; 6]> {
        self.coordinates
            .frames
            .first()
            .and_then(|frame| frame.dimensions)
    }
}

impl std::str::FromStr for TxyzFile {
    type Err = TxyzError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Read a Tinker XYZ or ARC document from a path.
pub fn read_txyz<P: AsRef<Path>>(path: P) -> Result<TxyzFile, TxyzError> {
    TxyzFile::read_file(path)
}

/// Write a Tinker XYZ or ARC document to a path.
pub fn write_txyz<P: AsRef<Path>>(path: P, file: &TxyzFile) -> Result<(), TxyzError> {
    file.write_file(path)
}

/// ARC aliases for the generic Tinker XYZ path helpers.
pub fn read_arc<P: AsRef<Path>>(path: P) -> Result<TxyzFile, TxyzError> {
    read_txyz(path)
}

pub fn write_arc<P: AsRef<Path>>(path: P, file: &TxyzFile) -> Result<(), TxyzError> {
    write_txyz(path, file)
}

impl CoordinateFile {
    /// Parse Tinker XYZ/ARC and return only its coordinate frames.
    pub fn read_txyz<R: Read>(reader: R) -> Result<Self, TxyzError> {
        Ok(TxyzFile::read(reader)?.coordinates)
    }

    /// Parse Tinker XYZ/ARC text and return only its coordinate frames.
    pub fn from_txyz_str(input: &str) -> Result<Self, TxyzError> {
        Ok(TxyzFile::from_str(input)?.coordinates)
    }

    /// Parse Tinker ARC text and return only its coordinate frames.
    pub fn from_arc_str(input: &str) -> Result<Self, TxyzError> {
        Self::from_txyz_str(input)
    }

    /// Write coordinate frames with default atom types and no bonds.
    pub fn write_txyz<W: Write>(&self, writer: W) -> Result<(), TxyzError> {
        TxyzFile::from_coordinates(self.clone())?.write(writer)
    }

    /// Serialize coordinate frames with default atom types and no bonds.
    pub fn to_txyz_string(&self) -> Result<String, TxyzError> {
        let mut output = Vec::new();
        self.write_txyz(&mut output)?;
        String::from_utf8(output).map_err(|error| {
            TxyzError::InvalidStructure(format!("Tinker output is not UTF-8: {error}"))
        })
    }
}

/// Errors produced while reading or writing Tinker XYZ/ARC files.
#[derive(Debug)]
pub enum TxyzError {
    Io(io::Error),
    Coordinate(CoordinateError),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for TxyzError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Coordinate(error) => write!(formatter, "coordinate error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "TXYZ parse error on line {line}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid Tinker XYZ structure: {message}")
            }
        }
    }
}

impl std::error::Error for TxyzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Coordinate(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for TxyzError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CoordinateError> for TxyzError {
    fn from(error: CoordinateError) -> Self {
        Self::Coordinate(error)
    }
}

impl crate::core::Universe {
    /// Construct a universe from a Tinker XYZ/ARC path.
    pub fn from_txyz(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_txyz_file(TxyzFile::read_file(path)?)
    }

    /// Construct a universe from Tinker XYZ/ARC text.
    pub fn from_txyz_str(input: &str) -> crate::Result<Self> {
        Self::from_txyz_file(TxyzFile::from_str(input)?)
    }

    /// Construct a universe from parsed Tinker XYZ/ARC data.
    pub fn from_txyz_file(data: TxyzFile) -> crate::Result<Self> {
        validate_txyz(&data)?;
        let first_count = data
            .coordinates
            .frames
            .first()
            .ok_or_else(|| {
                crate::Error::InvalidInput("Tinker XYZ file has no coordinate frames".to_owned())
            })?
            .positions
            .len();
        let mut atoms = Vec::with_capacity(data.atoms.len());
        for (index, source) in data.atoms.iter().enumerate() {
            let mut atom = crate::core::Atom::new(index, source.name.clone(), source.position);
            atom.atom_type = Some(source.atom_type.clone());
            atom.element = infer_element(&source.name);
            atom.mass = crate::guesser::guess_atom_mass(&source.name);
            atom.resid = 1;
            atom.resname = "SYSTEM".to_owned();
            atoms.push(atom);
        }
        let mut topology = crate::core::Topology::new(atoms);
        let id_to_index = data
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom.id, index))
            .collect::<HashMap<_, _>>();
        for bond in &data.bonds {
            let atom1 = id_to_index.get(&bond.atom1).copied().ok_or_else(|| {
                crate::Error::InvalidInput("Tinker bond references unknown atom".to_owned())
            })?;
            let atom2 = id_to_index.get(&bond.atom2).copied().ok_or_else(|| {
                crate::Error::InvalidInput("Tinker bond references unknown atom".to_owned())
            })?;
            topology.add_bond(crate::core::Bond::new(atom1, atom2));
        }
        let frames = data
            .coordinates
            .frames
            .into_iter()
            .map(|source| {
                let mut frame = crate::core::Frame::new(source.positions);
                frame.velocities = source.velocities;
                frame.dimensions = source.dimensions;
                frame.step = source.step;
                frame.time = source.time;
                frame
            })
            .collect();
        if first_count != data.atoms.len() {
            return Err(crate::Error::InvalidInput(
                "Tinker topology and first frame atom counts differ".to_owned(),
            ));
        }
        Ok(Self {
            topology,
            trajectory: crate::core::Trajectory::new(frames),
        })
    }

    /// ARC aliases for the Tinker XYZ constructors.
    pub fn from_arc(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_txyz(path)
    }

    pub fn from_arc_str(input: &str) -> crate::Result<Self> {
        Self::from_txyz_str(input)
    }

    pub fn from_arc_file(data: TxyzFile) -> crate::Result<Self> {
        Self::from_txyz_file(data)
    }
}

fn parse_txyz(input: &str) -> Result<TxyzFile, TxyzError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut cursor = 0;
    let mut coordinate_frames = Vec::new();
    let mut topology_atoms: Option<Vec<TxyzAtom>> = None;
    let mut topology_bonds: Option<Vec<TxyzBond>> = None;
    let mut first_title = None;

    while let Some((line_number, header)) = next_nonempty(&lines, &mut cursor) {
        let (atom_count, title) = parse_header_line(header, line_number)?;
        if atom_count == 0 {
            return Err(parse_error(
                line_number,
                "frame must contain at least one atom",
            ));
        }
        let dimensions = if cursor < lines.len() && is_box_line(lines[cursor]) {
            let box_line = lines[cursor];
            let box_number = cursor + 1;
            cursor += 1;
            Some(parse_box_line(box_line, box_number)?)
        } else {
            None
        };

        let mut records = Vec::with_capacity(atom_count);
        let mut seen_ids = HashSet::with_capacity(atom_count);
        for _ in 0..atom_count {
            if cursor >= lines.len() {
                return Err(parse_error(
                    lines.len() + 1,
                    format!("frame declares {atom_count} atoms but file ends early"),
                ));
            }
            let line_number = cursor + 1;
            let line = lines[cursor];
            cursor += 1;
            if line.trim().is_empty() {
                return Err(parse_error(line_number, "blank atom record"));
            }
            let (atom, neighbors) = parse_atom_line(line, line_number)?;
            if !seen_ids.insert(atom.id) {
                return Err(parse_error(
                    line_number,
                    format!("duplicate atom ID {}", atom.id),
                ));
            }
            records.push((atom, neighbors));
        }
        records.sort_by_key(|(atom, _)| atom.id);
        let frame_atoms = records
            .iter()
            .map(|(atom, _)| atom.clone())
            .collect::<Vec<_>>();
        let frame_bonds = collect_bonds(&records, &seen_ids)?;

        if let Some(expected) = &topology_atoms {
            if expected.len() != frame_atoms.len() {
                return Err(TxyzError::InvalidStructure(format!(
                    "frame {} contains {} atoms; expected {}",
                    coordinate_frames.len() + 1,
                    frame_atoms.len(),
                    expected.len()
                )));
            }
            if expected
                .iter()
                .zip(&frame_atoms)
                .any(|(reference, current)| reference.id != current.id)
            {
                return Err(TxyzError::InvalidStructure(format!(
                    "frame {} atom IDs differ from first frame",
                    coordinate_frames.len() + 1
                )));
            }
        } else {
            topology_atoms = Some(frame_atoms.clone());
            topology_bonds = Some(frame_bonds.clone());
            first_title = Some(title.clone());
        }

        let mut frame = CoordinateFrame::new(
            frame_atoms
                .iter()
                .map(|atom| atom.position)
                .collect::<Vec<_>>(),
        );
        frame.names = frame_atoms.iter().map(|atom| atom.name.clone()).collect();
        frame.atom_ids = frame_atoms.iter().map(|atom| atom.id).collect();
        frame.title = title;
        frame.dimensions = dimensions;
        coordinate_frames.push(frame);
    }

    let atoms = topology_atoms
        .ok_or_else(|| TxyzError::InvalidStructure("document has no frames".to_owned()))?;
    let bonds = topology_bonds.unwrap_or_default();
    let file = TxyzFile {
        title: first_title.unwrap_or_default(),
        atoms,
        bonds,
        coordinates: CoordinateFile::new(coordinate_frames),
    };
    validate_txyz(&file)?;
    Ok(file)
}

fn next_nonempty<'a>(lines: &'a [&str], cursor: &mut usize) -> Option<(usize, &'a str)> {
    while *cursor < lines.len() {
        let line_number = *cursor + 1;
        let line = lines[*cursor];
        *cursor += 1;
        if !line.trim().is_empty() {
            return Some((line_number, line));
        }
    }
    None
}

fn parse_header_line(line: &str, line_number: usize) -> Result<(usize, String), TxyzError> {
    let mut fields = line.split_whitespace();
    let count = fields
        .next()
        .ok_or_else(|| parse_error(line_number, "missing atom count"))?
        .parse::<usize>()
        .map_err(|error| parse_error(line_number, format!("invalid atom count: {error}")))?;
    let title = fields.collect::<Vec<_>>().join(" ");
    Ok((count, title))
}

fn is_box_line(line: &str) -> bool {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    (fields.len() == 3 || fields.len() == 6)
        && fields.iter().all(|field| field.parse::<f64>().is_ok())
}

fn parse_box_line(line: &str, line_number: usize) -> Result<[f64; 6], TxyzError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    let values = fields
        .iter()
        .map(|field| {
            field
                .parse::<f64>()
                .map_err(|error| parse_error(line_number, format!("invalid box value: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dimensions = match values.as_slice() {
        [a, b, c] => [*a, *b, *c, 90.0, 90.0, 90.0],
        [a, b, c, alpha, beta, gamma] => [*a, *b, *c, *alpha, *beta, *gamma],
        _ => return Err(parse_error(line_number, "box must contain 3 or 6 values")),
    };
    validate_dimensions(dimensions).map_err(|message| parse_error(line_number, message))?;
    Ok(dimensions)
}

fn parse_atom_line(line: &str, line_number: usize) -> Result<(TxyzAtom, Vec<usize>), TxyzError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 6 {
        return Err(parse_error(
            line_number,
            "atom record requires id, name, x, y, z, and type",
        ));
    }
    let id = fields[0]
        .parse::<usize>()
        .map_err(|error| parse_error(line_number, format!("invalid atom ID: {error}")))?;
    if id == 0 {
        return Err(parse_error(line_number, "atom IDs must be positive"));
    }
    let name = fields[1].to_owned();
    if name.is_empty() {
        return Err(parse_error(line_number, "atom name is empty"));
    }
    let position = [
        parse_float(fields[2], line_number, "x coordinate")?,
        parse_float(fields[3], line_number, "y coordinate")?,
        parse_float(fields[4], line_number, "z coordinate")?,
    ];
    if position.iter().any(|value| !value.is_finite()) {
        return Err(parse_error(line_number, "coordinates must be finite"));
    }
    let atom_type = fields[5].to_owned();
    if atom_type.is_empty() {
        return Err(parse_error(line_number, "atom type is empty"));
    }
    let mut neighbors = Vec::with_capacity(fields.len().saturating_sub(6));
    for field in &fields[6..] {
        let neighbor = field.parse::<usize>().map_err(|error| {
            parse_error(line_number, format!("invalid bonded atom ID: {error}"))
        })?;
        if neighbor == 0 || neighbor == id {
            return Err(parse_error(
                line_number,
                "bonded atom IDs must be positive and distinct",
            ));
        }
        neighbors.push(neighbor);
    }
    Ok((
        TxyzAtom {
            id,
            name,
            atom_type,
            position,
        },
        neighbors,
    ))
}

fn collect_bonds(
    records: &[(TxyzAtom, Vec<usize>)],
    atom_ids: &HashSet<usize>,
) -> Result<Vec<TxyzBond>, TxyzError> {
    let mut bonds = HashSet::new();
    for (atom, neighbors) in records {
        for &neighbor in neighbors {
            if !atom_ids.contains(&neighbor) {
                return Err(TxyzError::InvalidStructure(format!(
                    "atom {} references unknown bonded atom {}",
                    atom.id, neighbor
                )));
            }
            bonds.insert(TxyzBond::canonical(atom.id, neighbor));
        }
    }
    let mut bonds = bonds.into_iter().collect::<Vec<_>>();
    bonds.sort_by_key(|bond| (bond.atom1, bond.atom2));
    Ok(bonds)
}

fn write_txyz_document<W: Write>(file: &TxyzFile, mut writer: W) -> Result<(), TxyzError> {
    let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
    for bond in &file.bonds {
        neighbors.entry(bond.atom1).or_default().push(bond.atom2);
        neighbors.entry(bond.atom2).or_default().push(bond.atom1);
    }
    for values in neighbors.values_mut() {
        values.sort_unstable();
    }
    for (frame_index, frame) in file.coordinates.frames.iter().enumerate() {
        let title = if frame.title.trim().is_empty() {
            file.title.trim()
        } else {
            frame.title.trim()
        };
        if title.contains(['\n', '\r']) {
            return Err(TxyzError::InvalidStructure(format!(
                "frame {} title must be a single line",
                frame_index + 1
            )));
        }
        if title.is_empty() {
            writeln!(writer, "{}", file.atoms.len())?;
        } else {
            writeln!(writer, "{} {}", file.atoms.len(), title)?;
        }
        if let Some(dimensions) = frame.dimensions {
            writeln!(
                writer,
                "{:.12e} {:.12e} {:.12e} {:.12e} {:.12e} {:.12e}",
                dimensions[0],
                dimensions[1],
                dimensions[2],
                dimensions[3],
                dimensions[4],
                dimensions[5]
            )?;
        }
        for (index, atom) in file.atoms.iter().enumerate() {
            let position = frame.positions[index];
            write!(
                writer,
                "{} {} {:.12e} {:.12e} {:.12e} {}",
                atom.id, atom.name, position[0], position[1], position[2], atom.atom_type
            )?;
            if let Some(atom_neighbors) = neighbors.get(&atom.id) {
                for neighbor in atom_neighbors {
                    write!(writer, " {neighbor}")?;
                }
            }
            writeln!(writer)?;
        }
    }
    Ok(())
}

fn validate_txyz(file: &TxyzFile) -> Result<(), TxyzError> {
    if file.atoms.is_empty() || file.coordinates.frames.is_empty() {
        return Err(TxyzError::InvalidStructure(
            "Tinker document must contain atoms and at least one frame".to_owned(),
        ));
    }
    if file.title.contains(['\n', '\r']) {
        return Err(TxyzError::InvalidStructure(
            "title must be a single line".to_owned(),
        ));
    }
    let mut ids = HashSet::with_capacity(file.atoms.len());
    for atom in &file.atoms {
        if atom.id == 0 || !ids.insert(atom.id) {
            return Err(TxyzError::InvalidStructure(
                "atom IDs must be positive and unique".to_owned(),
            ));
        }
        if atom.name.trim().is_empty()
            || atom.name.split_whitespace().count() != 1
            || atom.name.contains(['\n', '\r'])
        {
            return Err(TxyzError::InvalidStructure(
                "atom names must be non-empty single tokens".to_owned(),
            ));
        }
        if atom.atom_type.trim().is_empty()
            || atom.atom_type.split_whitespace().count() != 1
            || atom.atom_type.contains(['\n', '\r'])
        {
            return Err(TxyzError::InvalidStructure(
                "atom types must be non-empty single tokens".to_owned(),
            ));
        }
        if atom.position.iter().any(|value| !value.is_finite()) {
            return Err(TxyzError::InvalidStructure(
                "atom coordinates must be finite".to_owned(),
            ));
        }
    }
    let mut bonds = HashSet::new();
    for bond in &file.bonds {
        if bond.atom1 == 0
            || bond.atom2 == 0
            || bond.atom1 == bond.atom2
            || !ids.contains(&bond.atom1)
            || !ids.contains(&bond.atom2)
        {
            return Err(TxyzError::InvalidStructure(
                "bonds must reference distinct known atom IDs".to_owned(),
            ));
        }
        if !bonds.insert(TxyzBond::canonical(bond.atom1, bond.atom2)) {
            return Err(TxyzError::InvalidStructure(
                "bonds must be unique".to_owned(),
            ));
        }
    }
    for (frame_index, frame) in file.coordinates.frames.iter().enumerate() {
        if frame.positions.len() != file.atoms.len() {
            return Err(TxyzError::InvalidStructure(format!(
                "frame {} contains {} atoms; expected {}",
                frame_index + 1,
                frame.positions.len(),
                file.atoms.len()
            )));
        }
        if frame
            .positions
            .iter()
            .flat_map(|position| position.iter())
            .any(|value| !value.is_finite())
        {
            return Err(TxyzError::InvalidStructure(format!(
                "frame {} coordinates must be finite",
                frame_index + 1
            )));
        }
        if let Some(dimensions) = frame.dimensions {
            validate_dimensions(dimensions).map_err(|message| {
                TxyzError::InvalidStructure(format!("frame {}: {message}", frame_index + 1))
            })?;
        }
    }
    Ok(())
}

fn validate_dimensions(dimensions: [f64; 6]) -> Result<(), String> {
    if dimensions.iter().any(|value| !value.is_finite()) {
        return Err("box dimensions must be finite".to_owned());
    }
    if dimensions[..3].iter().any(|value| *value <= 0.0) {
        return Err("box lengths must be positive".to_owned());
    }
    if dimensions[3..]
        .iter()
        .any(|value| *value <= 0.0 || *value >= 180.0)
    {
        return Err("box angles must be between 0 and 180 degrees".to_owned());
    }
    Ok(())
}

fn infer_element(name: &str) -> Option<String> {
    let mut chars = name.chars();
    let first = chars.next()?.to_ascii_uppercase();
    let second = chars.next().filter(char::is_ascii_lowercase);
    let symbol = second.map_or_else(|| first.to_string(), |value| format!("{first}{value}"));
    matches!(
        symbol.as_str(),
        "H" | "C"
            | "N"
            | "O"
            | "F"
            | "P"
            | "S"
            | "Cl"
            | "Br"
            | "I"
            | "Na"
            | "Mg"
            | "Si"
            | "K"
            | "Ca"
            | "Fe"
            | "Zn"
            | "Cu"
    )
    .then_some(symbol)
}

fn parse_float(value: &str, line_number: usize, field: &str) -> Result<f64, TxyzError> {
    value
        .parse::<f64>()
        .map_err(|error| parse_error(line_number, format!("invalid {field}: {error}")))
}

fn parse_error(line: usize, message: impl Into<String>) -> TxyzError {
    TxyzError::Parse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const TXYZ: &str = concat!(
        "3 methane\n",
        "1 C 0.0 0.0 0.0 1 2 3\n",
        "2 H 1.0 0.0 0.0 2 1\n",
        "3 H -1.0 0.0 0.0 2 1\n",
    );

    const ARC_PBC: &str = concat!(
        "2 frame one\n",
        "10.0 10.0 10.0 90.0 90.0 90.0\n",
        "1 O 0.0 0.0 0.0 8 2\n",
        "2 H 1.0 0.0 0.0 9 1\n",
        "2 frame two\n",
        "11.0 10.0 10.0 90.0 90.0 90.0\n",
        "1 O 0.1 0.0 0.0 8 2\n",
        "2 H 1.1 0.0 0.0 9 1\n",
    );

    const ARC_DYNAMIC_METADATA: &str = concat!(
        "2 frame one\n",
        "1 C 0.0 0.0 0.0 1 2\n",
        "2 H 1.0 0.0 0.0 2 1\n",
        "2 frame two\n",
        "1 CA 0.2 0.0 0.0 7\n",
        "2 O 1.2 0.0 0.0 8\n",
    );

    #[test]
    fn reads_txyz_atoms_types_bonds_and_coordinates() {
        let file = TxyzFile::from_str(TXYZ).unwrap();
        assert_eq!(file.n_atoms(), 3);
        assert_eq!(file.n_frames(), 1);
        assert_eq!(file.atoms[0].atom_type, "1");
        assert_eq!(
            file.bonds,
            vec![TxyzBond::canonical(1, 2), TxyzBond::canonical(1, 3)]
        );
        assert_eq!(file.coordinates.frames[0].names, vec!["C", "H", "H"]);
    }

    #[test]
    fn universe_guesses_masses_from_txyz_atom_labels() {
        let universe = crate::core::Universe::from_txyz_str(TXYZ).unwrap();
        assert_eq!(
            universe
                .topology
                .atoms
                .iter()
                .map(|atom| atom.mass)
                .collect::<Vec<_>>(),
            vec![12.011, 1.008, 1.008]
        );
    }

    #[test]
    fn reads_multiframe_arc_and_periodic_boxes() {
        let file = TxyzFile::read(Cursor::new(ARC_PBC.as_bytes())).unwrap();
        assert_eq!(file.n_frames(), 2);
        assert_eq!(
            file.coordinates.frames[0].dimensions.unwrap()[..3],
            [10.0, 10.0, 10.0]
        );
        assert_eq!(file.coordinates.frames[1].positions[0], [0.1, 0.0, 0.0]);
        let universe = crate::core::Universe::from_txyz_file(file).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.trajectory.n_frames(), 2);
        assert_eq!(universe.topology.bonds.len(), 1);
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("8"));
        assert_eq!(universe.trajectory.frames[1].dimensions.unwrap()[0], 11.0);
    }

    #[test]
    fn round_trip_preserves_multiframe_boxes_and_bonds() {
        let file = TxyzFile::from_str(ARC_PBC).unwrap();
        let output = file.to_string().unwrap();
        let reparsed = TxyzFile::from_str(&output).unwrap();
        assert_eq!(reparsed.atoms, file.atoms);
        assert_eq!(reparsed.bonds, file.bonds);
        assert_eq!(reparsed.coordinates.frames, file.coordinates.frames);
    }

    #[test]
    fn later_arc_frames_may_change_atom_metadata() {
        let file = TxyzFile::from_str(ARC_DYNAMIC_METADATA).unwrap();
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.atoms[0].name, "C");
        assert_eq!(file.atoms[0].atom_type, "1");
        assert_eq!(file.bonds, vec![TxyzBond::canonical(1, 2)]);
        assert_eq!(file.coordinates.frames[1].positions[0], [0.2, 0.0, 0.0]);
    }

    #[test]
    fn malformed_records_are_rejected() {
        assert!(TxyzFile::from_str("2\nno title\n1 C 0 0 0 1\n").is_err());
        assert!(TxyzFile::from_str("1\nbox\n1 1 1 90 90 90\n1 C 0 0 0 1\n").is_err());
        assert!(TxyzFile::from_str("1\natom\n1 C NaN 0 0 1\n").is_err());
    }
}
