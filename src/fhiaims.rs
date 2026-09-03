//! FHI-AIMS geometry.in support.
//!
//! The format is a small text representation of a single geometry. This
//! module handles Cartesian (atom) and fractional (atom_frac) positions,
//! optional triclinic lattice vectors, and per-atom velocities.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::mdamath::{triclinic_box, triclinic_vectors};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// A parsed FHI-AIMS geometry file.
#[derive(Clone, Debug, PartialEq)]
pub struct FhiaimsFile {
    /// The single coordinate frame represented by the input file.
    pub coordinates: CoordinateFile,
    /// Original lattice vectors, when the input supplied a complete cell.
    /// Vectors are rows in Cartesian coordinates.
    pub lattice_vectors: Option<[[f64; 3]; 3]>,
}

pub type FhiaimsStructure = FhiaimsFile;
pub type FhiaimsData = FhiaimsFile;

impl FhiaimsFile {
    /// Parse a FHI-AIMS geometry document from text.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, FhiaimsError> {
        parse_fhiaims(input)
    }

    /// Read a FHI-AIMS geometry document from a reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, FhiaimsError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Read a FHI-AIMS geometry document from a path.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, FhiaimsError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    /// Serialize the geometry using absolute Cartesian atom positions.
    pub fn to_string(&self) -> Result<String, FhiaimsError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output).map_err(|error| {
            FhiaimsError::InvalidStructure(format!("FHI-AIMS output is not UTF-8: {error}"))
        })
    }

    /// Write the geometry to a writer.
    pub fn write<W: Write>(&self, writer: W) -> Result<(), FhiaimsError> {
        write_document(self, writer)
    }

    /// Write the geometry to a path.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), FhiaimsError> {
        self.write(File::create(path)?)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.coordinates.n_atoms()
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    /// Return cell dimensions in [a, b, c, alpha, beta, gamma] form.
    #[must_use]
    pub fn dimensions(&self) -> Option<[f64; 6]> {
        self.coordinates
            .frames
            .first()
            .and_then(|frame| frame.dimensions)
    }
}

impl std::str::FromStr for FhiaimsFile {
    type Err = FhiaimsError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Errors produced while reading or writing FHI-AIMS geometry files.
#[derive(Debug)]
pub enum FhiaimsError {
    Io(io::Error),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for FhiaimsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "FHI-AIMS parse error on line {line}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid FHI-AIMS structure: {message}")
            }
        }
    }
}

impl std::error::Error for FhiaimsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { .. } | Self::InvalidStructure(_) => None,
        }
    }
}

impl From<io::Error> for FhiaimsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Read a FHI-AIMS geometry file from a path.
pub fn read_fhiaims<P: AsRef<Path>>(path: P) -> Result<FhiaimsFile, FhiaimsError> {
    FhiaimsFile::read_file(path)
}

/// Write a FHI-AIMS geometry file to a path.
pub fn write_fhiaims_file<P: AsRef<Path>>(path: P, file: &FhiaimsFile) -> Result<(), FhiaimsError> {
    file.write_file(path)
}

/// Write a FHI-AIMS geometry file to a path.
pub fn write_fhiaims<P: AsRef<Path>>(path: P, file: &FhiaimsFile) -> Result<(), FhiaimsError> {
    write_fhiaims_file(path, file)
}

impl CoordinateFile {
    /// Read FHI-AIMS text and retain only its coordinate frame.
    pub fn read_fhiaims<R: Read>(reader: R) -> Result<Self, FhiaimsError> {
        Ok(FhiaimsFile::read(reader)?.coordinates)
    }

    /// Parse FHI-AIMS text and retain only its coordinate frame.
    pub fn from_fhiaims_str(input: &str) -> Result<Self, FhiaimsError> {
        Ok(FhiaimsFile::from_str(input)?.coordinates)
    }

    /// Write a coordinate frame as FHI-AIMS geometry text.
    pub fn to_fhiaims_string(&self) -> Result<String, FhiaimsError> {
        FhiaimsFile::from_coordinates(self.clone())?.to_string()
    }
}

impl FhiaimsFile {
    /// Build a FHI-AIMS document from a single coordinate frame.
    pub fn from_coordinates(coordinates: CoordinateFile) -> Result<Self, FhiaimsError> {
        if coordinates.frames.len() != 1 {
            return Err(FhiaimsError::InvalidStructure(
                "FHI-AIMS supports exactly one coordinate frame".to_owned(),
            ));
        }
        let lattice_vectors = coordinates.frames[0].dimensions.map(triclinic_vectors);
        let file = Self {
            coordinates,
            lattice_vectors,
        };
        validate_fhiaims(&file)?;
        Ok(file)
    }
}

impl crate::core::Universe {
    /// Construct a universe from an FHI-AIMS geometry file.
    pub fn from_fhiaims(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_fhiaims_file(FhiaimsFile::read_file(path)?)
    }

    /// Construct a universe from FHI-AIMS text held in memory.
    pub fn from_fhiaims_str(input: &str) -> crate::Result<Self> {
        Self::from_fhiaims_file(FhiaimsFile::from_str(input)?)
    }

    /// Construct a universe from parsed FHI-AIMS data.
    pub fn from_fhiaims_file(file: FhiaimsFile) -> crate::Result<Self> {
        validate_fhiaims(&file)?;
        let frame = file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("FHI-AIMS file has no coordinate frame".to_owned())
        })?;
        let atoms = frame
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| {
                let name = frame
                    .names
                    .get(index)
                    .filter(|name| !name.trim().is_empty())
                    .map_or("X", String::as_str);
                let mut atom = crate::core::Atom::new(index, name, *position);
                atom.element = crate::guesser::guess_element(name, None, None).ok();
                atom.mass = crate::guesser::guess_atom_mass(name);
                atom.resid = 1;
                atom.resname = "SYSTEM".to_owned();
                atom
            })
            .collect();
        let topology = crate::core::Topology::new(atoms);
        let frames = file
            .coordinates
            .frames
            .into_iter()
            .map(|source| {
                let mut result = crate::core::Frame::new(source.positions);
                result.velocities = source.velocities;
                result.dimensions = source.dimensions;
                result.step = source.step;
                result.time = source.time;
                result
            })
            .collect();
        Ok(Self {
            topology,
            trajectory: crate::core::Trajectory::new(frames),
        })
    }
}

#[derive(Clone, Debug)]
struct PendingAtom {
    coordinates: [f64; 3],
    fractional: bool,
    name: String,
    velocity: Option<[f64; 3]>,
}

fn parse_fhiaims(input: &str) -> Result<FhiaimsFile, FhiaimsError> {
    let mut lattice = Vec::new();
    let mut atoms = Vec::new();
    let mut last_was_atom = false;

    for (offset, raw_line) in input.lines().enumerate() {
        let line_number = offset + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            last_was_atom = false;
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        match fields[0].to_ascii_lowercase().as_str() {
            "lattice_vector" => {
                if fields.len() != 4 {
                    return Err(parse_error(
                        line_number,
                        "lattice_vector requires exactly three components",
                    ));
                }
                if lattice.len() == 3 {
                    return Err(parse_error(
                        line_number,
                        "at most three lattice vectors are allowed",
                    ));
                }
                lattice.push([
                    parse_finite(fields[1], line_number, "lattice x")?,
                    parse_finite(fields[2], line_number, "lattice y")?,
                    parse_finite(fields[3], line_number, "lattice z")?,
                ]);
                last_was_atom = false;
            }
            "atom" | "atom_frac" => {
                if fields.len() != 5 {
                    return Err(parse_error(
                        line_number,
                        "atom records require x, y, z, and an element name",
                    ));
                }
                let name = fields[4].to_owned();
                if name.is_empty() {
                    return Err(parse_error(line_number, "atom name is empty"));
                }
                atoms.push(PendingAtom {
                    coordinates: [
                        parse_finite(fields[1], line_number, "x coordinate")?,
                        parse_finite(fields[2], line_number, "y coordinate")?,
                        parse_finite(fields[3], line_number, "z coordinate")?,
                    ],
                    fractional: fields[0].eq_ignore_ascii_case("atom_frac"),
                    name,
                    velocity: None,
                });
                last_was_atom = true;
            }
            "velocity" => {
                if !last_was_atom || atoms.last().is_none() {
                    return Err(parse_error(
                        line_number,
                        "velocity must follow an atom record",
                    ));
                }
                if fields.len() != 4 {
                    return Err(parse_error(
                        line_number,
                        "velocity requires exactly three components",
                    ));
                }
                let velocity = [
                    parse_finite(fields[1], line_number, "velocity x")?,
                    parse_finite(fields[2], line_number, "velocity y")?,
                    parse_finite(fields[3], line_number, "velocity z")?,
                ];
                let atom = atoms.last_mut().expect("checked above");
                if atom.velocity.is_some() {
                    return Err(parse_error(line_number, "duplicate velocity for atom"));
                }
                atom.velocity = Some(velocity);
                last_was_atom = false;
            }
            "initial_moment" => {
                last_was_atom = false;
            }
            _ => {
                return Err(parse_error(
                    line_number,
                    format!("unsupported record {:?}", fields[0]),
                ));
            }
        }
    }

    if atoms.is_empty() {
        return Err(FhiaimsError::InvalidStructure(
            "geometry contains no atoms".to_owned(),
        ));
    }
    let lattice_vectors = match lattice.as_slice() {
        [] => None,
        [first, second, third] => Some([*first, *second, *third]),
        _ => {
            return Err(FhiaimsError::InvalidStructure(
                "partial periodicity requires exactly three lattice vectors".to_owned(),
            ));
        }
    };
    let has_fractional = atoms.iter().any(|atom| atom.fractional);
    if has_fractional && lattice_vectors.is_none() {
        return Err(FhiaimsError::InvalidStructure(
            "fractional coordinates require lattice vectors".to_owned(),
        ));
    }
    if atoms.iter().any(|atom| atom.velocity.is_some())
        && atoms.iter().any(|atom| atom.velocity.is_none())
    {
        return Err(FhiaimsError::InvalidStructure(
            "either every atom or no atom may have a velocity".to_owned(),
        ));
    }

    let positions = atoms
        .iter()
        .map(|atom| {
            if atom.fractional {
                multiply_fractional(atom.coordinates, lattice_vectors.expect("checked above"))
            } else {
                atom.coordinates
            }
        })
        .collect::<Vec<_>>();
    let dimensions = lattice_vectors.map(triclinic_box);
    if let Some(dimensions) = dimensions
        && dimensions
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(FhiaimsError::InvalidStructure(
            "lattice vectors must define a non-degenerate cell".to_owned(),
        ));
    }
    let mut frame = CoordinateFrame::new(positions);
    frame.names = atoms.iter().map(|atom| atom.name.clone()).collect();
    frame.velocities = atoms
        .iter()
        .map(|atom| atom.velocity)
        .collect::<Option<Vec<_>>>();
    frame.dimensions = dimensions;
    let file = FhiaimsFile {
        coordinates: CoordinateFile::new(vec![frame]),
        lattice_vectors,
    };
    validate_fhiaims(&file)?;
    Ok(file)
}

fn write_document<W: Write>(file: &FhiaimsFile, mut writer: W) -> Result<(), FhiaimsError> {
    validate_fhiaims(file)?;
    let frame = &file.coordinates.frames[0];
    let lattice = file
        .lattice_vectors
        .or_else(|| frame.dimensions.map(triclinic_vectors));
    if let Some(vectors) = lattice {
        for vector in vectors {
            writeln!(
                writer,
                "lattice_vector {:.12e} {:.12e} {:.12e}",
                vector[0], vector[1], vector[2]
            )?;
        }
    }
    let has_velocities = frame.velocities.is_some();
    for (index, position) in frame.positions.iter().enumerate() {
        let name = frame
            .names
            .get(index)
            .filter(|name| !name.trim().is_empty())
            .map_or("X", String::as_str);
        writeln!(
            writer,
            "atom {:.12e} {:.12e} {:.12e} {}",
            position[0], position[1], position[2], name
        )?;
        if has_velocities {
            let velocity = frame
                .velocities
                .as_ref()
                .expect("checked by validate_fhiaims")[index];
            writeln!(
                writer,
                "velocity {:.12e} {:.12e} {:.12e}",
                velocity[0], velocity[1], velocity[2]
            )?;
        }
    }
    Ok(())
}

fn validate_fhiaims(file: &FhiaimsFile) -> Result<(), FhiaimsError> {
    if file.coordinates.frames.len() != 1 {
        return Err(FhiaimsError::InvalidStructure(
            "FHI-AIMS supports exactly one coordinate frame".to_owned(),
        ));
    }
    let frame = &file.coordinates.frames[0];
    if frame.positions.is_empty() {
        return Err(FhiaimsError::InvalidStructure(
            "geometry contains no atoms".to_owned(),
        ));
    }
    if !frame.metadata_is_consistent() {
        return Err(FhiaimsError::InvalidStructure(
            "per-atom metadata lengths do not match coordinates".to_owned(),
        ));
    }
    if frame
        .positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(FhiaimsError::InvalidStructure(
            "coordinates must be finite".to_owned(),
        ));
    }
    if let Some(velocities) = &frame.velocities
        && velocities.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(FhiaimsError::InvalidStructure(
            "velocities must be finite".to_owned(),
        ));
    }
    if let Some(vectors) = file.lattice_vectors {
        if vectors.iter().flatten().any(|value| !value.is_finite()) {
            return Err(FhiaimsError::InvalidStructure(
                "lattice vectors must be finite".to_owned(),
            ));
        }
        let dimensions = triclinic_box(vectors);
        if dimensions[..3].iter().any(|value| *value <= 0.0)
            || dimensions[3..]
                .iter()
                .any(|value| *value <= 0.0 || *value >= 180.0)
        {
            return Err(FhiaimsError::InvalidStructure(
                "lattice vectors must define a valid cell".to_owned(),
            ));
        }
    }
    Ok(())
}

fn multiply_fractional(fractional: [f64; 3], vectors: [[f64; 3]; 3]) -> [f64; 3] {
    [
        fractional[0] * vectors[0][0]
            + fractional[1] * vectors[1][0]
            + fractional[2] * vectors[2][0],
        fractional[0] * vectors[0][1]
            + fractional[1] * vectors[1][1]
            + fractional[2] * vectors[2][1],
        fractional[0] * vectors[0][2]
            + fractional[1] * vectors[1][2]
            + fractional[2] * vectors[2][2],
    ]
}

fn parse_finite(value: &str, line: usize, field: &str) -> Result<f64, FhiaimsError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| parse_error(line, format!("invalid {field}: {error}")))?;
    if !parsed.is_finite() {
        return Err(parse_error(line, format!("{field} must be finite")));
    }
    Ok(parsed)
}

fn parse_error(line: usize, message: impl Into<String>) -> FhiaimsError {
    FhiaimsError::Parse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOMETRY: &str = concat!(
        "# water cell\n",
        "lattice_vector 18.6 0.0 0.0\n",
        "lattice_vector 0.0 18.6 0.0\n",
        "lattice_vector 0.0 0.0 55.8\n",
        "atom 6.861735 2.103823 37.753513 O\n",
        "atom 7.119867 2.218342 36.808137 H\n",
        "atom 7.394193 1.415300 38.335713 H\n",
    );

    #[test]
    fn parses_cartesian_positions_cell_and_names() {
        let file = FhiaimsFile::from_str(GEOMETRY).unwrap();
        assert_eq!(file.n_atoms(), 3);
        assert_eq!(file.n_frames(), 1);
        assert_eq!(file.coordinates.frames[0].names, vec!["O", "H", "H"]);
        assert_eq!(
            file.coordinates.frames[0].positions[0],
            [6.861735, 2.103823, 37.753513]
        );
        assert_eq!(
            file.dimensions().unwrap(),
            [18.6, 18.6, 55.8, 90.0, 90.0, 90.0]
        );
    }

    #[test]
    fn fractional_and_cartesian_coordinates_share_the_same_frame() {
        let input = concat!(
            "lattice_vector 1 0 0\n",
            "lattice_vector 0 2 0\n",
            "lattice_vector 0 0 3\n",
            "atom 0.1 0.2 0.3 H\n",
            "atom_frac 0.2 0.2 0.2 H\n",
        );
        let file = FhiaimsFile::from_str(input).unwrap();
        assert_eq!(file.coordinates.frames[0].positions[0], [0.1, 0.2, 0.3]);
        let position = file.coordinates.frames[0].positions[1];
        assert!((position[0] - 0.2).abs() < 1e-12);
        assert!((position[1] - 0.4).abs() < 1e-12);
        assert!((position[2] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn velocities_must_follow_every_atom() {
        let valid = "atom 0 0 0 H\nvelocity 0.1 0.2 0.3\n";
        let file = FhiaimsFile::from_str(valid).unwrap();
        assert_eq!(
            file.coordinates.frames[0].velocities,
            Some(vec![[0.1, 0.2, 0.3]])
        );
        let missing = "atom 0 0 0 H\natom 1 0 0 H\nvelocity 0 0 0\n";
        assert!(matches!(
            FhiaimsFile::from_str(missing),
            Err(FhiaimsError::InvalidStructure(message)) if message.contains("every atom")
        ));
        let misplaced = "velocity 0 0 0\natom 0 0 0 H\n";
        assert!(matches!(
            FhiaimsFile::from_str(misplaced),
            Err(FhiaimsError::Parse { message, .. }) if message.contains("must follow")
        ));
    }

    #[test]
    fn writer_round_trips_positions_velocities_and_cell() {
        let input = concat!(
            "lattice_vector 2 0 0\n",
            "lattice_vector 0 2 0\n",
            "lattice_vector 0 0 2\n",
            "atom 0.1 0.2 0.3 O\n",
            "velocity 1 2 3\n",
        );
        let file = FhiaimsFile::from_str(input).unwrap();
        let reparsed = FhiaimsFile::from_str(&file.to_string().unwrap()).unwrap();
        assert_eq!(reparsed.coordinates.frames[0].names, vec!["O"]);
        assert_eq!(reparsed.coordinates.frames[0].positions[0], [0.1, 0.2, 0.3]);
        assert_eq!(
            reparsed.coordinates.frames[0].velocities,
            Some(vec![[1.0, 2.0, 3.0]])
        );
        assert_eq!(reparsed.lattice_vectors, file.lattice_vectors);
    }

    #[test]
    fn rejects_partial_cells_and_fractional_positions_without_a_cell() {
        let partial = "lattice_vector 1 0 0\nlattice_vector 0 1 0\natom 0 0 0 H\n";
        assert!(matches!(
            FhiaimsFile::from_str(partial),
            Err(FhiaimsError::InvalidStructure(message)) if message.contains("partial")
        ));
        let fractional = "atom_frac 0.1 0.1 0.1 H\n";
        assert!(matches!(
            FhiaimsFile::from_str(fractional),
            Err(FhiaimsError::InvalidStructure(message)) if message.contains("fractional")
        ));
    }

    #[test]
    fn universe_constructor_maps_velocity_and_masses() {
        let file = FhiaimsFile::from_str("atom 0 0 0 O\nvelocity 0.1 0.2 0.3\n").unwrap();
        let universe = crate::core::Universe::from_fhiaims_file(file).unwrap();
        assert_eq!(universe.topology.atoms[0].mass, 15.999);
        assert_eq!(
            universe.trajectory.frames[0].velocities,
            Some(vec![[0.1, 0.2, 0.3]])
        );
    }
}
