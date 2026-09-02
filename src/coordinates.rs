//! Coordinate-file containers and readers for the text based XYZ and GRO
//! formats.
//!
//! [`CoordinateFrame`] deliberately mirrors the coordinate portion of the
//! crate's topology [`crate::core::Frame`], while retaining the labels and
//! residue fields that are present in text coordinate formats.  Coordinates
//! are kept in the units used by the file: XYZ values are conventionally
//! Angstroms and GRO values are conventionally nanometres.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

/// A single coordinate frame.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinateFrame {
    /// Cartesian coordinates, one triplet per atom.
    pub positions: Vec<[f64; 3]>,
    /// Optional Cartesian velocities, one triplet per atom.
    pub velocities: Option<Vec<[f64; 3]>>,
    /// Unit-cell dimensions as `[a, b, c, alpha, beta, gamma]`.
    pub dimensions: Option<[f64; 6]>,
    /// Atom labels (the first column in XYZ and the atom name in GRO).
    pub names: Vec<String>,
    /// Residue names (written by GRO; empty for XYZ).
    pub residue_names: Vec<String>,
    /// Residue identifiers (written by GRO; zero for XYZ).
    pub residue_ids: Vec<i32>,
    /// Atom serial numbers (written by GRO; one-based defaults).
    pub atom_ids: Vec<usize>,
    /// The comment/title line associated with this frame.
    pub title: String,
}

impl CoordinateFrame {
    /// Construct a frame with labels and residue metadata filled with useful
    /// defaults.
    #[must_use]
    pub fn new(positions: Vec<[f64; 3]>) -> Self {
        let count = positions.len();
        Self {
            positions,
            velocities: None,
            dimensions: None,
            names: vec!["X".to_owned(); count],
            residue_names: vec!["UNK".to_owned(); count],
            residue_ids: vec![0; count],
            atom_ids: (1..=count).collect(),
            title: String::new(),
        }
    }

    /// Number of atoms in this frame.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    /// Return whether all per-atom metadata arrays have the frame's length.
    #[must_use]
    pub fn metadata_is_consistent(&self) -> bool {
        let count = self.positions.len();
        self.names.is_empty_or_len(count)
            && self.residue_names.is_empty_or_len(count)
            && self.residue_ids.is_empty_or_len(count)
            && self.atom_ids.is_empty_or_len(count)
            && self
                .velocities
                .as_ref()
                .is_none_or(|velocities| velocities.len() == count)
    }
}

trait OptionalVecLength {
    fn is_empty_or_len(&self, length: usize) -> bool;
}

impl<T> OptionalVecLength for Vec<T> {
    fn is_empty_or_len(&self, length: usize) -> bool {
        self.is_empty() || self.len() == length
    }
}

/// A coordinate file containing one or more frames.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoordinateFile {
    pub frames: Vec<CoordinateFrame>,
}

impl CoordinateFile {
    /// Construct a coordinate file from its frames.
    #[must_use]
    pub fn new(frames: Vec<CoordinateFrame>) -> Self {
        Self { frames }
    }

    /// Number of frames in this file.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Number of atoms, or zero for an empty file.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.frames.first().map_or(0, CoordinateFrame::n_atoms)
    }

    /// Return one frame by zero-based index.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.frames.get(index)
    }

    /// Parse an XYZ document from a reader.
    pub fn read_xyz<R: Read>(reader: R) -> Result<Self, CoordinateError> {
        let mut input = String::new();
        BufReader::new(reader).read_to_string(&mut input)?;
        Self::from_xyz_str(&input)
    }

    /// Parse an XYZ document from a string.
    pub fn from_xyz_str(input: &str) -> Result<Self, CoordinateError> {
        parse_xyz(input)
    }

    /// Write this file as an XYZ document.
    pub fn write_xyz<W: Write>(&self, writer: W) -> Result<(), CoordinateError> {
        write_xyz_document(self, writer)
    }

    /// Serialize this file as an XYZ document.
    pub fn to_xyz_string(&self) -> Result<String, CoordinateError> {
        let mut output = Vec::new();
        self.write_xyz(&mut output)?;
        Ok(String::from_utf8_lossy(&output).into_owned())
    }

    /// Parse a GRO document from a reader.
    pub fn read_gro<R: Read>(reader: R) -> Result<Self, CoordinateError> {
        let mut input = String::new();
        BufReader::new(reader).read_to_string(&mut input)?;
        Self::from_gro_str(&input)
    }

    /// Parse a GRO document from a string.
    pub fn from_gro_str(input: &str) -> Result<Self, CoordinateError> {
        parse_gro(input)
    }

    /// Write this file as a GRO document.
    pub fn write_gro<W: Write>(&self, writer: W) -> Result<(), CoordinateError> {
        write_gro_document(self, writer)
    }

    /// Serialize this file as a GRO document.
    pub fn to_gro_string(&self) -> Result<String, CoordinateError> {
        let mut output = Vec::new();
        self.write_gro(&mut output)?;
        Ok(String::from_utf8_lossy(&output).into_owned())
    }
}

/// Errors produced while reading or writing coordinate files.
#[derive(Debug)]
pub enum CoordinateError {
    Io(io::Error),
    Parse {
        format: &'static str,
        line: usize,
        message: String,
    },
    InconsistentFrame {
        frame: usize,
        expected: usize,
        found: usize,
    },
    InvalidStructure(String),
}

impl fmt::Display for CoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse {
                format,
                line,
                message,
            } => write!(formatter, "{format} parse error on line {line}: {message}"),
            Self::InconsistentFrame {
                frame,
                expected,
                found,
            } => write!(
                formatter,
                "coordinate frame {frame} contains {found} atoms; expected {expected}"
            ),
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid coordinate structure: {message}")
            }
        }
    }
}

impl std::error::Error for CoordinateError {}

impl From<io::Error> for CoordinateError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Read an XYZ coordinate file from a path.
pub fn read_xyz<P: AsRef<Path>>(path: P) -> Result<CoordinateFile, CoordinateError> {
    CoordinateFile::read_xyz(File::open(path)?)
}

/// Write an XYZ coordinate file to a path.
pub fn write_xyz<P: AsRef<Path>>(
    path: P,
    coordinate_file: &CoordinateFile,
) -> Result<(), CoordinateError> {
    coordinate_file.write_xyz(File::create(path)?)
}

/// Read a GRO coordinate file from a path.
pub fn read_gro<P: AsRef<Path>>(path: P) -> Result<CoordinateFile, CoordinateError> {
    CoordinateFile::read_gro(File::open(path)?)
}

/// Write a GRO coordinate file to a path.
pub fn write_gro<P: AsRef<Path>>(
    path: P,
    coordinate_file: &CoordinateFile,
) -> Result<(), CoordinateError> {
    coordinate_file.write_gro(File::create(path)?)
}

fn parse_xyz(input: &str) -> Result<CoordinateFile, CoordinateError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut index = 0;
    let mut frames = Vec::new();
    let mut expected_atoms = None;

    while index < lines.len() {
        // Blank lines between models are harmless, but the second line of a
        // model is always consumed as its title (and may itself be blank).
        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }
        if index == lines.len() {
            break;
        }
        let line_number = index + 1;
        let atom_count = lines[index].trim().parse::<usize>().map_err(|error| {
            parse_error("XYZ", line_number, format!("invalid atom count: {error}"))
        })?;
        index += 1;
        if index >= lines.len() {
            return Err(parse_error("XYZ", line_number + 1, "missing title line"));
        }
        let title = lines[index].to_owned();
        index += 1;
        let mut frame = CoordinateFrame::new(Vec::with_capacity(atom_count));
        frame.title = title;
        frame.names.clear();
        frame.residue_names.clear();
        frame.residue_ids.clear();
        frame.atom_ids.clear();
        for atom_index in 0..atom_count {
            if index >= lines.len() {
                return Err(parse_error(
                    "XYZ",
                    lines.len() + 1,
                    format!("frame declares {atom_count} atoms but ends after {atom_index}"),
                ));
            }
            let line_number = index + 1;
            let fields: Vec<&str> = lines[index].split_whitespace().collect();
            index += 1;
            if fields.len() < 4 {
                return Err(parse_error(
                    "XYZ",
                    line_number,
                    "atom line requires a label and three coordinates",
                ));
            }
            let position = [
                parse_float("XYZ", line_number, fields[1], "x coordinate")?,
                parse_float("XYZ", line_number, fields[2], "y coordinate")?,
                parse_float("XYZ", line_number, fields[3], "z coordinate")?,
            ];
            frame.positions.push(position);
            frame.names.push(fields[0].to_owned());
            frame.atom_ids.push(atom_index + 1);
        }
        if let Some(expected) = expected_atoms {
            if atom_count != expected {
                return Err(CoordinateError::InconsistentFrame {
                    frame: frames.len() + 1,
                    expected,
                    found: atom_count,
                });
            }
        } else {
            expected_atoms = Some(atom_count);
        }
        frames.push(frame);
    }
    Ok(CoordinateFile::new(frames))
}

fn write_xyz_document<W: Write>(
    file: &CoordinateFile,
    mut writer: W,
) -> Result<(), CoordinateError> {
    for (frame_index, frame) in file.frames.iter().enumerate() {
        validate_frame(frame, frame_index)?;
        writeln!(writer, "{}", frame.positions.len())?;
        writeln!(writer, "{}", frame.title)?;
        for (atom_index, position) in frame.positions.iter().enumerate() {
            let name = frame
                .names
                .get(atom_index)
                .map_or("X", String::as_str)
                .trim();
            let name = if name.is_empty() {
                "X"
            } else {
                name.split_whitespace().next().unwrap_or("X")
            };
            writeln!(
                writer,
                "{name} {:.8} {:.8} {:.8}",
                position[0], position[1], position[2]
            )?;
        }
    }
    Ok(())
}

fn parse_gro(input: &str) -> Result<CoordinateFile, CoordinateError> {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() < 2 {
        return Err(parse_error(
            "GRO",
            lines.len() + 1,
            "missing title or atom-count line",
        ));
    }
    let title = lines[0].to_owned();
    let atom_count = lines[1]
        .trim()
        .parse::<usize>()
        .map_err(|error| parse_error("GRO", 2, format!("invalid atom count: {error}")))?;
    let expected_lines = atom_count + 3;
    if lines.len() < expected_lines {
        return Err(parse_error(
            "GRO",
            lines.len() + 1,
            format!("frame declares {atom_count} atoms but file is incomplete"),
        ));
    }
    let mut frame = CoordinateFrame::new(Vec::with_capacity(atom_count));
    frame.title = title;
    frame.names.clear();
    frame.residue_names.clear();
    frame.residue_ids.clear();
    frame.atom_ids.clear();
    let mut velocities = Vec::with_capacity(atom_count);
    let mut any_velocity = false;
    for atom_index in 0..atom_count {
        let line_number = atom_index + 3;
        let line = lines[atom_index + 2];
        let residue_id = parse_field::<i32>(line, 0, 5, "GRO", line_number, "residue id")?;
        let residue_name = fixed_field(line, 5, 10).trim().to_owned();
        let atom_name = fixed_field(line, 10, 15).trim().to_owned();
        if residue_name.is_empty() {
            return Err(parse_error("GRO", line_number, "residue name is empty"));
        }
        if atom_name.is_empty() {
            return Err(parse_error("GRO", line_number, "atom name is empty"));
        }
        let atom_id = parse_field::<usize>(line, 15, 20, "GRO", line_number, "atom id")?;
        let position = [
            parse_field::<f64>(line, 20, 28, "GRO", line_number, "x coordinate")?,
            parse_field::<f64>(line, 28, 36, "GRO", line_number, "y coordinate")?,
            parse_field::<f64>(line, 36, 44, "GRO", line_number, "z coordinate")?,
        ];
        let velocity = parse_optional_velocity(line, line_number)?;
        if velocity.is_some() {
            any_velocity = true;
        }
        velocities.push(velocity.unwrap_or([0.0; 3]));
        frame.positions.push(position);
        frame.names.push(atom_name);
        frame.residue_names.push(residue_name);
        frame.residue_ids.push(residue_id);
        frame.atom_ids.push(atom_id);
    }
    let box_line_number = atom_count + 3;
    let box_fields = parse_box_fields(lines[atom_count + 2], box_line_number)?;
    frame.dimensions = Some(box_to_dimensions(&box_fields, box_line_number)?);
    frame.velocities = any_velocity.then_some(velocities);
    Ok(CoordinateFile::new(vec![frame]))
}

fn write_gro_document<W: Write>(
    file: &CoordinateFile,
    mut writer: W,
) -> Result<(), CoordinateError> {
    if file.frames.len() != 1 {
        return Err(CoordinateError::InvalidStructure(
            "GRO supports exactly one coordinate frame".to_owned(),
        ));
    }
    let frame = &file.frames[0];
    validate_frame(frame, 0)?;
    let atom_count = frame.positions.len();
    if atom_count > 99_999 {
        return Err(CoordinateError::InvalidStructure(
            "GRO atom count cannot exceed 99999".to_owned(),
        ));
    }
    writeln!(writer, "{}", frame.title)?;
    writeln!(writer, "{atom_count}")?;
    for (index, position) in frame.positions.iter().enumerate() {
        if position.iter().any(|value| !value.is_finite()) {
            return Err(CoordinateError::InvalidStructure(format!(
                "GRO atom {} has a non-finite coordinate",
                index + 1
            )));
        }
        let residue_id = frame
            .residue_ids
            .get(index)
            .copied()
            .unwrap_or((index + 1) as i32);
        let atom_id = frame.atom_ids.get(index).copied().unwrap_or(index + 1);
        if !(0..=99_999).contains(&atom_id) {
            return Err(CoordinateError::InvalidStructure(format!(
                "GRO atom id {atom_id} does not fit the five-column field"
            )));
        }
        let residue_name = fit_field(
            frame.residue_names.get(index).map_or("UNK", String::as_str),
            5,
        );
        let atom_name = fit_field(frame.names.get(index).map_or("X", String::as_str), 5);
        if !(-99_999..=99_999).contains(&residue_id) {
            return Err(CoordinateError::InvalidStructure(format!(
                "GRO residue id {residue_id} does not fit the five-column field"
            )));
        }
        writeln!(
            writer,
            "{residue_id:>5}{residue_name:<5}{atom_name:>5}{atom_id:>5}{:8.3}{:8.3}{:8.3}{}",
            position[0],
            position[1],
            position[2],
            format_velocity(
                frame
                    .velocities
                    .as_ref()
                    .and_then(|values| values.get(index))
            )
        )?;
    }
    let dimensions = frame
        .dimensions
        .unwrap_or([0.0, 0.0, 0.0, 90.0, 90.0, 90.0]);
    if dimensions[..3]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(CoordinateError::InvalidStructure(
            "GRO box contains non-finite lengths".to_owned(),
        ));
    }
    if is_orthorhombic(dimensions) {
        writeln!(
            writer,
            "{:10.5}{:10.5}{:10.5}",
            dimensions[0], dimensions[1], dimensions[2]
        )?;
    } else {
        let vectors = dimensions_to_vectors(dimensions)?;
        // GROMACS order: v1x v2y v3z v1y v1z v2x v2z v3x v3y.
        writeln!(
            writer,
            "{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}",
            vectors[0][0],
            vectors[1][1],
            vectors[2][2],
            vectors[0][1],
            vectors[0][2],
            vectors[1][0],
            vectors[1][2],
            vectors[2][0],
            vectors[2][1]
        )?;
    }
    Ok(())
}

fn validate_frame(frame: &CoordinateFrame, frame_index: usize) -> Result<(), CoordinateError> {
    let count = frame.positions.len();
    if frame
        .velocities
        .as_ref()
        .is_some_and(|values| values.len() != count)
    {
        return Err(CoordinateError::InconsistentFrame {
            frame: frame_index + 1,
            expected: count,
            found: frame.velocities.as_ref().map_or(0, Vec::len),
        });
    }
    for metadata_len in [
        frame.names.len(),
        frame.residue_names.len(),
        frame.residue_ids.len(),
        frame.atom_ids.len(),
    ] {
        if metadata_len != 0 && metadata_len != count {
            return Err(CoordinateError::InconsistentFrame {
                frame: frame_index + 1,
                expected: count,
                found: metadata_len,
            });
        }
    }
    Ok(())
}

fn parse_optional_velocity(
    line: &str,
    line_number: usize,
) -> Result<Option<[f64; 3]>, CoordinateError> {
    if line.len() <= 44 {
        return Ok(None);
    }
    let suffix = fixed_field(line, 44, 68);
    if suffix.trim().is_empty() {
        return Ok(None);
    }
    let fields: Vec<&str> = suffix.split_whitespace().collect();
    if fields.len() < 3 {
        return Err(parse_error(
            "GRO",
            line_number,
            "velocity requires three components",
        ));
    }
    Ok(Some([
        parse_float("GRO", line_number, fields[0], "x velocity")?,
        parse_float("GRO", line_number, fields[1], "y velocity")?,
        parse_float("GRO", line_number, fields[2], "z velocity")?,
    ]))
}

fn parse_box_fields(line: &str, line_number: usize) -> Result<Vec<f64>, CoordinateError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() == 3 || fields.len() == 9 {
        return fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                parse_float(
                    "GRO",
                    line_number,
                    field,
                    &format!("box field {}", index + 1),
                )
            })
            .collect();
    }
    // Some writers omit whitespace between fixed-width 8.5 fields.
    if fields.len() == 1 && line.trim().len() >= 8 * 9 {
        let mut values = Vec::with_capacity(9);
        for index in 0..9 {
            values.push(parse_float(
                "GRO",
                line_number,
                &line[index * 8..(index + 1) * 8],
                &format!("box field {}", index + 1),
            )?);
        }
        return Ok(values);
    }
    Err(parse_error(
        "GRO",
        line_number,
        format!("box must contain 3 or 9 values, found {}", fields.len()),
    ))
}

fn box_to_dimensions(values: &[f64], line_number: usize) -> Result<[f64; 6], CoordinateError> {
    match values {
        [a, b, c] => Ok([*a, *b, *c, 90.0, 90.0, 90.0]),
        [v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y] => {
            if values.iter().all(|value| value.abs() <= f64::EPSILON) {
                return Ok([0.0, 0.0, 0.0, 90.0, 90.0, 90.0]);
            }
            let vectors = [[*v1x, *v1y, *v1z], [*v2x, *v2y, *v2z], [*v3x, *v3y, *v3z]];
            let lengths = vectors.map(vector_length);
            if lengths.iter().any(|length| *length <= f64::EPSILON) {
                return Err(parse_error(
                    "GRO",
                    line_number,
                    "box vectors must have non-zero length",
                ));
            }
            Ok([
                lengths[0],
                lengths[1],
                lengths[2],
                angle_between(vectors[1], vectors[2]),
                angle_between(vectors[0], vectors[2]),
                angle_between(vectors[0], vectors[1]),
            ])
        }
        _ => Err(parse_error(
            "GRO",
            line_number,
            "box must contain 3 or 9 values",
        )),
    }
}

fn dimensions_to_vectors(dimensions: [f64; 6]) -> Result<[[f64; 3]; 3], CoordinateError> {
    let [a, b, c, alpha, beta, gamma] = dimensions;
    if [a, b, c, alpha, beta, gamma]
        .iter()
        .any(|value| !value.is_finite())
        || a <= 0.0
        || b <= 0.0
        || c <= 0.0
        || !(0.0 < alpha
            && alpha < 180.0
            && 0.0 < beta
            && beta < 180.0
            && 0.0 < gamma
            && gamma < 180.0)
    {
        return Err(CoordinateError::InvalidStructure(
            "GRO box lengths and angles are invalid".to_owned(),
        ));
    }
    let alpha = alpha.to_radians();
    let beta = beta.to_radians();
    let gamma = gamma.to_radians();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();
    if sin_gamma.abs() <= f64::EPSILON {
        return Err(CoordinateError::InvalidStructure(
            "GRO box angle gamma is degenerate".to_owned(),
        ));
    }
    let v1 = [a, 0.0, 0.0];
    let v2 = [b * cos_gamma, b * sin_gamma, 0.0];
    let v3x = c * beta.cos();
    let v3y = c * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma;
    let v3z_squared = c * c - v3x * v3x - v3y * v3y;
    if v3z_squared < -1.0e-10 {
        return Err(CoordinateError::InvalidStructure(
            "GRO box angles are inconsistent".to_owned(),
        ));
    }
    Ok([v1, v2, [v3x, v3y, v3z_squared.max(0.0).sqrt()]])
}

fn is_orthorhombic(dimensions: [f64; 6]) -> bool {
    dimensions[3..]
        .iter()
        .all(|angle| (*angle - 90.0).abs() <= 1.0e-8)
}

fn vector_length(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn angle_between(first: [f64; 3], second: [f64; 3]) -> f64 {
    let denominator = vector_length(first) * vector_length(second);
    let cosine = (first[0] * second[0] + first[1] * second[1] + first[2] * second[2]) / denominator;
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}

fn format_velocity(velocity: Option<&[f64; 3]>) -> String {
    velocity.map_or_else(String::new, |velocity| {
        format!("{:8.4}{:8.4}{:8.4}", velocity[0], velocity[1], velocity[2])
    })
}

fn fit_field(value: &str, width: usize) -> String {
    value.trim().chars().take(width).collect()
}

fn fixed_field(line: &str, start: usize, end: usize) -> &str {
    let bytes = line.as_bytes();
    if start >= bytes.len() {
        return "";
    }
    std::str::from_utf8(&bytes[start..end.min(bytes.len())]).unwrap_or("")
}

fn parse_field<T: std::str::FromStr>(
    line: &str,
    start: usize,
    end: usize,
    format: &'static str,
    line_number: usize,
    field_name: &str,
) -> Result<T, CoordinateError>
where
    T::Err: fmt::Display,
{
    let value = fixed_field(line, start, end).trim();
    if value.is_empty() {
        return Err(parse_error(
            format,
            line_number,
            format!("missing {field_name}"),
        ));
    }
    value.parse::<T>().map_err(|error| {
        parse_error(
            format,
            line_number,
            format!("invalid {field_name} {value:?}: {error}"),
        )
    })
}

fn parse_float(
    format: &'static str,
    line: usize,
    value: &str,
    field_name: &str,
) -> Result<f64, CoordinateError> {
    value.parse::<f64>().map_err(|error| {
        parse_error(
            format,
            line,
            format!("invalid {field_name} {value:?}: {error}"),
        )
    })
}

fn parse_error(format: &'static str, line: usize, message: impl Into<String>) -> CoordinateError {
    CoordinateError::Parse {
        format,
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyz_reads_multiple_frames_and_ignores_extra_columns() {
        let input = "2\nfirst\nC 0 1 2 charge=0\nO 3 4 5\n2\nsecond\nC 1 2 3\nO 4 5 6\n";
        let file = CoordinateFile::from_xyz_str(input).unwrap();
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.frames[0].names, vec!["C", "O"]);
        assert_eq!(file.frames[1].positions[1], [4.0, 5.0, 6.0]);
        assert_eq!(file.frames[0].title, "first");
    }

    #[test]
    fn xyz_round_trip_preserves_labels_and_coordinates() {
        let mut frame = CoordinateFrame::new(vec![[1.23456789, -2.0, 3.0]]);
        frame.names = vec!["He".into()];
        frame.title = "example".into();
        let file = CoordinateFile::new(vec![frame]);
        let parsed = CoordinateFile::from_xyz_str(&file.to_xyz_string().unwrap()).unwrap();
        assert_eq!(parsed.frames[0].names, vec!["He"]);
        assert_eq!(parsed.frames[0].positions[0][0], 1.23456789);
    }

    #[test]
    fn gro_reads_fixed_fields_velocities_and_triclinic_box() {
        let input = concat!(
            "water\n",
            "2\n",
            "    1SOL     OW    1   0.100   0.200   0.300  0.0100  0.0200  0.0300\n",
            "    1SOL     HW    2   0.400   0.500   0.600\n",
            "  1.00000  1.00000  1.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000\n",
        );
        let file = CoordinateFile::from_gro_str(input).unwrap();
        let frame = &file.frames[0];
        assert_eq!(frame.names, vec!["OW", "HW"]);
        assert_eq!(frame.positions[0], [0.1, 0.2, 0.3]);
        assert_eq!(frame.velocities.as_ref().unwrap()[0], [0.01, 0.02, 0.03]);
        assert_eq!(frame.velocities.as_ref().unwrap()[1], [0.0, 0.0, 0.0]);
        assert_eq!(frame.dimensions.unwrap()[..3], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn gro_round_trip_writes_fixed_fields_and_box() {
        let mut frame = CoordinateFrame::new(vec![[0.1, 0.2, 0.3]]);
        frame.title = "water".into();
        frame.residue_names = vec!["SOL".into()];
        frame.residue_ids = vec![1];
        frame.names = vec!["OW".into()];
        frame.dimensions = Some([2.0, 3.0, 4.0, 90.0, 90.0, 90.0]);
        let parsed = CoordinateFile::from_gro_str(
            &CoordinateFile::new(vec![frame]).to_gro_string().unwrap(),
        )
        .unwrap();
        assert_eq!(parsed.frames[0].positions[0], [0.1, 0.2, 0.3]);
        assert_eq!(
            parsed.frames[0].dimensions.unwrap(),
            [2.0, 3.0, 4.0, 90.0, 90.0, 90.0]
        );
    }

    #[test]
    fn malformed_xyz_reports_format_and_line() {
        let error = CoordinateFile::from_xyz_str("1\ncomment\nC 0 nope 1\n").unwrap_err();
        assert!(error.to_string().contains("XYZ parse error on line 3"));
    }
}
