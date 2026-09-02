//! Amber restart (INPCRD/RESTRT) and NAMD binary coordinate readers.
//!
//! Amber restart files are text records with fixed-width coordinate values;
//! optional unit-cell and velocity records are retained when present.  NAMD's
//! `coor`/`namdbin` format contains a native-endian atom count followed by
//! double-precision Cartesian coordinates.  The public containers use the
//! same [`crate::coordinates::CoordinateFrame`] representation as the other
//! trajectory readers.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

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
}
