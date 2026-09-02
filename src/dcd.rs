//! Binary DCD trajectory reader and writer.
//!
//! DCD is a Fortran-record based format used by CHARMM, NAMD, VMD and
//! several other molecular-dynamics programs. The format has a few dialects;
//! this module accepts the common 32-bit little- and big-endian variants and
//! preserves the coordinate and unit-cell data shared by those dialects.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Byte order used by a DCD document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DcdEndian {
    Little,
    Big,
}

impl DcdEndian {
    fn read_i32(self, bytes: &[u8]) -> i32 {
        let mut value = [0_u8; 4];
        value.copy_from_slice(bytes);
        match self {
            Self::Little => i32::from_le_bytes(value),
            Self::Big => i32::from_be_bytes(value),
        }
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        let mut value = [0_u8; 4];
        value.copy_from_slice(bytes);
        match self {
            Self::Little => f32::from_le_bytes(value),
            Self::Big => f32::from_be_bytes(value),
        }
    }

    fn read_f64(self, bytes: &[u8]) -> f64 {
        let mut value = [0_u8; 8];
        value.copy_from_slice(bytes);
        match self {
            Self::Little => f64::from_le_bytes(value),
            Self::Big => f64::from_be_bytes(value),
        }
    }

    fn write_i32(self, value: i32, output: &mut Vec<u8>) {
        let bytes = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        output.extend_from_slice(&bytes);
    }

    fn write_f32(self, value: f32, output: &mut Vec<u8>) {
        let bytes = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        output.extend_from_slice(&bytes);
    }
}

/// Header and topology-independent metadata in a DCD file.
#[derive(Clone, Debug, PartialEq)]
pub struct DcdHeader {
    pub n_frames: usize,
    pub istart: i32,
    pub nsavc: i32,
    pub n_fixed: i32,
    pub delta: f64,
    pub has_unitcell: bool,
    pub title: String,
    pub n_atoms: usize,
    pub endian: DcdEndian,
}

impl Default for DcdHeader {
    fn default() -> Self {
        Self {
            n_frames: 0,
            istart: 0,
            nsavc: 1,
            n_fixed: 0,
            delta: 1.0,
            has_unitcell: false,
            title: "Created by mdanalysis_rs".to_owned(),
            n_atoms: 0,
            endian: DcdEndian::Little,
        }
    }
}

/// A parsed DCD document.
#[derive(Clone, Debug, PartialEq)]
pub struct DcdFile {
    pub header: DcdHeader,
    pub coordinates: CoordinateFile,
}

/// Options controlling DCD serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct DcdWriteOptions {
    pub istart: i32,
    pub nsavc: i32,
    pub delta: f64,
    pub title: String,
    pub endian: DcdEndian,
}

impl Default for DcdWriteOptions {
    fn default() -> Self {
        Self {
            istart: 0,
            nsavc: 1,
            delta: 1.0,
            title: "Created by mdanalysis_rs".to_owned(),
            endian: DcdEndian::Little,
        }
    }
}

/// Errors produced while reading or writing DCD files.
#[derive(Debug)]
pub enum DcdError {
    Io(io::Error),
    Parse { offset: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for DcdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { offset, message } => {
                write!(formatter, "DCD parse error at byte {offset}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid DCD structure: {message}")
            }
        }
    }
}

impl std::error::Error for DcdError {}

impl From<io::Error> for DcdError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CoordinateError> for DcdError {
    fn from(error: CoordinateError) -> Self {
        Self::InvalidStructure(error.to_string())
    }
}

impl DcdFile {
    /// Parse a DCD document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, DcdError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a DCD document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DcdError> {
        let endian = detect_endian(bytes)?;
        let mut cursor = 0;
        let header_record = read_record(bytes, &mut cursor, endian)?;
        if header_record.len() < 84 || &header_record[..4] != b"CORD" {
            return Err(parse_error(
                cursor,
                "header record is not a CORD DCD header",
            ));
        }
        let _declared_frames =
            positive_count(endian.read_i32(&header_record[4..8]), "frame count")?;
        let istart = endian.read_i32(&header_record[8..12]);
        let nsavc = endian.read_i32(&header_record[12..16]);
        let n_fixed = endian.read_i32(&header_record[36..40]);
        let is_charmm = header_record.len() >= 84 && endian.read_i32(&header_record[80..84]) != 0;
        let delta = if is_charmm {
            f64::from(endian.read_f32(&header_record[40..44]))
        } else {
            endian.read_f64(&header_record[40..48])
        };
        let has_unitcell =
            is_charmm && header_record.len() >= 48 && endian.read_i32(&header_record[44..48]) != 0;
        let has_four_dimensions =
            is_charmm && header_record.len() >= 52 && endian.read_i32(&header_record[48..52]) == 1;
        if n_fixed != 0 {
            return Err(DcdError::InvalidStructure(
                "DCD trajectories with fixed atoms are not supported".to_owned(),
            ));
        }

        let title_record = read_title_record(bytes, &mut cursor, endian)?;
        if title_record.len() < 4 {
            return Err(parse_error(cursor, "title record is truncated"));
        }
        let title_count = positive_count(endian.read_i32(&title_record[..4]), "title count")?;
        let required_title_bytes = 4_usize
            .checked_add(title_count.saturating_mul(80))
            .ok_or_else(|| parse_error(cursor, "title count overflows file size"))?;
        if title_record.len() < required_title_bytes {
            return Err(parse_error(
                cursor,
                "title record is shorter than its title count",
            ));
        }
        let mut title = String::new();
        for line in title_record[4..required_title_bytes].chunks(80) {
            let line = String::from_utf8_lossy(line).trim_end().to_owned();
            if !line.is_empty() {
                if !title.is_empty() {
                    title.push('\n');
                }
                title.push_str(&line);
            }
        }

        let atom_record = read_record(bytes, &mut cursor, endian)?;
        if atom_record.len() != 4 {
            return Err(parse_error(
                cursor,
                "atom-count record must contain one integer",
            ));
        }
        let n_atoms = positive_count(endian.read_i32(atom_record), "atom count")?;
        let mut frames = Vec::new();
        while cursor < bytes.len() {
            let mut dimensions = None;
            let first_len = peek_record_len(bytes, cursor, endian)?;
            let cell_flag = has_unitcell
                || (first_len == 48 && looks_like_cell_prefix(bytes, cursor, endian, n_atoms));
            if cell_flag {
                let cell_record = read_record(bytes, &mut cursor, endian)?;
                if cell_record.len() != 48 {
                    return Err(parse_error(
                        cursor,
                        "unit-cell record must contain six doubles",
                    ));
                }
                let mut values = [0.0_f64; 6];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = endian.read_f64(&cell_record[index * 8..index * 8 + 8]);
                }
                dimensions = decode_cell(values);
            }
            let x = read_coordinate_record(bytes, &mut cursor, endian, n_atoms, "x")?;
            let y = read_coordinate_record(bytes, &mut cursor, endian, n_atoms, "y")?;
            let z = read_coordinate_record(bytes, &mut cursor, endian, n_atoms, "z")?;
            if has_four_dimensions {
                let _ = read_record(bytes, &mut cursor, endian)?;
            }
            let positions = x
                .into_iter()
                .zip(y)
                .zip(z)
                .map(|((x, y), z)| [x, y, z])
                .collect();
            let mut frame = CoordinateFrame::new(positions);
            frame.title.clone_from(&title);
            frame.dimensions = dimensions;
            frames.push(frame);
        }
        // A handful of legacy files have an over-large NSET header (for
        // example, an interrupted run that still contains complete frames).
        // The record stream is authoritative, so retain all complete frames
        // read up to EOF.
        if cursor != bytes.len() {
            return Err(parse_error(cursor, "trailing bytes after final frame"));
        }
        let header = DcdHeader {
            n_frames: frames.len(),
            istart,
            nsavc,
            n_fixed,
            delta,
            has_unitcell: frames.iter().any(|frame| frame.dimensions.is_some()),
            title,
            n_atoms,
            endian,
        };
        Ok(Self {
            header,
            coordinates: CoordinateFile::new(frames),
        })
    }

    /// Write a DCD document using the header values held by this object.
    pub fn write<W: Write>(&self, writer: W) -> Result<(), DcdError> {
        self.write_with_options(
            writer,
            DcdWriteOptions {
                istart: self.header.istart,
                nsavc: self.header.nsavc,
                delta: self.header.delta,
                title: self.header.title.clone(),
                endian: self.header.endian,
            },
        )
    }

    /// Write a DCD document with explicit header values.
    pub fn write_with_options<W: Write>(
        &self,
        mut writer: W,
        options: DcdWriteOptions,
    ) -> Result<(), DcdError> {
        write_coordinate_file(&self.coordinates, &mut writer, options)
    }

    /// Serialize this document to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DcdError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes)?;
        Ok(bytes)
    }
}

impl CoordinateFile {
    /// Parse a DCD document from a reader.
    pub fn read_dcd<R: Read>(reader: R) -> Result<Self, DcdError> {
        Ok(DcdFile::read(reader)?.coordinates)
    }

    /// Parse a DCD document held in memory.
    pub fn from_dcd_bytes(bytes: &[u8]) -> Result<Self, DcdError> {
        Ok(DcdFile::from_bytes(bytes)?.coordinates)
    }

    /// Write this coordinate file as a DCD document.
    pub fn write_dcd<W: Write>(&self, writer: W) -> Result<(), DcdError> {
        let dcd = DcdFile {
            header: DcdHeader {
                n_frames: self.frames.len(),
                n_atoms: self.n_atoms(),
                has_unitcell: self.frames.iter().any(|frame| frame.dimensions.is_some()),
                ..DcdHeader::default()
            },
            coordinates: self.clone(),
        };
        dcd.write(writer)
    }

    /// Serialize this coordinate file as DCD bytes.
    pub fn to_dcd_bytes(&self) -> Result<Vec<u8>, DcdError> {
        let mut bytes = Vec::new();
        self.write_dcd(&mut bytes)?;
        Ok(bytes)
    }
}

/// Read a DCD trajectory from a filesystem path.
pub fn read_dcd<P: AsRef<Path>>(path: P) -> Result<DcdFile, DcdError> {
    DcdFile::read(File::open(path)?)
}

/// Write a coordinate file to a DCD filesystem path.
pub fn write_dcd<P: AsRef<Path>>(path: P, coordinates: &CoordinateFile) -> Result<(), DcdError> {
    coordinates.write_dcd(File::create(path)?)
}

fn write_coordinate_file<W: Write>(
    coordinates: &CoordinateFile,
    writer: &mut W,
    options: DcdWriteOptions,
) -> Result<(), DcdError> {
    if options.nsavc <= 0 {
        return Err(DcdError::InvalidStructure(
            "nsavc must be positive".to_owned(),
        ));
    }
    if !options.delta.is_finite() {
        return Err(DcdError::InvalidStructure(
            "delta must be finite".to_owned(),
        ));
    }
    let n_atoms = coordinates.n_atoms();
    if coordinates
        .frames
        .iter()
        .any(|frame| frame.n_atoms() != n_atoms)
    {
        return Err(DcdError::InvalidStructure(
            "all frames must contain the same atom count".to_owned(),
        ));
    }
    let has_cell = coordinates
        .frames
        .iter()
        .any(|frame| frame.dimensions.is_some());
    if has_cell
        && coordinates
            .frames
            .iter()
            .any(|frame| frame.dimensions.is_none())
    {
        return Err(DcdError::InvalidStructure(
            "unit-cell dimensions must be present on every frame".to_owned(),
        ));
    }
    let endian = options.endian;
    let mut header = Vec::with_capacity(84);
    header.extend_from_slice(b"CORD");
    endian.write_i32(
        i32::try_from(coordinates.frames.len())
            .map_err(|_| DcdError::InvalidStructure("too many frames".to_owned()))?,
        &mut header,
    );
    endian.write_i32(options.istart, &mut header);
    endian.write_i32(options.nsavc, &mut header);
    for _ in 0..5 {
        endian.write_i32(0, &mut header);
    }
    endian.write_i32(0, &mut header);
    endian.write_f32(options.delta as f32, &mut header);
    endian.write_i32(if has_cell { 1 } else { 0 }, &mut header);
    endian.write_i32(0, &mut header);
    for _ in 0..7 {
        endian.write_i32(0, &mut header);
    }
    endian.write_i32(24, &mut header);
    debug_assert_eq!(header.len(), 84);
    write_record(writer, &header, endian)?;

    let mut title = options.title.into_bytes();
    title.truncate(80);
    title.resize(80, b' ');
    let mut title_record = Vec::with_capacity(84);
    endian.write_i32(1, &mut title_record);
    title_record.extend_from_slice(&title);
    write_record(writer, &title_record, endian)?;

    let n_atoms_i32 = i32::try_from(n_atoms)
        .map_err(|_| DcdError::InvalidStructure("too many atoms".to_owned()))?;
    let mut atom_record = Vec::with_capacity(4);
    endian.write_i32(n_atoms_i32, &mut atom_record);
    write_record(writer, &atom_record, endian)?;

    for frame in &coordinates.frames {
        if let Some(dimensions) = frame.dimensions {
            let values = encode_cell(dimensions)?;
            let mut cell = Vec::with_capacity(48);
            for value in values {
                cell.extend_from_slice(&match endian {
                    DcdEndian::Little => value.to_le_bytes(),
                    DcdEndian::Big => value.to_be_bytes(),
                });
            }
            write_record(writer, &cell, endian)?;
        }
        for axis in 0..3 {
            let mut values = Vec::with_capacity(n_atoms * 4);
            for position in &frame.positions {
                let value = position[axis] as f32;
                values.extend_from_slice(&match endian {
                    DcdEndian::Little => value.to_le_bytes(),
                    DcdEndian::Big => value.to_be_bytes(),
                });
            }
            write_record(writer, &values, endian)?;
        }
    }
    Ok(())
}

fn write_record<W: Write>(
    writer: &mut W,
    payload: &[u8],
    endian: DcdEndian,
) -> Result<(), DcdError> {
    let length = i32::try_from(payload.len())
        .map_err(|_| DcdError::InvalidStructure("DCD record is too large".to_owned()))?;
    let marker = match endian {
        DcdEndian::Little => length.to_le_bytes(),
        DcdEndian::Big => length.to_be_bytes(),
    };
    writer.write_all(&marker)?;
    writer.write_all(payload)?;
    writer.write_all(&marker)?;
    Ok(())
}

fn detect_endian(bytes: &[u8]) -> Result<DcdEndian, DcdError> {
    if bytes.len() < 4 {
        return Err(parse_error(
            0,
            "file is shorter than a Fortran record marker",
        ));
    }
    let little = i32::from_le_bytes(bytes[..4].try_into().unwrap());
    let big = i32::from_be_bytes(bytes[..4].try_into().unwrap());
    if little == 84 {
        Ok(DcdEndian::Little)
    } else if big == 84 {
        Ok(DcdEndian::Big)
    } else {
        Err(parse_error(0, "first record is not an 84-byte DCD header"))
    }
}

fn read_record<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    endian: DcdEndian,
) -> Result<&'a [u8], DcdError> {
    let start = *cursor;
    if bytes.len().saturating_sub(*cursor) < 4 {
        return Err(parse_error(start, "missing record marker"));
    }
    let length = endian.read_i32(&bytes[*cursor..*cursor + 4]);
    if length < 0 {
        return Err(parse_error(start, "negative record length"));
    }
    let length =
        usize::try_from(length).map_err(|_| parse_error(start, "record length overflows usize"))?;
    *cursor += 4;
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| parse_error(start, "record length overflows file size"))?;
    if end.saturating_add(4) > bytes.len() {
        return Err(parse_error(start, "record extends past end of file"));
    }
    let payload = &bytes[*cursor..end];
    let trailing = endian.read_i32(&bytes[end..end + 4]);
    if trailing != i32::try_from(length).unwrap_or(i32::MAX) {
        return Err(parse_error(
            start,
            "leading and trailing record lengths differ",
        ));
    }
    *cursor = end + 4;
    Ok(payload)
}

fn read_title_record<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    endian: DcdEndian,
) -> Result<&'a [u8], DcdError> {
    let start = *cursor;
    if bytes.len().saturating_sub(start) < 8 {
        return Err(parse_error(start, "missing title record marker"));
    }
    let marker = peek_record_len(bytes, start, endian)?;
    let payload_start = start + 4;
    let count = endian.read_i32(&bytes[payload_start..payload_start + 4]);
    let count = positive_count(count, "title count")?;
    let expected = 4_usize
        .checked_add(count.saturating_mul(80))
        .ok_or_else(|| parse_error(start, "title count overflows file size"))?;
    let actual_payload = if marker == expected {
        marker
    } else if marker.checked_add(80) == Some(expected) {
        // Some old CHARMM writers report the title marker excluding one
        // 80-byte line even though they write the complete payload.
        expected
    } else {
        return Err(parse_error(
            start,
            format!("title record has {marker} bytes; expected {expected}"),
        ));
    };
    let end = payload_start
        .checked_add(actual_payload)
        .ok_or_else(|| parse_error(start, "title record overflows file size"))?;
    if end.saturating_add(4) > bytes.len() {
        return Err(parse_error(start, "title record extends past end of file"));
    }
    let trailing = endian.read_i32(&bytes[end..end + 4]);
    if trailing != i32::try_from(marker).unwrap_or(i32::MAX) {
        return Err(parse_error(
            start,
            "title record has an unexpected trailing marker",
        ));
    }
    *cursor = end + 4;
    Ok(&bytes[payload_start..end])
}

fn peek_record_len(bytes: &[u8], cursor: usize, endian: DcdEndian) -> Result<usize, DcdError> {
    if bytes.len().saturating_sub(cursor) < 4 {
        return Err(parse_error(cursor, "missing frame record marker"));
    }
    let length = endian.read_i32(&bytes[cursor..cursor + 4]);
    if length < 0 {
        return Err(parse_error(cursor, "negative frame record length"));
    }
    usize::try_from(length).map_err(|_| parse_error(cursor, "frame record length overflows usize"))
}

fn read_coordinate_record(
    bytes: &[u8],
    cursor: &mut usize,
    endian: DcdEndian,
    n_atoms: usize,
    axis: &str,
) -> Result<Vec<f64>, DcdError> {
    let record = read_record(bytes, cursor, endian)?;
    let expected = n_atoms
        .checked_mul(4)
        .ok_or_else(|| parse_error(*cursor, "coordinate record length overflows usize"))?;
    if record.len() != expected {
        return Err(parse_error(
            *cursor,
            format!(
                "{axis} coordinate record contains {} bytes; expected {expected}",
                record.len()
            ),
        ));
    }
    Ok(record
        .chunks(4)
        .map(|value| f64::from(endian.read_f32(value)))
        .collect())
}

fn looks_like_cell_prefix(bytes: &[u8], cursor: usize, endian: DcdEndian, n_atoms: usize) -> bool {
    if n_atoms == 12 {
        return false;
    }
    let after_cell = match read_record_len_only(bytes, cursor, endian) {
        Some((_, next)) => next,
        None => return false,
    };
    let expected = n_atoms.saturating_mul(4);
    peek_record_len(bytes, after_cell, endian).ok() == Some(expected)
}

fn read_record_len_only(bytes: &[u8], cursor: usize, endian: DcdEndian) -> Option<(usize, usize)> {
    let length = peek_record_len(bytes, cursor, endian).ok()?;
    let next = cursor.checked_add(8)?.checked_add(length)?;
    (next <= bytes.len()).then_some((length, next))
}

fn decode_cell(values: [f64; 6]) -> Option<[f64; 6]> {
    if values.iter().all(|value| value.abs() <= f64::EPSILON) {
        return None;
    }
    let [a, gamma, b, beta, alpha, c] = values;
    let cosine_style = [gamma, beta, alpha]
        .iter()
        .all(|value| value.abs() <= 1.0 + 1.0e-6);
    let angles = if cosine_style {
        [alpha.acos(), beta.acos(), gamma.acos()].map(f64::to_degrees)
    } else {
        [alpha, beta, gamma]
    };
    Some([a, b, c, angles[0], angles[1], angles[2]])
}

fn encode_cell(dimensions: [f64; 6]) -> Result<[f64; 6], DcdError> {
    let [a, b, c, alpha, beta, gamma] = dimensions;
    if [a, b, c, alpha, beta, gamma]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || [alpha, beta, gamma].iter().any(|value| *value >= 180.0)
    {
        return Err(DcdError::InvalidStructure(
            "unit-cell lengths and angles are invalid".to_owned(),
        ));
    }
    Ok([
        a,
        gamma.to_radians().cos(),
        b,
        beta.to_radians().cos(),
        alpha.to_radians().cos(),
        c,
    ])
}

fn positive_count(value: i32, name: &str) -> Result<usize, DcdError> {
    usize::try_from(value).map_err(|_| parse_error(0, format!("{name} must be non-negative")))
}

fn parse_error(offset: usize, message: impl Into<String>) -> DcdError {
    DcdError::Parse {
        offset,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcd_round_trip_coordinates_and_cell() {
        let mut first = CoordinateFrame::new(vec![[1.25, 2.5, 3.75], [-1.0, 0.0, 4.0]]);
        first.dimensions = Some([10.0, 11.0, 12.0, 90.0, 80.0, 70.0]);
        let second = CoordinateFrame {
            positions: vec![[2.0, 3.0, 4.0], [-2.0, 1.0, 5.0]],
            dimensions: first.dimensions,
            ..CoordinateFrame::new(Vec::new())
        };
        let source = CoordinateFile::new(vec![first, second]);
        let bytes = source.to_dcd_bytes().unwrap();
        let parsed = DcdFile::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.header.n_frames, 2);
        assert_eq!(parsed.header.n_atoms, 2);
        assert_eq!(
            parsed.coordinates.frames[0].positions,
            source.frames[0].positions
        );
        assert_eq!(
            parsed.coordinates.frames[1].positions,
            source.frames[1].positions
        );
        let dimensions = parsed.coordinates.frames[0].dimensions.unwrap();
        assert!((dimensions[0] - 10.0).abs() < 1.0e-5);
        assert!((dimensions[4] - 80.0).abs() < 1.0e-5);
    }

    #[test]
    fn dcd_supports_big_endian() {
        let source = CoordinateFile::new(vec![CoordinateFrame::new(vec![[1.0, 2.0, 3.0]])]);
        let dcd = DcdFile {
            header: DcdHeader {
                n_frames: 1,
                n_atoms: 1,
                endian: DcdEndian::Big,
                ..DcdHeader::default()
            },
            coordinates: source,
        };
        let bytes = dcd.to_bytes().unwrap();
        let parsed = DcdFile::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.header.endian, DcdEndian::Big);
        assert_eq!(parsed.coordinates.frames[0].positions[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_dcd_is_rejected() {
        assert!(DcdFile::from_bytes(&[]).is_err());
        assert!(DcdFile::from_bytes(&84_i32.to_le_bytes()).is_err());
    }
}
