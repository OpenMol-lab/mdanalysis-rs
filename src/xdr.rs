//! Gromacs XDR trajectory readers and writers.
//!
//! This module implements the portable XTC and TRR formats used by Gromacs.
//! XDR values are big-endian; XTC coordinates use the reference Gromacs
//! integer/bit-packing scheme while TRR stores ordinary floating-point
//! arrays. Coordinates and box lengths are kept in the native nanometre units
//! of these formats.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use crate::mdamath::{triclinic_box, triclinic_vectors};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const XTC_MAGIC: i32 = 1995;
const TRR_MAGIC: i32 = 1993;
const FIRSTIDX: usize = 9;
const MAGIC_INTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256,
    322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321,
    13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063,
    262144, 330280, 416127, 524287, 660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245,
    3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];

/// Errors produced by XTC/TRR parsing and serialization.
#[derive(Debug)]
pub enum XdrError {
    Io(io::Error),
    Parse { offset: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for XdrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { offset, message } => {
                write!(formatter, "XDR parse error at byte {offset}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid XDR structure: {message}")
            }
        }
    }
}

impl std::error::Error for XdrError {}

impl From<io::Error> for XdrError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CoordinateError> for XdrError {
    fn from(error: CoordinateError) -> Self {
        Self::InvalidStructure(error.to_string())
    }
}

fn parse_error(offset: usize, message: impl Into<String>) -> XdrError {
    XdrError::Parse {
        offset,
        message: message.into(),
    }
}

fn checked_count(value: i32, name: &str, offset: usize) -> Result<usize, XdrError> {
    usize::try_from(value).map_err(|_| parse_error(offset, format!("{name} must be non-negative")))
}

fn checked_size(value: i32, name: &str, offset: usize) -> Result<usize, XdrError> {
    checked_count(value, name, offset)
}

/// Metadata and trajectory data parsed from an XTC file.
#[derive(Clone, Debug, PartialEq)]
pub struct XtcFile {
    pub n_atoms: usize,
    pub coordinates: CoordinateFile,
    pub steps: Vec<i32>,
    pub times: Vec<f32>,
    pub precisions: Vec<f32>,
}

impl XtcFile {
    /// Parse an XTC document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, XdrError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse an XTC document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, XdrError> {
        let mut reader = XdrReader::new(bytes);
        let mut frames = Vec::new();
        let mut steps = Vec::new();
        let mut times = Vec::new();
        let mut precisions = Vec::new();
        let mut n_atoms = None;
        while !reader.is_empty() {
            let magic = reader.read_i32()?;
            if magic != XTC_MAGIC {
                return Err(parse_error(
                    reader.offset().saturating_sub(4),
                    format!("expected XTC magic {XTC_MAGIC}, found {magic}"),
                ));
            }
            let count = checked_count(reader.read_i32()?, "atom count", reader.offset())?;
            if count == 0 {
                return Err(parse_error(
                    reader.offset(),
                    "XTC atom count must be positive",
                ));
            }
            if let Some(expected) = n_atoms {
                if expected != count {
                    return Err(XdrError::InvalidStructure(format!(
                        "XTC frame contains {count} atoms; expected {expected}"
                    )));
                }
            } else {
                n_atoms = Some(count);
            }
            let step = reader.read_i32()?;
            let time = reader.read_f32()?;
            let mut box_matrix = [[0.0_f64; 3]; 3];
            for row in &mut box_matrix {
                for value in row {
                    *value = f64::from(reader.read_f32()?);
                }
            }
            let (positions, precision) = read_xtc_coordinates(&mut reader, count)?;
            let mut frame = CoordinateFrame::new(positions);
            frame.dimensions = dimensions_from_box(box_matrix);
            frame.step = usize::try_from(step).unwrap_or(0);
            frame.time = f64::from(time);
            frames.push(frame);
            steps.push(step);
            times.push(time);
            precisions.push(precision);
        }
        let count = n_atoms
            .ok_or_else(|| XdrError::InvalidStructure("XTC file has no frames".to_owned()))?;
        Ok(Self {
            n_atoms: count,
            coordinates: CoordinateFile::new(frames),
            steps,
            times,
            precisions,
        })
    }

    /// Write this XTC document with the supplied coordinate precision.
    pub fn write<W: Write>(&self, writer: W, options: XtcWriteOptions) -> Result<(), XdrError> {
        write_xtc_document(self, writer, options)
    }

    /// Serialize this XTC document to bytes.
    pub fn to_bytes(&self, options: XtcWriteOptions) -> Result<Vec<u8>, XdrError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes, options)?;
        Ok(bytes)
    }
}

/// Options controlling XTC serialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XtcWriteOptions {
    /// Coordinate precision in native nm units. Gromacs commonly uses 1000.
    pub precision: f32,
}

impl Default for XtcWriteOptions {
    fn default() -> Self {
        Self { precision: 1000.0 }
    }
}

/// Metadata and trajectory data parsed from a TRR file.
#[derive(Clone, Debug, PartialEq)]
pub struct TrrFile {
    pub n_atoms: usize,
    pub coordinates: CoordinateFile,
    /// Integration step stored in each frame header.
    pub steps: Vec<i32>,
    /// Simulation time stored in each frame header (native ps units).
    pub times: Vec<f64>,
    /// Per-frame forces, when the source TRR contains them.
    pub forces: Vec<Option<Vec<[f64; 3]>>>,
    /// Lambda values stored in each frame header.
    pub lambdas: Vec<f64>,
    /// Whether each frame used double precision payloads.
    pub double_precision: Vec<bool>,
}

impl TrrFile {
    /// Parse a TRR document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, XdrError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a TRR document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, XdrError> {
        let mut reader = XdrReader::new(bytes);
        let mut frames = Vec::new();
        let mut forces = Vec::new();
        let mut lambdas = Vec::new();
        let mut double_precision = Vec::new();
        let mut steps = Vec::new();
        let mut times = Vec::new();
        let mut n_atoms = None;
        while !reader.is_empty() {
            let frame = read_trr_frame(&mut reader)?;
            if let Some(expected) = n_atoms {
                if expected != frame.positions.len() {
                    return Err(XdrError::InvalidStructure(format!(
                        "TRR frame contains {} atoms; expected {expected}",
                        frame.positions.len()
                    )));
                }
            } else {
                n_atoms = Some(frame.positions.len());
            }
            let mut coordinate = CoordinateFrame::new(frame.positions);
            coordinate.velocities = frame.velocities;
            coordinate.dimensions = frame.dimensions;
            coordinate.step = usize::try_from(frame.step).unwrap_or(0);
            coordinate.time = frame.time;
            frames.push(coordinate);
            forces.push(frame.forces);
            lambdas.push(frame.lambda);
            double_precision.push(frame.double_precision);
            steps.push(frame.step);
            times.push(frame.time);
        }
        let count = n_atoms
            .ok_or_else(|| XdrError::InvalidStructure("TRR file has no frames".to_owned()))?;
        Ok(Self {
            n_atoms: count,
            coordinates: CoordinateFile::new(frames),
            steps,
            times,
            forces,
            lambdas,
            double_precision,
        })
    }

    /// Write this TRR document with the supplied payload precision.
    pub fn write<W: Write>(&self, writer: W, options: TrrWriteOptions) -> Result<(), XdrError> {
        write_trr_document(self, writer, options)
    }

    /// Serialize this TRR document to bytes.
    pub fn to_bytes(&self, options: TrrWriteOptions) -> Result<Vec<u8>, XdrError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes, options)?;
        Ok(bytes)
    }
}

/// Options controlling TRR serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrrPrecision {
    Single,
    Double,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrrWriteOptions {
    pub precision: TrrPrecision,
    pub lambda: f64,
    pub include_velocities: bool,
    pub include_forces: bool,
}

impl Default for TrrWriteOptions {
    fn default() -> Self {
        Self {
            precision: TrrPrecision::Single,
            lambda: 0.0,
            include_velocities: true,
            include_forces: true,
        }
    }
}

/// Read an XTC trajectory from a filesystem path.
pub fn read_xtc<P: AsRef<Path>>(path: P) -> Result<XtcFile, XdrError> {
    XtcFile::read(File::open(path)?)
}

/// Write an XTC trajectory to a filesystem path.
pub fn write_xtc<P: AsRef<Path>>(path: P, coordinates: &CoordinateFile) -> Result<(), XdrError> {
    let file = xtc_from_coordinates(coordinates)?;
    file.write(File::create(path)?, XtcWriteOptions::default())
}

/// Read a TRR trajectory from a filesystem path.
pub fn read_trr<P: AsRef<Path>>(path: P) -> Result<TrrFile, XdrError> {
    TrrFile::read(File::open(path)?)
}

/// Write a TRR trajectory to a filesystem path.
pub fn write_trr<P: AsRef<Path>>(path: P, coordinates: &CoordinateFile) -> Result<(), XdrError> {
    let file = trr_from_coordinates(coordinates)?;
    file.write(File::create(path)?, TrrWriteOptions::default())
}

impl CoordinateFile {
    /// Parse an XTC trajectory from a reader.
    pub fn read_xtc<R: Read>(reader: R) -> Result<Self, XdrError> {
        Ok(XtcFile::read(reader)?.coordinates)
    }

    /// Parse an XTC trajectory from bytes.
    pub fn from_xtc_bytes(bytes: &[u8]) -> Result<Self, XdrError> {
        Ok(XtcFile::from_bytes(bytes)?.coordinates)
    }

    /// Write this coordinate file in XTC format.
    pub fn write_xtc<W: Write>(&self, writer: W) -> Result<(), XdrError> {
        xtc_from_coordinates(self)?.write(writer, XtcWriteOptions::default())
    }

    /// Parse a TRR trajectory from a reader.
    pub fn read_trr<R: Read>(reader: R) -> Result<Self, XdrError> {
        Ok(TrrFile::read(reader)?.coordinates)
    }

    /// Parse a TRR trajectory from bytes.
    pub fn from_trr_bytes(bytes: &[u8]) -> Result<Self, XdrError> {
        Ok(TrrFile::from_bytes(bytes)?.coordinates)
    }

    /// Write this coordinate file in TRR format.
    pub fn write_trr<W: Write>(&self, writer: W) -> Result<(), XdrError> {
        trr_from_coordinates(self)?.write(writer, TrrWriteOptions::default())
    }
}

fn xtc_from_coordinates(coordinates: &CoordinateFile) -> Result<XtcFile, XdrError> {
    let n_atoms = coordinates.n_atoms();
    if n_atoms == 0
        || coordinates
            .frames
            .iter()
            .any(|frame| frame.n_atoms() != n_atoms)
    {
        return Err(XdrError::InvalidStructure(
            "XTC frames must be non-empty and have a consistent atom count".to_owned(),
        ));
    }
    Ok(XtcFile {
        n_atoms,
        coordinates: coordinates.clone(),
        steps: coordinates
            .frames
            .iter()
            .map(|frame| i32::try_from(frame.step).unwrap_or(i32::MAX))
            .collect(),
        times: coordinates
            .frames
            .iter()
            .map(|frame| frame.time as f32)
            .collect(),
        precisions: vec![1000.0; coordinates.frames.len()],
    })
}

fn trr_from_coordinates(coordinates: &CoordinateFile) -> Result<TrrFile, XdrError> {
    let n_atoms = coordinates.n_atoms();
    if n_atoms == 0
        || coordinates
            .frames
            .iter()
            .any(|frame| frame.n_atoms() != n_atoms)
    {
        return Err(XdrError::InvalidStructure(
            "TRR frames must be non-empty and have a consistent atom count".to_owned(),
        ));
    }
    Ok(TrrFile {
        n_atoms,
        coordinates: coordinates.clone(),
        steps: coordinates
            .frames
            .iter()
            .map(|frame| i32::try_from(frame.step).unwrap_or(i32::MAX))
            .collect(),
        times: coordinates.frames.iter().map(|frame| frame.time).collect(),
        forces: vec![None; coordinates.frames.len()],
        lambdas: vec![0.0; coordinates.frames.len()],
        double_precision: vec![false; coordinates.frames.len()],
    })
}

fn write_xtc_document<W: Write>(
    file: &XtcFile,
    mut writer: W,
    options: XtcWriteOptions,
) -> Result<(), XdrError> {
    if file.n_atoms == 0 || file.coordinates.frames.is_empty() {
        return Err(XdrError::InvalidStructure(
            "XTC file must contain at least one frame".to_owned(),
        ));
    }
    if file
        .coordinates
        .frames
        .iter()
        .any(|frame| frame.n_atoms() != file.n_atoms)
    {
        return Err(XdrError::InvalidStructure(
            "XTC frames have inconsistent atom counts".to_owned(),
        ));
    }
    if !options.precision.is_finite() || options.precision <= 0.0 {
        return Err(XdrError::InvalidStructure(
            "XTC precision must be finite and positive".to_owned(),
        ));
    }
    for (index, frame) in file.coordinates.frames.iter().enumerate() {
        write_i32(&mut writer, XTC_MAGIC)?;
        write_i32(
            &mut writer,
            i32::try_from(file.n_atoms)
                .map_err(|_| XdrError::InvalidStructure("too many atoms".to_owned()))?,
        )?;
        write_i32(
            &mut writer,
            file.steps
                .get(index)
                .copied()
                .unwrap_or_else(|| i32::try_from(frame.step).unwrap_or(i32::MAX)),
        )?;
        write_f32(
            &mut writer,
            file.times.get(index).copied().unwrap_or(frame.time as f32),
        )?;
        let matrix = frame
            .dimensions
            .map(triclinic_vectors)
            .unwrap_or([[0.0; 3]; 3]);
        for row in matrix {
            for value in row {
                write_f32(&mut writer, value as f32)?;
            }
        }
        // A writer option controls the precision of every output frame. The
        // parsed precision metadata remains available for diagnostics and is
        // intentionally not reused when rewriting a trajectory.
        write_xtc_coordinates(&mut writer, &frame.positions, options.precision)?;
    }
    Ok(())
}

fn write_trr_document<W: Write>(
    file: &TrrFile,
    mut writer: W,
    options: TrrWriteOptions,
) -> Result<(), XdrError> {
    if file.n_atoms == 0 || file.coordinates.frames.is_empty() {
        return Err(XdrError::InvalidStructure(
            "TRR file must contain at least one frame".to_owned(),
        ));
    }
    if file
        .coordinates
        .frames
        .iter()
        .any(|frame| frame.n_atoms() != file.n_atoms)
    {
        return Err(XdrError::InvalidStructure(
            "TRR frames have inconsistent atom counts".to_owned(),
        ));
    }
    if file.coordinates.frames.iter().any(|frame| {
        frame
            .velocities
            .as_ref()
            .is_some_and(|velocities| velocities.len() != file.n_atoms)
    }) {
        return Err(XdrError::InvalidStructure(
            "TRR velocity arrays have inconsistent atom counts".to_owned(),
        ));
    }
    if file.forces.iter().any(|forces| {
        forces
            .as_ref()
            .is_some_and(|forces| forces.len() != file.n_atoms)
    }) {
        return Err(XdrError::InvalidStructure(
            "TRR force arrays have inconsistent atom counts".to_owned(),
        ));
    }
    for (index, frame) in file.coordinates.frames.iter().enumerate() {
        let use_velocities = options.include_velocities && frame.velocities.is_some();
        let use_forces =
            options.include_forces && file.forces.get(index).and_then(Option::as_ref).is_some();
        let has_box = frame.dimensions.is_some();
        let unit = match options.precision {
            TrrPrecision::Single => 4_i32,
            TrrPrecision::Double => 8_i32,
        };
        let n = i32::try_from(file.n_atoms)
            .map_err(|_| XdrError::InvalidStructure("too many atoms".to_owned()))?;
        write_i32(&mut writer, TRR_MAGIC)?;
        // TRR stores the historical version length followed by the XDR
        // string itself (which has its own length word).
        write_i32(&mut writer, 13)?;
        write_xdr_string(&mut writer, "GMX_trn_file")?;
        write_i32(&mut writer, 0)?; // ir_size
        write_i32(&mut writer, 0)?; // e_size
        write_i32(&mut writer, if has_box { 9 * unit } else { 0 })?;
        write_i32(&mut writer, 0)?; // vir_size
        write_i32(&mut writer, 0)?; // pres_size
        write_i32(&mut writer, 0)?; // top_size
        write_i32(&mut writer, 0)?; // sym_size
        write_i32(&mut writer, 3 * n * unit)?; // x_size
        write_i32(&mut writer, if use_velocities { 3 * n * unit } else { 0 })?;
        write_i32(&mut writer, if use_forces { 3 * n * unit } else { 0 })?;
        write_i32(&mut writer, n)?;
        write_i32(
            &mut writer,
            file.steps.get(index).copied().unwrap_or(index as i32),
        )?;
        write_i32(&mut writer, 0)?; // nre
        write_real(
            &mut writer,
            file.times.get(index).copied().unwrap_or(frame.time),
            options.precision,
        )?;
        write_real(
            &mut writer,
            file.lambdas.get(index).copied().unwrap_or(options.lambda),
            options.precision,
        )?;
        if let Some(dimensions) = frame.dimensions {
            let matrix = triclinic_vectors(dimensions);
            for row in matrix {
                for value in row {
                    write_real(&mut writer, value, options.precision)?;
                }
            }
        }
        let positions = frame
            .positions
            .iter()
            .flat_map(|position| position.iter().copied());
        write_real_iter(&mut writer, positions, options.precision)?;
        if use_velocities {
            let velocities = frame
                .velocities
                .as_ref()
                .expect("checked above")
                .iter()
                .flat_map(|v| v.iter().copied());
            write_real_iter(&mut writer, velocities, options.precision)?;
        }
        if use_forces {
            let force_values = file.forces[index]
                .as_ref()
                .expect("checked above")
                .iter()
                .flat_map(|v| v.iter().copied());
            write_real_iter(&mut writer, force_values, options.precision)?;
        }
    }
    Ok(())
}

fn write_real_iter<W: Write, I: IntoIterator<Item = f64>>(
    writer: &mut W,
    values: I,
    precision: TrrPrecision,
) -> Result<(), XdrError> {
    for value in values {
        write_real(writer, value, precision)?;
    }
    Ok(())
}

fn write_real<W: Write>(
    writer: &mut W,
    value: f64,
    precision: TrrPrecision,
) -> Result<(), XdrError> {
    match precision {
        TrrPrecision::Single => write_f32(writer, value as f32),
        TrrPrecision::Double => write_f64(writer, value),
    }
}

fn write_xdr_string<W: Write>(writer: &mut W, value: &str) -> Result<(), XdrError> {
    // xdr_string encodes the string bytes without the terminating NUL. The
    // TRR header has a separate historical length word (written by caller).
    let length = value
        .len()
        .checked_add(0)
        .ok_or_else(|| XdrError::InvalidStructure("XDR string length overflow".to_owned()))?;
    write_i32(
        writer,
        i32::try_from(length)
            .map_err(|_| XdrError::InvalidStructure("XDR string is too long".to_owned()))?,
    )?;
    writer.write_all(value.as_bytes())?;
    let padding = (4 - (length % 4)) % 4;
    if padding != 0 {
        writer.write_all(&[0; 3][..padding])?;
    }
    Ok(())
}

fn read_xtc_coordinates(
    reader: &mut XdrReader<'_>,
    natoms: usize,
) -> Result<(Vec<[f64; 3]>, f32), XdrError> {
    let count = checked_count(
        reader.read_i32()?,
        "compressed coordinate count",
        reader.offset(),
    )?;
    if count != natoms {
        return Err(XdrError::InvalidStructure(format!(
            "compressed coordinate count {count} differs from header atom count {natoms}"
        )));
    }
    if natoms <= 9 {
        let mut positions = Vec::with_capacity(natoms);
        for _ in 0..natoms {
            positions.push([
                f64::from(reader.read_f32()?),
                f64::from(reader.read_f32()?),
                f64::from(reader.read_f32()?),
            ]);
        }
        return Ok((positions, 0.0));
    }
    let precision = reader.read_f32()?;
    if !precision.is_finite() || precision <= 0.0 {
        return Err(parse_error(
            reader.offset(),
            "XTC precision must be finite and positive",
        ));
    }
    let mut minint = [0_i32; 3];
    let mut maxint = [0_i32; 3];
    for value in &mut minint {
        *value = reader.read_i32()?;
    }
    for value in &mut maxint {
        *value = reader.read_i32()?;
    }
    let mut sizeint = [0_u64; 3];
    for axis in 0..3 {
        let size = i64::from(maxint[axis]) - i64::from(minint[axis]) + 1;
        if size <= 0 {
            return Err(parse_error(reader.offset(), "invalid XTC coordinate range"));
        }
        sizeint[axis] = u64::try_from(size)
            .map_err(|_| parse_error(reader.offset(), "XTC coordinate range overflows"))?;
    }
    let smallidx = checked_count(reader.read_i32()?, "small index", reader.offset())?;
    if !(FIRSTIDX..MAGIC_INTS.len()).contains(&smallidx) {
        return Err(parse_error(
            reader.offset(),
            "XTC small index is out of range",
        ));
    }
    let nbytes = checked_size(reader.read_i32()?, "compressed byte count", reader.offset())?;
    let compressed = reader.read_opaque(nbytes)?;
    let mut bits = BitReader::new(compressed);
    let large_ranges = sizeint.iter().any(|size| *size > 0x00ff_ffff);
    let bitsize = if large_ranges { 0 } else { sizeofints(sizeint) };
    let mut bitsizeint = [0_u32; 3];
    if large_ranges {
        for axis in 0..3 {
            bitsizeint[axis] = sizeofint(sizeint[axis]);
        }
    }
    let mut smallidx_state = smallidx;
    let mut smaller = MAGIC_INTS[smallidx_state.saturating_sub(1).max(FIRSTIDX)] / 2;
    let mut smallnum = MAGIC_INTS[smallidx_state] / 2;
    let mut sizesmall = u64::try_from(MAGIC_INTS[smallidx_state])
        .map_err(|_| parse_error(reader.offset(), "invalid small index"))?;
    let mut result = Vec::with_capacity(natoms);
    let mut run = 0_i32;
    let mut i = 0_usize;
    while i < natoms {
        let mut current = if bitsize == 0 {
            [
                bits.read(bitsizeint[0])? as i32,
                bits.read(bitsizeint[1])? as i32,
                bits.read(bitsizeint[2])? as i32,
            ]
        } else {
            decode_ints(&mut bits, bitsize, sizeint)?
        };
        for axis in 0..3 {
            current[axis] = current[axis]
                .checked_add(minint[axis])
                .ok_or_else(|| parse_error(reader.offset(), "XTC coordinate integer overflow"))?;
        }
        i += 1;
        let mut previous = current;
        let flag = bits.read(1)?;
        let mut is_smaller = 0_i32;
        if flag == 1 {
            let encoded_run = bits.read(5)? as i32;
            is_smaller = encoded_run % 3;
            run = encoded_run - is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            let run_usize = usize::try_from(run)
                .map_err(|_| parse_error(reader.offset(), "invalid XTC run length"))?;
            if run_usize % 3 != 0 || i.checked_add(run_usize / 3).is_none_or(|end| end > natoms) {
                return Err(parse_error(
                    reader.offset(),
                    "XTC run exceeds frame atom count",
                ));
            }
            let mut first = true;
            for _ in (0..run_usize).step_by(3) {
                let mut next = decode_ints(&mut bits, smallidx_state as u32, [sizesmall; 3])?;
                for axis in 0..3 {
                    next[axis] = next[axis]
                        .checked_add(previous[axis] - smallnum)
                        .ok_or_else(|| {
                            parse_error(reader.offset(), "XTC run coordinate overflow")
                        })?;
                }
                i += 1;
                if first {
                    std::mem::swap(&mut next, &mut previous);
                    result.push(scale_coordinate(previous, precision));
                    first = false;
                } else {
                    previous = next;
                }
                result.push(scale_coordinate(next, precision));
            }
        } else {
            result.push(scale_coordinate(current, precision));
        }
        smallidx_state = usize::try_from((smallidx_state as i32) + is_smaller)
            .map_err(|_| parse_error(reader.offset(), "XTC small index underflow"))?;
        if !(FIRSTIDX..MAGIC_INTS.len()).contains(&smallidx_state) {
            return Err(parse_error(
                reader.offset(),
                "XTC small index moved out of range",
            ));
        }
        if is_smaller < 0 {
            smallnum = smaller;
            smaller = if smallidx_state > FIRSTIDX {
                MAGIC_INTS[smallidx_state - 1] / 2
            } else {
                0
            };
        } else if is_smaller > 0 {
            smaller = smallnum;
            smallnum = MAGIC_INTS[smallidx_state] / 2;
        }
        sizesmall = u64::try_from(MAGIC_INTS[smallidx_state])
            .map_err(|_| parse_error(reader.offset(), "invalid XTC small index"))?;
    }
    if result.len() != natoms {
        return Err(parse_error(
            reader.offset(),
            "XTC decompression produced the wrong atom count",
        ));
    }
    Ok((result, precision))
}

fn scale_coordinate(value: [i32; 3], precision: f32) -> [f64; 3] {
    [
        f64::from(value[0]) / f64::from(precision),
        f64::from(value[1]) / f64::from(precision),
        f64::from(value[2]) / f64::from(precision),
    ]
}

fn write_xtc_coordinates<W: Write>(
    writer: &mut W,
    positions: &[[f64; 3]],
    precision: f32,
) -> Result<(), XdrError> {
    write_i32(
        writer,
        i32::try_from(positions.len())
            .map_err(|_| XdrError::InvalidStructure("too many atoms".to_owned()))?,
    )?;
    if positions.len() <= 9 {
        for position in positions {
            for value in position {
                write_f32(writer, *value as f32)?;
            }
        }
        return Ok(());
    }
    let mut integer_positions = Vec::with_capacity(positions.len());
    let mut minint = [i32::MAX; 3];
    let mut maxint = [i32::MIN; 3];
    for position in positions {
        let mut current = [0_i32; 3];
        for axis in 0..3 {
            let value = (position[axis] as f32) * precision;
            if !value.is_finite() || value.abs() > (i32::MAX - 2) as f32 {
                return Err(XdrError::InvalidStructure(
                    "XTC coordinate overflows scaled integer".to_owned(),
                ));
            }
            current[axis] = if value >= 0.0 {
                (value + 0.5) as i32
            } else {
                (value - 0.5) as i32
            };
            minint[axis] = minint[axis].min(current[axis]);
            maxint[axis] = maxint[axis].max(current[axis]);
        }
        integer_positions.push(current);
    }
    write_f32(writer, precision)?;
    for value in minint {
        write_i32(writer, value)?;
    }
    for value in maxint {
        write_i32(writer, value)?;
    }
    let mut sizeint = [0_u64; 3];
    for axis in 0..3 {
        sizeint[axis] = (i64::from(maxint[axis]) - i64::from(minint[axis]) + 1) as u64;
    }
    let large_ranges = sizeint.iter().any(|size| *size > 0x00ff_ffff);
    let bitsize = if large_ranges { 0 } else { sizeofints(sizeint) };
    let mut bitsizeint = [0_u32; 3];
    if large_ranges {
        for axis in 0..3 {
            bitsizeint[axis] = sizeofint(sizeint[axis]);
        }
    }
    let mut smallidx = FIRSTIDX;
    let mut mindiff = i64::MAX;
    let mut previous = [0_i32; 3];
    for (index, current) in integer_positions.iter().enumerate() {
        let diff = coord_diff(*current, previous);
        if index > 0 {
            mindiff = mindiff.min(diff);
        }
        previous = *current;
    }
    while smallidx < MAGIC_INTS.len() && i64::from(MAGIC_INTS[smallidx]) < mindiff {
        smallidx += 1;
    }
    if smallidx >= MAGIC_INTS.len() {
        smallidx = MAGIC_INTS.len() - 1;
    }
    write_i32(writer, i32::try_from(smallidx).unwrap_or(i32::MAX))?;
    let maxidx = (smallidx + 8).min(MAGIC_INTS.len() - 1);
    let minidx = maxidx - 8;
    let mut smaller = MAGIC_INTS[smallidx.saturating_sub(1).max(FIRSTIDX)] / 2;
    let mut smallnum = MAGIC_INTS[smallidx] / 2;
    let mut sizesmall = u64::try_from(MAGIC_INTS[smallidx]).unwrap_or(1);
    let larger = MAGIC_INTS[maxidx] / 2;
    let mut bits = BitWriter::new();
    let mut i = 0_usize;
    let mut previous = [0_i32; 3];
    let mut prevrun = -1_i32;
    while i < integer_positions.len() {
        let mut current = integer_positions[i];
        let mut is_smaller =
            if smallidx < maxidx && i >= 1 && coord_diff(current, previous) < i64::from(larger) {
                1
            } else if smallidx > minidx {
                -1
            } else {
                0
            };
        let mut is_small = false;
        if i + 1 < integer_positions.len()
            && coord_diff(current, integer_positions[i + 1]) < i64::from(smallnum)
        {
            integer_positions.swap(i, i + 1);
            current = integer_positions[i];
            is_small = true;
        }
        encode_base(&mut bits, current, minint, bitsize, bitsizeint, sizeint);
        previous = current;
        i += 1;
        let mut run = 0_i32;
        if !is_small && is_smaller == -1 {
            is_smaller = 0;
        }
        let mut run_values = Vec::new();
        while is_small && run < 24 {
            let next = integer_positions[i];
            let delta = coord_diff(next, previous);
            if is_smaller == -1 && delta >= i64::from(smaller) * i64::from(smaller) {
                is_smaller = 0;
            }
            for axis in 0..3 {
                run_values.push((next[axis] - previous[axis] + smallnum) as u64);
            }
            previous = next;
            i += 1;
            run += 3;
            is_small = i < integer_positions.len()
                && coord_diff(integer_positions[i], previous) < i64::from(smallnum);
        }
        if run != prevrun || is_smaller != 0 {
            prevrun = run;
            bits.write(1, 1);
            bits.write(5, (run + is_smaller + 1) as u32);
        } else {
            bits.write(1, 0);
        }
        for values in run_values.chunks(3) {
            encode_ints(
                &mut bits,
                smallidx as u32,
                [sizesmall; 3],
                [values[0], values[1], values[2]],
            );
        }
        if is_smaller < 0 {
            smallidx -= 1;
            smallnum = smaller;
            smaller = MAGIC_INTS[smallidx.saturating_sub(1).max(FIRSTIDX)] / 2;
        } else if is_smaller > 0 {
            smallidx += 1;
            smaller = smallnum;
            smallnum = MAGIC_INTS[smallidx] / 2;
        }
        sizesmall = u64::try_from(MAGIC_INTS[smallidx]).unwrap_or(1);
    }
    let compressed = bits.finish();
    write_i32(
        writer,
        i32::try_from(compressed.len()).map_err(|_| {
            XdrError::InvalidStructure("compressed XTC frame is too large".to_owned())
        })?,
    )?;
    writer.write_all(&compressed)?;
    let padding = (4 - compressed.len() % 4) % 4;
    if padding != 0 {
        writer.write_all(&[0; 3][..padding])?;
    }
    Ok(())
}

fn encode_base(
    bits: &mut BitWriter,
    value: [i32; 3],
    minint: [i32; 3],
    bitsize: u32,
    bitsizeint: [u32; 3],
    sizeint: [u64; 3],
) {
    let values = [
        (value[0] - minint[0]) as u64,
        (value[1] - minint[1]) as u64,
        (value[2] - minint[2]) as u64,
    ];
    if bitsize == 0 {
        for axis in 0..3 {
            bits.write(bitsizeint[axis], values[axis] as u32);
        }
    } else {
        encode_ints(bits, bitsize, sizeint, values);
    }
}

fn coord_diff(a: [i32; 3], b: [i32; 3]) -> i64 {
    let dx = i64::from(a[0]) - i64::from(b[0]);
    let dy = i64::from(a[1]) - i64::from(b[1]);
    let dz = i64::from(a[2]) - i64::from(b[2]);
    dx.abs() + dy.abs() + dz.abs()
}

fn sizeofint(size: u64) -> u32 {
    if size <= 1 {
        0
    } else {
        64 - (size - 1).leading_zeros()
    }
}

fn sizeofints(sizes: [u64; 3]) -> u32 {
    let product = u128::from(sizes[0])
        .saturating_mul(u128::from(sizes[1]))
        .saturating_mul(u128::from(sizes[2]));
    if product <= 1 {
        0
    } else {
        128 - (product - 1).leading_zeros()
    }
}

fn decode_ints(
    bits: &mut BitReader<'_>,
    bit_count: u32,
    sizes: [u64; 3],
) -> Result<[i32; 3], XdrError> {
    if bit_count > 128 {
        return Err(XdrError::InvalidStructure(
            "XTC integer bit count exceeds 128".to_owned(),
        ));
    }
    let mut value = 0_u128;
    let mut shift = 0_u32;
    let mut remaining = bit_count;
    while remaining > 0 {
        let take = remaining.min(8);
        value |= u128::from(bits.read(take)?) << shift;
        shift += take;
        remaining -= take;
    }
    let mut result = [0_i32; 3];
    for axis in (1..3).rev() {
        if sizes[axis] == 0 {
            return Err(XdrError::InvalidStructure(
                "zero XTC mixed-radix size".to_owned(),
            ));
        }
        result[axis] = i32::try_from(value % u128::from(sizes[axis]))
            .map_err(|_| XdrError::InvalidStructure("XTC integer exceeds i32".to_owned()))?;
        value /= u128::from(sizes[axis]);
    }
    result[0] = i32::try_from(value)
        .map_err(|_| XdrError::InvalidStructure("XTC integer exceeds i32".to_owned()))?;
    Ok(result)
}

fn encode_ints(bits: &mut BitWriter, bit_count: u32, sizes: [u64; 3], values: [u64; 3]) {
    let mut value = u128::from(values[0]);
    value = value * u128::from(sizes[1]) + u128::from(values[1]);
    value = value * u128::from(sizes[2]) + u128::from(values[2]);
    let mut shift = 0_u32;
    let mut remaining = bit_count;
    while remaining > 0 {
        let take = remaining.min(8);
        bits.write(take, ((value >> shift) & ((1_u128 << take) - 1)) as u32);
        shift += take;
        remaining -= take;
    }
}

#[derive(Clone, Debug)]
struct TrrFrameData {
    positions: Vec<[f64; 3]>,
    velocities: Option<Vec<[f64; 3]>>,
    forces: Option<Vec<[f64; 3]>>,
    dimensions: Option<[f64; 6]>,
    step: i32,
    time: f64,
    lambda: f64,
    double_precision: bool,
}

fn read_trr_frame(reader: &mut XdrReader<'_>) -> Result<TrrFrameData, XdrError> {
    let start = reader.offset();
    let magic = reader.read_i32()?;
    if magic != TRR_MAGIC {
        return Err(parse_error(
            start,
            format!("expected TRR magic {TRR_MAGIC}, found {magic}"),
        ));
    }
    let version_len = checked_count(reader.read_i32()?, "TRR version length", reader.offset())?;
    if version_len != 13 {
        return Err(parse_error(
            reader.offset(),
            "unsupported TRR version string length",
        ));
    }
    let version = reader.read_string()?;
    if version != "GMX_trn_file" {
        return Err(parse_error(
            reader.offset(),
            "unsupported TRR version string",
        ));
    }
    let _ir_size = checked_size(reader.read_i32()?, "ir_size", reader.offset())?;
    let _e_size = checked_size(reader.read_i32()?, "e_size", reader.offset())?;
    let box_size = checked_size(reader.read_i32()?, "box_size", reader.offset())?;
    let vir_size = checked_size(reader.read_i32()?, "vir_size", reader.offset())?;
    let pres_size = checked_size(reader.read_i32()?, "pres_size", reader.offset())?;
    let _top_size = checked_size(reader.read_i32()?, "top_size", reader.offset())?;
    let _sym_size = checked_size(reader.read_i32()?, "sym_size", reader.offset())?;
    let x_size = checked_size(reader.read_i32()?, "x_size", reader.offset())?;
    let v_size = checked_size(reader.read_i32()?, "v_size", reader.offset())?;
    let f_size = checked_size(reader.read_i32()?, "f_size", reader.offset())?;
    let natoms = checked_count(reader.read_i32()?, "atom count", reader.offset())?;
    if natoms == 0 {
        return Err(parse_error(
            reader.offset(),
            "TRR atom count must be positive",
        ));
    }
    let step = reader.read_i32()?;
    let _nre = reader.read_i32()?;
    let float_size = if box_size != 0 {
        if box_size % 9 != 0 {
            return Err(parse_error(
                reader.offset(),
                "TRR box_size is not divisible by 9",
            ));
        }
        box_size / 9
    } else if x_size != 0 {
        let denominator = natoms
            .checked_mul(3)
            .ok_or_else(|| parse_error(reader.offset(), "TRR atom count overflows"))?;
        if x_size % denominator != 0 {
            return Err(parse_error(reader.offset(), "TRR x_size has invalid shape"));
        }
        x_size / denominator
    } else if v_size != 0 {
        let denominator = natoms
            .checked_mul(3)
            .ok_or_else(|| parse_error(reader.offset(), "TRR atom count overflows"))?;
        if v_size % denominator != 0 {
            return Err(parse_error(reader.offset(), "TRR v_size has invalid shape"));
        }
        v_size / denominator
    } else if f_size != 0 {
        let denominator = natoms
            .checked_mul(3)
            .ok_or_else(|| parse_error(reader.offset(), "TRR atom count overflows"))?;
        if f_size % denominator != 0 {
            return Err(parse_error(reader.offset(), "TRR f_size has invalid shape"));
        }
        f_size / denominator
    } else {
        return Err(parse_error(
            reader.offset(),
            "TRR frame has no floating-point payload",
        ));
    };
    if float_size != 4 && float_size != 8 {
        return Err(parse_error(
            reader.offset(),
            "TRR floating-point payload must use 4 or 8 bytes",
        ));
    }
    let double_precision = float_size == 8;
    let time = read_real(reader, float_size)?;
    let lambda = read_real(reader, float_size)?;
    let dimensions = if box_size != 0 {
        let mut matrix = [[0.0; 3]; 3];
        for row in &mut matrix {
            for value in row {
                *value = read_real(reader, float_size)?;
            }
        }
        Some(dimensions_from_box(matrix).unwrap_or([0.0; 6]))
    } else {
        None
    };
    skip_values(reader, vir_size, float_size)?;
    skip_values(reader, pres_size, float_size)?;
    let positions = if x_size != 0 {
        read_vectors(reader, natoms, float_size)?
    } else {
        vec![[0.0; 3]; natoms]
    };
    let velocities = if v_size != 0 {
        Some(read_vectors(reader, natoms, float_size)?)
    } else {
        None
    };
    let forces = if f_size != 0 {
        Some(read_vectors(reader, natoms, float_size)?)
    } else {
        None
    };
    Ok(TrrFrameData {
        positions,
        velocities,
        forces,
        dimensions,
        step,
        time,
        lambda,
        double_precision,
    })
}

fn skip_values(
    reader: &mut XdrReader<'_>,
    bytes: usize,
    float_size: usize,
) -> Result<(), XdrError> {
    if bytes == 0 {
        return Ok(());
    }
    if !bytes.is_multiple_of(float_size) {
        return Err(parse_error(
            reader.offset(),
            "TRR matrix size is not aligned to scalar size",
        ));
    }
    for _ in 0..bytes / float_size {
        let _ = read_real(reader, float_size)?;
    }
    Ok(())
}
fn read_vectors(
    reader: &mut XdrReader<'_>,
    count: usize,
    float_size: usize,
) -> Result<Vec<[f64; 3]>, XdrError> {
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push([
            read_real(reader, float_size)?,
            read_real(reader, float_size)?,
            read_real(reader, float_size)?,
        ]);
    }
    Ok(result)
}
fn read_real(reader: &mut XdrReader<'_>, size: usize) -> Result<f64, XdrError> {
    match size {
        4 => Ok(f64::from(reader.read_f32()?)),
        8 => reader.read_f64(),
        _ => Err(parse_error(reader.offset(), "invalid TRR scalar size")),
    }
}
fn dimensions_from_box(matrix: [[f64; 3]; 3]) -> Option<[f64; 6]> {
    if matrix
        .iter()
        .flatten()
        .all(|value| value.abs() <= f64::EPSILON)
    {
        None
    } else {
        Some(triclinic_box(matrix))
    }
}

struct XdrReader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> XdrReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn offset(&self) -> usize {
        self.position
    }
    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
    fn read_exact(&mut self, count: usize) -> Result<&'a [u8], XdrError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| parse_error(self.position, "XDR offset overflow"))?;
        if end > self.bytes.len() {
            return Err(parse_error(
                self.position,
                format!("truncated XDR value: need {count} bytes"),
            ));
        }
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }
    fn read_i32(&mut self) -> Result<i32, XdrError> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_be_bytes(bytes.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, XdrError> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_be_bytes(bytes.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, XdrError> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_be_bytes(bytes.try_into().unwrap()))
    }
    fn read_opaque(&mut self, count: usize) -> Result<&'a [u8], XdrError> {
        let value = self.read_exact(count)?;
        let padding = (4 - count % 4) % 4;
        if padding != 0 {
            self.read_exact(padding)?;
        }
        Ok(value)
    }
    fn read_string(&mut self) -> Result<String, XdrError> {
        let count = checked_count(self.read_i32()?, "TRR string length", self.position)?;
        let bytes = self.read_opaque(count)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| parse_error(self.position, "TRR version is not valid UTF-8"))
    }
}

fn write_i32<W: Write>(writer: &mut W, value: i32) -> Result<(), XdrError> {
    writer.write_all(&value.to_be_bytes()).map_err(XdrError::Io)
}
fn write_f32<W: Write>(writer: &mut W, value: f32) -> Result<(), XdrError> {
    writer.write_all(&value.to_be_bytes()).map_err(XdrError::Io)
}
fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<(), XdrError> {
    writer.write_all(&value.to_be_bytes()).map_err(XdrError::Io)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_position: usize,
}
impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_position: 0,
        }
    }
    fn read(&mut self, bits: u32) -> Result<u32, XdrError> {
        if bits > 32 {
            return Err(XdrError::InvalidStructure(
                "XTC bit read exceeds 32 bits".to_owned(),
            ));
        }
        let end = self
            .bit_position
            .checked_add(bits as usize)
            .ok_or_else(|| XdrError::InvalidStructure("XTC bit offset overflow".to_owned()))?;
        if end > self.bytes.len() * 8 {
            return Err(XdrError::Parse {
                offset: self.bytes.len(),
                message: "truncated XTC compressed bitstream".to_owned(),
            });
        }
        let mut value = 0_u32;
        for _ in 0..bits {
            let byte = self.bytes[self.bit_position / 8];
            let bit = 7 - self.bit_position % 8;
            value = (value << 1) | u32::from((byte >> bit) & 1);
            self.bit_position += 1;
        }
        Ok(value)
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    bit_position: usize,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_position: 0,
        }
    }
    fn write(&mut self, bits: u32, value: u32) {
        if bits == 0 {
            return;
        }
        for index in (0..bits).rev() {
            let bit = ((value >> index) & 1) as u8;
            if self.bit_position.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let byte = self.bit_position / 8;
            let shift = 7 - self.bit_position % 8;
            self.bytes[byte] |= bit << shift;
            self.bit_position += 1;
        }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xtc_fixture_reads_compressed_frames() {
        let path = "/home/bignox/work/test/mdanalysis/mdanalysis/testsuite/MDAnalysisTests/data/xtc_test_only_10_frame_10_atoms.xtc";
        let file = XtcFile::read(File::open(path).unwrap()).unwrap();
        assert_eq!(file.n_atoms, 10);
        assert_eq!(file.coordinates.n_frames(), 10);
        assert_eq!(file.steps, (0..10).collect::<Vec<_>>());
        assert!((file.coordinates.frames[0].positions[0][0] - 0.0).abs() < 1e-6);
        assert!((file.coordinates.frames[1].positions[0][0] - 1.0).abs() < 0.01);
        assert_eq!(file.coordinates.frames[0].dimensions.unwrap()[0], 20.0);
    }

    #[test]
    fn xtc_round_trip_compressed_coordinates() {
        let positions: Vec<[f64; 3]> = (0..10)
            .map(|index| {
                [
                    index as f64 * 0.11,
                    1.0 + index as f64 * 0.07,
                    -0.5 + index as f64 * 0.03,
                ]
            })
            .collect();
        let mut frame = CoordinateFrame::new(positions.clone());
        frame.dimensions = Some([8.0, 9.0, 10.0, 90.0, 90.0, 90.0]);
        let file = xtc_from_coordinates(&CoordinateFile::new(vec![frame])).unwrap();
        let bytes = file.to_bytes(XtcWriteOptions::default()).unwrap();
        let parsed = XtcFile::from_bytes(&bytes).unwrap();
        for (expected, actual) in positions
            .iter()
            .zip(&parsed.coordinates.frames[0].positions)
        {
            for axis in 0..3 {
                assert!((expected[axis] - actual[axis]).abs() <= 0.001);
            }
        }
    }

    #[test]
    fn xtc_large_fixture_reads_all_frames() {
        let path = "/home/bignox/work/test/mdanalysis/mdanalysis/testsuite/MDAnalysisTests/data/adk_oplsaa.xtc";
        let file = XtcFile::read(File::open(path).unwrap()).unwrap();
        assert_eq!(file.n_atoms, 47_681);
        assert_eq!(file.coordinates.n_frames(), 10);
        assert_eq!(file.steps[0], 0);
        assert_eq!(file.steps[9], 450_000);
        assert!(
            file.coordinates.frames[0].positions[0]
                .iter()
                .all(|value| value.is_finite())
        );
        // Canonical MDAnalysis regression value for frame 3, CA of residue
        // 122, checks compressed coordinate values as well as frame count.
        let pdb_path = "/home/bignox/work/test/mdanalysis/mdanalysis/testsuite/MDAnalysisTests/data/adk_oplsaa.pdb";
        let pdb = crate::pdb::read_pdb(pdb_path).unwrap();
        let atom_index = pdb
            .atoms
            .iter()
            .position(|atom| atom.name.trim() == "CA" && atom.residue_sequence == 122)
            .unwrap();
        let position = file.coordinates.frames[2].positions[atom_index];
        assert!((position[0] - 6.043369675).abs() < 0.002);
        assert!((position[1] - 7.385184479).abs() < 0.002);
        assert!((position[2] - 1.381425762).abs() < 0.002);
    }

    #[test]
    fn trr_fixture_reads_velocities() {
        let path = "/home/bignox/work/test/mdanalysis/mdanalysis/testsuite/MDAnalysisTests/data/trr_test_only_10_frame_10_atoms.trr";
        let file = TrrFile::read(File::open(path).unwrap()).unwrap();
        assert_eq!(file.n_atoms, 10);
        assert_eq!(file.coordinates.n_frames(), 10);
        assert!(file.coordinates.frames[0].velocities.is_some());
        assert_eq!(file.coordinates.frames[0].dimensions.unwrap()[0], 20.0);
    }

    #[test]
    fn trr_round_trip_single_and_double() {
        let mut frame = CoordinateFrame::new(vec![[1.25, 2.5, 3.75], [-1.0, 0.0, 4.0]]);
        frame.velocities = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        frame.dimensions = Some([4.0, 5.0, 6.0, 90.0, 80.0, 70.0]);
        let source = CoordinateFile::new(vec![frame]);
        let file = trr_from_coordinates(&source).unwrap();
        for precision in [TrrPrecision::Single, TrrPrecision::Double] {
            let bytes = file
                .to_bytes(TrrWriteOptions {
                    precision,
                    ..TrrWriteOptions::default()
                })
                .unwrap();
            let parsed = TrrFile::from_bytes(&bytes).unwrap();
            assert_eq!(parsed.n_atoms, 2);
            assert!(parsed.coordinates.frames[0].velocities.is_some());
            assert!((parsed.coordinates.frames[0].positions[0][0] - 1.25).abs() < 1e-5);
        }
    }

    #[test]
    fn trr_large_fixture_reads_all_properties() {
        let path = "/home/bignox/work/test/mdanalysis/mdanalysis/testsuite/MDAnalysisTests/data/adk_oplsaa.trr";
        let file = TrrFile::read(File::open(path).unwrap()).unwrap();
        assert_eq!(file.n_atoms, 47_681);
        assert_eq!(file.coordinates.n_frames(), 10);
        assert!(file.coordinates.frames[0].velocities.is_some());
        assert!(file.forces[0].is_none());
        assert_eq!(file.coordinates.frames[2].step, 100_000);
    }

    #[test]
    fn rejects_bad_magic_and_truncated_data() {
        assert!(XtcFile::from_bytes(&[]).is_err());
        assert!(TrrFile::from_bytes(&[0, 0, 7]).is_err());
    }
}
