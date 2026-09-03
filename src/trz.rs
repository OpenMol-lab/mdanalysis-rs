//! Legacy IBIsCO/YASP TRZ binary trajectory support.
//!
//! TRZ is a little-endian, Fortran-record-like format containing Cartesian
//! coordinates, velocities, optional forces, unit-cell vectors, and a small
//! set of thermodynamic values.  Values are kept in the native units used by
//! TRZ (nanometres, nanometres/ps, and ps), consistent with the XTC/TRR
//! readers in this crate.

use crate::coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Frame, Topology, Trajectory, Universe};
use crate::mdamath::{triclinic_box, triclinic_vectors};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const HEADER_SIZE: usize = 100;
const HEADER_TITLE_SIZE: usize = 80;

/// Header metadata in a TRZ trajectory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrzHeader {
    pub title: String,
    pub has_forces: bool,
    pub n_atoms: usize,
}

/// A parsed TRZ trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct TrzFile {
    pub header: TrzHeader,
    pub coordinates: CoordinateFile,
    /// The source `ntrj` value for each frame (the integration step).
    pub steps: Vec<i32>,
    /// The source one-based `nframe` value for each frame.
    pub frame_numbers: Vec<i32>,
    pub pressure: Vec<f64>,
    pub pressure_tensor: Vec<[f64; 6]>,
    pub total_energy: Vec<f64>,
    pub potential_energy: Vec<f64>,
    pub kinetic_energy: Vec<f64>,
    pub temperature: Vec<f64>,
    /// Per-frame forces, when the source contains a force payload.
    pub forces: Vec<Option<Vec<[f64; 3]>>>,
}

/// Options controlling TRZ serialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrzWriteOptions {
    pub title: String,
    pub include_forces: bool,
}

impl Default for TrzWriteOptions {
    fn default() -> Self {
        Self {
            title: "TRZ".to_owned(),
            include_forces: false,
        }
    }
}

/// Errors produced while reading or writing TRZ files.
#[derive(Debug)]
pub enum TrzError {
    Io(io::Error),
    Parse { offset: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for TrzError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "TRZ I/O error: {error}"),
            Self::Parse { offset, message } => {
                write!(formatter, "TRZ parse error at byte {offset}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid TRZ structure: {message}")
            }
        }
    }
}

impl std::error::Error for TrzError {}

impl From<io::Error> for TrzError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CoordinateError> for TrzError {
    fn from(error: CoordinateError) -> Self {
        Self::InvalidStructure(error.to_string())
    }
}

impl TrzFile {
    /// Parse a TRZ document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, TrzError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a TRZ document held in memory, inferring atom count from its
    /// first frame.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrzError> {
        Self::parse_bytes(bytes, None)
    }

    /// Parse a TRZ document and require a particular atom count.  This is
    /// useful when a separate topology (such as PSF) supplies the count.
    pub fn from_bytes_with_n_atoms(bytes: &[u8], n_atoms: usize) -> Result<Self, TrzError> {
        Self::parse_bytes(bytes, Some(n_atoms))
    }

    fn parse_bytes(bytes: &[u8], expected_n_atoms: Option<usize>) -> Result<Self, TrzError> {
        let mut reader = Reader::new(bytes);
        let header = read_header(&mut reader)?;
        let mut coordinates = Vec::new();
        let mut steps = Vec::new();
        let mut frame_numbers = Vec::new();
        let mut pressure = Vec::new();
        let mut pressure_tensor = Vec::new();
        let mut total_energy = Vec::new();
        let mut potential_energy = Vec::new();
        let mut kinetic_energy = Vec::new();
        let mut temperature = Vec::new();
        let mut forces = Vec::new();
        let mut n_atoms = expected_n_atoms;

        while !reader.is_empty() {
            let frame_offset = reader.offset();
            let frame = read_frame(&mut reader, header.has_forces, n_atoms)?;
            if let Some(expected) = n_atoms {
                if frame.n_atoms != expected {
                    return Err(parse_error(
                        frame_offset,
                        format!(
                            "frame contains {} atoms; expected {expected}",
                            frame.n_atoms
                        ),
                    ));
                }
            } else {
                n_atoms = Some(frame.n_atoms);
            }
            let mut coordinate = CoordinateFrame::new(frame.positions);
            coordinate.velocities = Some(frame.velocities);
            coordinate.dimensions = box_dimensions(frame.box_vectors);
            coordinate.title.clone_from(&header.title);
            coordinate.step = usize::try_from(frame.step).unwrap_or(0);
            coordinate.time = frame.time;
            coordinates.push(coordinate);
            steps.push(frame.step);
            frame_numbers.push(frame.frame_number);
            pressure.push(frame.pressure);
            pressure_tensor.push(frame.pressure_tensor);
            total_energy.push(frame.total_energy);
            potential_energy.push(frame.potential_energy);
            kinetic_energy.push(frame.kinetic_energy);
            temperature.push(frame.temperature);
            forces.push(frame.forces);
        }

        let n_atoms = n_atoms.ok_or_else(|| {
            TrzError::InvalidStructure("TRZ trajectory contains no frames".to_owned())
        })?;
        if coordinates.is_empty() {
            return Err(TrzError::InvalidStructure(
                "TRZ trajectory contains no frames".to_owned(),
            ));
        }
        Ok(Self {
            header: TrzHeader {
                title: header.title,
                has_forces: header.has_forces,
                n_atoms,
            },
            coordinates: CoordinateFile::new(coordinates),
            steps,
            frame_numbers,
            pressure,
            pressure_tensor,
            total_energy,
            potential_energy,
            kinetic_energy,
            temperature,
            forces,
        })
    }

    /// Parse a TRZ document from a filesystem path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, TrzError> {
        Self::read(File::open(path)?)
    }

    /// Number of atoms in each frame.
    #[must_use]
    pub const fn n_atoms(&self) -> usize {
        self.header.n_atoms
    }

    /// Number of frames in the trajectory.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    /// Write the trajectory using the original title and force mode.
    pub fn write<W: Write>(&self, writer: W) -> Result<(), TrzError> {
        self.write_with_options(
            writer,
            TrzWriteOptions {
                title: self.header.title.clone(),
                include_forces: self.header.has_forces,
            },
        )
    }

    /// Write the trajectory with an explicit title and force mode.
    pub fn write_with_options<W: Write>(
        &self,
        mut writer: W,
        options: TrzWriteOptions,
    ) -> Result<(), TrzError> {
        write_document(self, &mut writer, options)
    }

    /// Serialize the trajectory to little-endian bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TrzError> {
        let mut bytes = Vec::new();
        self.write(&mut bytes)?;
        Ok(bytes)
    }
}

/// Read a TRZ trajectory from a path.
pub fn read_trz(path: impl AsRef<Path>) -> Result<TrzFile, TrzError> {
    TrzFile::read_file(path)
}

/// Write a coordinate file as a TRZ trajectory.
pub fn write_trz(path: impl AsRef<Path>, coordinates: &CoordinateFile) -> Result<(), TrzError> {
    coordinate_file_to_trz(coordinates).write(File::create(path)?)
}

impl CoordinateFile {
    /// Read TRZ coordinates, velocities, dimensions, and timing.
    pub fn read_trz<R: Read>(reader: R) -> Result<Self, TrzError> {
        Ok(TrzFile::read(reader)?.coordinates)
    }

    /// Parse TRZ coordinates from bytes.
    pub fn from_trz_bytes(bytes: &[u8]) -> Result<Self, TrzError> {
        Ok(TrzFile::from_bytes(bytes)?.coordinates)
    }

    /// Write this coordinate file as a TRZ trajectory.
    pub fn write_trz<W: Write>(&self, writer: W) -> Result<(), TrzError> {
        coordinate_file_to_trz(self).write(writer)
    }

    /// Serialize this coordinate file as TRZ bytes.
    pub fn to_trz_bytes(&self) -> Result<Vec<u8>, TrzError> {
        let mut bytes = Vec::new();
        self.write_trz(&mut bytes)?;
        Ok(bytes)
    }
}

fn coordinate_file_to_trz(coordinates: &CoordinateFile) -> TrzFile {
    let n_atoms = coordinates.n_atoms();
    let frame_count = coordinates.n_frames();
    let pressure = vec![0.0; frame_count];
    let pressure_tensor = vec![[0.0; 6]; frame_count];
    let total_energy = vec![0.0; frame_count];
    let potential_energy = vec![0.0; frame_count];
    let kinetic_energy = vec![0.0; frame_count];
    let temperature = vec![0.0; frame_count];
    let forces = vec![None; frame_count];
    let steps = coordinates
        .frames
        .iter()
        .map(|frame| i32::try_from(frame.step).unwrap_or(i32::MAX))
        .collect();
    let frame_numbers = coordinates
        .frames
        .iter()
        .enumerate()
        .map(|(index, _)| i32::try_from(index.saturating_add(1)).unwrap_or(i32::MAX))
        .collect();
    // Keep the vectors aligned with frames even for an empty input; write()
    // emits the useful validation error in that case.
    TrzFile {
        header: TrzHeader {
            title: "TRZ".to_owned(),
            has_forces: false,
            n_atoms,
        },
        coordinates: coordinates.clone(),
        steps,
        frame_numbers,
        pressure,
        pressure_tensor,
        total_energy,
        potential_energy,
        kinetic_energy,
        temperature,
        forces,
    }
}

impl Universe {
    /// Construct a universe from a TRZ trajectory without a separate topology.
    pub fn from_trz(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_trz_file(read_trz(path)?)
    }

    /// Construct a universe from TRZ bytes without a separate topology.
    pub fn from_trz_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_trz_file(TrzFile::from_bytes(bytes)?)
    }

    /// Construct a universe from PSF topology and a TRZ trajectory.
    pub fn from_psf_and_trz(
        psf_path: impl AsRef<Path>,
        trz_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf(psf_path)?;
        universe.attach_trz(read_trz(trz_path)?)?;
        Ok(universe)
    }

    /// Construct a universe from PSF text and TRZ bytes.
    pub fn from_psf_and_trz_bytes(psf: &str, trz: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_psf_str(psf)?;
        universe.attach_trz(TrzFile::from_bytes_with_n_atoms(trz, universe.n_atoms())?)?;
        Ok(universe)
    }

    fn from_trz_file(file: TrzFile) -> crate::Result<Self> {
        let first = file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("TRZ trajectory contains no frames".to_owned())
        })?;
        let atoms = first
            .positions
            .iter()
            .enumerate()
            .map(|(index, position)| Atom::new(index, "X", *position))
            .collect::<Vec<_>>();
        let mut universe = Self {
            topology: Topology::new(atoms),
            trajectory: Trajectory::default(),
        };
        universe.set_trz_frames(file)?;
        Ok(universe)
    }

    fn attach_trz(&mut self, file: TrzFile) -> crate::Result<()> {
        if file.n_atoms() != self.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "TRZ contains {} atoms, topology contains {}",
                file.n_atoms(),
                self.n_atoms()
            )));
        }
        self.set_trz_frames(file)
    }

    fn set_trz_frames(&mut self, file: TrzFile) -> crate::Result<()> {
        if file.coordinates.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "TRZ trajectory contains no frames".to_owned(),
            ));
        }
        let TrzFile {
            coordinates,
            forces,
            ..
        } = file;
        let mut frames = Vec::with_capacity(coordinates.frames.len());
        for (index, coordinate) in coordinates.frames.into_iter().enumerate() {
            let mut frame = Frame::new(coordinate.positions);
            frame.velocities = coordinate.velocities;
            frame.dimensions = coordinate.dimensions;
            frame.time = coordinate.time;
            frame.step = coordinate.step;
            if let Some(force) = forces.get(index).cloned().flatten() {
                frame.forces = Some(force);
            }
            if frame.positions.len() != self.n_atoms() {
                return Err(crate::Error::InvalidInput(format!(
                    "TRZ frame contains {} atoms, topology contains {}",
                    frame.positions.len(),
                    self.n_atoms()
                )));
            }
            frames.push(frame);
        }
        self.trajectory = Trajectory::new(frames);
        Ok(())
    }
}

struct ParsedHeader {
    title: String,
    has_forces: bool,
}

struct ParsedFrame {
    frame_number: i32,
    step: i32,
    n_atoms: usize,
    time: f64,
    box_vectors: [[f64; 3]; 3],
    pressure: f64,
    pressure_tensor: [f64; 6],
    total_energy: f64,
    potential_energy: f64,
    kinetic_energy: f64,
    temperature: f64,
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    forces: Option<Vec<[f64; 3]>>,
}

fn parse_error(offset: usize, message: impl Into<String>) -> TrzError {
    TrzError::Parse {
        offset,
        message: message.into(),
    }
}

fn read_header(reader: &mut Reader<'_>) -> Result<ParsedHeader, TrzError> {
    if reader.remaining() < HEADER_SIZE {
        return Err(parse_error(reader.offset(), "TRZ header is truncated"));
    }
    let first = reader.read_i32()?;
    if first != 80 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "invalid header marker",
        ));
    }
    let title_bytes = reader.read_bytes(HEADER_TITLE_SIZE)?;
    let title = String::from_utf8_lossy(title_bytes)
        .trim_matches(['\0', ' '])
        .to_owned();
    let second = reader.read_i32()?;
    if second != 80 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "invalid title marker",
        ));
    }
    let third = reader.read_i32()?;
    if third != 4 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "invalid force marker",
        ));
    }
    let force_mode = reader.read_i32()?;
    let has_forces = match force_mode {
        10 => false,
        20 => true,
        _ => {
            return Err(parse_error(
                reader.offset().saturating_sub(4),
                format!("unsupported force mode {force_mode}; expected 10 or 20"),
            ));
        }
    };
    let fourth = reader.read_i32()?;
    if fourth != 4 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "invalid header terminator",
        ));
    }
    Ok(ParsedHeader { title, has_forces })
}

fn read_frame(
    reader: &mut Reader<'_>,
    has_forces: bool,
    expected_n_atoms: Option<usize>,
) -> Result<ParsedFrame, TrzError> {
    let marker = reader.read_i32()?;
    if marker != 20 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "invalid frame marker",
        ));
    }
    let frame_number = reader.read_i32()?;
    let step = reader.read_i32()?;
    let raw_n_atoms = reader.read_i32()?;
    let n_atoms = usize::try_from(raw_n_atoms)
        .map_err(|_| parse_error(reader.offset().saturating_sub(4), "atom count is negative"))?;
    if n_atoms == 0 {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            "atom count is zero",
        ));
    }
    if let Some(expected) = expected_n_atoms
        && expected != n_atoms
    {
        return Err(parse_error(
            reader.offset().saturating_sub(4),
            format!("frame contains {n_atoms} atoms; expected {expected}"),
        ));
    }
    let time = reader.read_f64()?;
    if !time.is_finite() {
        return Err(parse_error(
            reader.offset().saturating_sub(8),
            "time is not finite",
        ));
    }
    let p2a = reader.read_i32()?;
    let p2b = reader.read_i32()?;
    if p2a != 20 || p2b != 72 {
        return Err(parse_error(
            reader.offset().saturating_sub(8),
            "invalid box markers",
        ));
    }
    let mut box_vectors = [[0.0_f64; 3]; 3];
    for row in &mut box_vectors {
        for value in row {
            *value = reader.read_f64()?;
            if !value.is_finite() {
                return Err(parse_error(
                    reader.offset().saturating_sub(8),
                    "box value is not finite",
                ));
            }
        }
    }
    let p3a = reader.read_i32()?;
    let p3b = reader.read_i32()?;
    if p3a != 72 || p3b != 56 {
        return Err(parse_error(
            reader.offset().saturating_sub(8),
            "invalid pressure markers",
        ));
    }
    let pressure = reader.read_f64()?;
    let mut pressure_tensor = [0.0; 6];
    for value in &mut pressure_tensor {
        *value = reader.read_f64()?;
    }
    if !pressure.is_finite() || pressure_tensor.iter().any(|value| !value.is_finite()) {
        return Err(parse_error(
            reader.offset(),
            "thermodynamic values are not finite",
        ));
    }
    // The next three integers vary slightly between IBIsCO and YASP writers;
    // they are record markers and have no effect on the decoded data.
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let total_energy = reader.read_f64()?;
    let potential_energy = reader.read_f64()?;
    let kinetic_energy = reader.read_f64()?;
    let temperature = reader.read_f64()?;
    for value in [total_energy, potential_energy, kinetic_energy, temperature] {
        if !value.is_finite() {
            return Err(parse_error(
                reader.offset(),
                "energy or temperature is not finite",
            ));
        }
    }
    for _ in 0..6 {
        let _ = reader.read_i32()?;
    }
    let x = read_float_array(reader, n_atoms)?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let y = read_float_array(reader, n_atoms)?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let z = read_float_array(reader, n_atoms)?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let vx = read_float_array(reader, n_atoms)?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let vy = read_float_array(reader, n_atoms)?;
    let _ = reader.read_i32()?;
    let _ = reader.read_i32()?;
    let vz = read_float_array(reader, n_atoms)?;
    let forces = if has_forces {
        let _ = reader.read_i32()?;
        let _ = reader.read_i32()?;
        let fx = read_float_array(reader, n_atoms)?;
        let _ = reader.read_i32()?;
        let _ = reader.read_i32()?;
        let fy = read_float_array(reader, n_atoms)?;
        let _ = reader.read_i32()?;
        let _ = reader.read_i32()?;
        let fz = read_float_array(reader, n_atoms)?;
        let _ = reader.read_i32()?;
        Some(
            fx.into_iter()
                .zip(fy)
                .zip(fz)
                .map(|((x, y), z)| [f64::from(x), f64::from(y), f64::from(z)])
                .collect(),
        )
    } else {
        let _ = reader.read_i32()?;
        None
    };
    let positions = x
        .into_iter()
        .zip(y)
        .zip(z)
        .map(|((x, y), z)| [f64::from(x), f64::from(y), f64::from(z)])
        .collect();
    let velocities = vx
        .into_iter()
        .zip(vy)
        .zip(vz)
        .map(|((x, y), z)| [f64::from(x), f64::from(y), f64::from(z)])
        .collect();
    Ok(ParsedFrame {
        frame_number,
        step,
        n_atoms,
        time,
        box_vectors,
        pressure,
        pressure_tensor,
        total_energy,
        potential_energy,
        kinetic_energy,
        temperature,
        positions,
        velocities,
        forces,
    })
}

fn read_float_array(reader: &mut Reader<'_>, count: usize) -> Result<Vec<f32>, TrzError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let value = reader.read_f32()?;
        if !value.is_finite() {
            return Err(parse_error(
                reader.offset().saturating_sub(4),
                "array value is not finite",
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn box_dimensions(vectors: [[f64; 3]; 3]) -> Option<[f64; 6]> {
    if vectors
        .iter()
        .flatten()
        .all(|value| value.abs() <= f64::EPSILON)
    {
        None
    } else {
        let dimensions = triclinic_box(vectors);
        (dimensions[..3].iter().all(|value| *value > f64::EPSILON)).then_some(dimensions)
    }
}

fn write_document<W: Write>(
    file: &TrzFile,
    writer: &mut W,
    options: TrzWriteOptions,
) -> Result<(), TrzError> {
    if options.title.len() > HEADER_TITLE_SIZE {
        return Err(TrzError::InvalidStructure(
            "TRZ title must contain at most 80 bytes".to_owned(),
        ));
    }
    let n_atoms = file.header.n_atoms;
    if n_atoms == 0 {
        return Err(TrzError::InvalidStructure(
            "TRZ trajectory contains no atoms".to_owned(),
        ));
    }
    if file.coordinates.frames.is_empty() {
        return Err(TrzError::InvalidStructure(
            "TRZ trajectory contains no frames".to_owned(),
        ));
    }
    if file
        .coordinates
        .frames
        .iter()
        .any(|frame| frame.n_atoms() != n_atoms)
    {
        return Err(TrzError::InvalidStructure(
            "all TRZ frames must contain the same atom count".to_owned(),
        ));
    }
    let metadata_len = file.coordinates.frames.len();
    for (name, length) in [
        ("steps", file.steps.len()),
        ("frame_numbers", file.frame_numbers.len()),
        ("pressure", file.pressure.len()),
        ("pressure_tensor", file.pressure_tensor.len()),
        ("total_energy", file.total_energy.len()),
        ("potential_energy", file.potential_energy.len()),
        ("kinetic_energy", file.kinetic_energy.len()),
        ("temperature", file.temperature.len()),
        ("forces", file.forces.len()),
    ] {
        if length != 0 && length != metadata_len {
            return Err(TrzError::InvalidStructure(format!(
                "{name} metadata has {length} entries; expected {metadata_len}"
            )));
        }
    }
    write_i32(writer, 80)?;
    let mut title = [b' '; HEADER_TITLE_SIZE];
    title[..options.title.len()].copy_from_slice(options.title.as_bytes());
    writer.write_all(&title)?;
    write_i32(writer, 80)?;
    write_i32(writer, 4)?;
    write_i32(writer, if options.include_forces { 20 } else { 10 })?;
    write_i32(writer, 4)?;

    for (index, frame) in file.coordinates.frames.iter().enumerate() {
        let frame_number = file
            .frame_numbers
            .get(index)
            .copied()
            .unwrap_or_else(|| i32::try_from(index.saturating_add(1)).unwrap_or(i32::MAX));
        if frame_number <= 0 {
            return Err(TrzError::InvalidStructure(
                "TRZ frame numbers must be positive".to_owned(),
            ));
        }
        let step = file
            .steps
            .get(index)
            .copied()
            .unwrap_or_else(|| i32::try_from(frame.step).unwrap_or(i32::MAX));
        if !frame.time.is_finite()
            || frame
                .positions
                .iter()
                .flat_map(|position| position.iter())
                .any(|value| !value.is_finite())
            || frame
                .velocities
                .as_ref()
                .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
        {
            return Err(TrzError::InvalidStructure(
                "TRZ frame contains non-finite data".to_owned(),
            ));
        }
        write_i32(writer, 20)?;
        write_i32(writer, frame_number)?;
        write_i32(writer, step)?;
        write_i32(
            writer,
            i32::try_from(n_atoms)
                .map_err(|_| TrzError::InvalidStructure("too many atoms for TRZ".to_owned()))?,
        )?;
        write_f64(writer, frame.time)?;
        write_i32(writer, 20)?;
        write_i32(writer, 72)?;
        let vectors = frame
            .dimensions
            .map(triclinic_vectors)
            .unwrap_or([[0.0; 3]; 3]);
        for row in vectors {
            for value in row {
                write_f64(writer, value)?;
            }
        }
        write_i32(writer, 72)?;
        write_i32(writer, 56)?;
        write_f64(writer, value_at(&file.pressure, index))?;
        for value in value_at_array(&file.pressure_tensor, index) {
            write_f64(writer, value)?;
        }
        write_i32(writer, 56)?;
        write_i32(writer, 60)?;
        write_i32(writer, 6)?;
        write_f64(writer, value_at(&file.total_energy, index))?;
        write_f64(writer, value_at(&file.potential_energy, index))?;
        write_f64(writer, value_at(&file.kinetic_energy, index))?;
        write_f64(writer, value_at(&file.temperature, index))?;
        write_f64(writer, 0.0)?;
        write_f64(writer, 0.0)?;
        write_i32(writer, 60)?;
        let size = i32::try_from(n_atoms.checked_mul(4).ok_or_else(|| {
            TrzError::InvalidStructure("TRZ coordinate payload is too large".to_owned())
        })?)
        .map_err(|_| {
            TrzError::InvalidStructure("TRZ coordinate payload is too large".to_owned())
        })?;
        write_i32(writer, size)?;
        write_array(writer, frame.positions.iter().map(|value| value[0] as f32))?;
        write_i32(writer, size)?;
        write_i32(writer, size)?;
        write_array(writer, frame.positions.iter().map(|value| value[1] as f32))?;
        write_i32(writer, size)?;
        write_i32(writer, size)?;
        write_array(writer, frame.positions.iter().map(|value| value[2] as f32))?;
        write_i32(writer, size)?;
        write_i32(writer, size)?;
        let velocities = frame
            .velocities
            .clone()
            .unwrap_or_else(|| vec![[0.0; 3]; n_atoms]);
        if velocities.len() != n_atoms {
            return Err(TrzError::InvalidStructure(
                "velocity count does not match atom count".to_owned(),
            ));
        }
        write_array(writer, velocities.iter().map(|value| value[0] as f32))?;
        write_i32(writer, size)?;
        write_i32(writer, size)?;
        write_array(writer, velocities.iter().map(|value| value[1] as f32))?;
        write_i32(writer, size)?;
        write_i32(writer, size)?;
        write_array(writer, velocities.iter().map(|value| value[2] as f32))?;
        if options.include_forces {
            write_i32(writer, size)?;
            write_i32(writer, size)?;
            let force = file.forces.get(index).and_then(Option::as_ref);
            let force = force.cloned().unwrap_or_else(|| vec![[0.0; 3]; n_atoms]);
            if force.len() != n_atoms {
                return Err(TrzError::InvalidStructure(
                    "force count does not match atom count".to_owned(),
                ));
            }
            write_array(writer, force.iter().map(|value| value[0] as f32))?;
            write_i32(writer, size)?;
            write_i32(writer, size)?;
            write_array(writer, force.iter().map(|value| value[1] as f32))?;
            write_i32(writer, size)?;
            write_i32(writer, size)?;
            write_array(writer, force.iter().map(|value| value[2] as f32))?;
            write_i32(writer, size)?;
        } else {
            write_i32(writer, size)?;
        }
    }
    Ok(())
}

fn value_at(values: &[f64], index: usize) -> f64 {
    values.get(index).copied().unwrap_or(0.0)
}

fn value_at_array(values: &[[f64; 6]], index: usize) -> [f64; 6] {
    values.get(index).copied().unwrap_or([0.0; 6])
}

fn write_array<W: Write>(
    writer: &mut W,
    values: impl IntoIterator<Item = f32>,
) -> Result<(), TrzError> {
    for value in values {
        if !value.is_finite() {
            return Err(TrzError::InvalidStructure(
                "TRZ array contains non-finite data".to_owned(),
            ));
        }
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i32<W: Write>(writer: &mut W, value: i32) -> Result<(), TrzError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<(), TrzError> {
    if !value.is_finite() {
        return Err(TrzError::InvalidStructure(
            "TRZ scalar contains non-finite data".to_owned(),
        ));
    }
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], TrzError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| parse_error(self.offset, "TRZ offset overflows usize"))?;
        if end > self.bytes.len() {
            return Err(parse_error(self.offset, "TRZ payload is truncated"));
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn read_i32(&mut self) -> Result<i32, TrzError> {
        Ok(i32::from_le_bytes(
            self.read_bytes(4)?.try_into().expect("four-byte slice"),
        ))
    }

    fn read_f32(&mut self) -> Result<f32, TrzError> {
        Ok(f32::from_le_bytes(
            self.read_bytes(4)?.try_into().expect("four-byte slice"),
        ))
    }

    fn read_f64(&mut self) -> Result<f64, TrzError> {
        Ok(f64::from_le_bytes(
            self.read_bytes(8)?.try_into().expect("eight-byte slice"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data")
            .join(name)
    }

    #[test]
    fn reads_legacy_fixture() {
        let file = TrzFile::read_file(fixture("trzfile.trz")).unwrap();
        assert_eq!(file.n_atoms(), 8184);
        assert_eq!(file.n_frames(), 6);
        assert_eq!(file.header.title.len(), 80);
        assert!(!file.header.has_forces);
        assert!((file.coordinates.frames[0].time - 0.01).abs() < 1.0e-12);
        assert_eq!(
            file.coordinates.frames[0].dimensions.unwrap()[0],
            5.542283058166504
        );
        assert!((file.coordinates.frames[0].positions[41][0] - 7.23163681).abs() < 1.0e-6);
        assert!(
            (file.coordinates.frames[0].velocities.as_ref().unwrap()[41][0] - 1.48329744).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn attaches_fixture_to_psf_topology() {
        let universe =
            Universe::from_psf_and_trz(fixture("trz_psf.psf"), fixture("trzfile.trz")).unwrap();
        assert_eq!(universe.n_atoms(), 8184);
        assert_eq!(universe.trajectory.n_frames(), 6);
        assert!(!universe.topology.atoms[0].name.is_empty());
        assert!((universe.trajectory.frames[0].time - 0.01).abs() < 1.0e-12);
        assert!(universe.trajectory.frames[0].velocities.is_some());
    }

    #[test]
    fn round_trips_coordinates_and_forces() {
        let mut frame = CoordinateFrame::new(vec![[1.25, 2.5, 3.75], [-1.0, 0.0, 4.0]]);
        frame.velocities = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        frame.dimensions = Some([8.0, 9.0, 10.0, 90.0, 80.0, 70.0]);
        let source = TrzFile {
            header: TrzHeader {
                title: "Test title TRZ".to_owned(),
                has_forces: true,
                n_atoms: 2,
            },
            coordinates: CoordinateFile::new(vec![frame]),
            steps: vec![4],
            frame_numbers: vec![5],
            pressure: vec![1.0],
            pressure_tensor: vec![[1.0; 6]],
            total_energy: vec![2.0],
            potential_energy: vec![3.0],
            kinetic_energy: vec![4.0],
            temperature: vec![5.0],
            forces: vec![Some(vec![[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])],
        };
        let bytes = source.to_bytes().unwrap();
        let parsed = TrzFile::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.header.title, "Test title TRZ");
        assert_eq!(parsed.steps, vec![4]);
        assert_eq!(parsed.frame_numbers, vec![5]);
        assert_eq!(parsed.coordinates.frames[0].positions.len(), 2);
        assert!(parsed.forces[0].is_some());
        assert!((parsed.coordinates.frames[0].positions[0][0] - 1.25).abs() < 1.0e-6);
    }

    #[test]
    fn coordinate_file_writer_uses_sequential_frame_numbers() {
        let mut first = CoordinateFrame::new(vec![[1.0, 2.0, 3.0]]);
        first.step = 100;
        let mut second = CoordinateFrame::new(vec![[4.0, 5.0, 6.0]]);
        second.step = 250;
        let coordinates = CoordinateFile::new(vec![first, second]);

        let parsed = TrzFile::from_bytes(&coordinates.to_trz_bytes().unwrap()).unwrap();
        assert_eq!(parsed.steps, vec![100, 250]);
        assert_eq!(parsed.frame_numbers, vec![1, 2]);
    }

    #[test]
    fn rejects_truncated_and_invalid_headers() {
        assert!(TrzFile::from_bytes(&[]).is_err());
        let mut bytes = vec![0; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&80_i32.to_le_bytes());
        assert!(TrzFile::from_bytes(&bytes).is_err());
    }
}
