//! Native reader for HOOMD GSD trajectory files.
//!
//! GSD is a compact binary format made up of a fixed header, a chunk-name
//! table, an index, and typed arrays.  This module reads the HOOMD schema
//! directly, including files written by the older GSD 1.x implementations
//! used by the MDAnalysis fixtures.  Missing chunks inherit their value from
//! the previous frame as required by the GSD specification.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const HEADER_SIZE: usize = 256;
const INDEX_ENTRY_SIZE: usize = 32;
const NAME_SIZE: usize = 64;
const MAGIC: u64 = 0x65df_65df_65df_65df;

/// One HOOMD particle from the first frame (topology metadata).
#[derive(Clone, Debug, PartialEq)]
pub struct GsdParticle {
    pub index: usize,
    pub type_id: usize,
    pub type_name: String,
    pub body: i32,
    pub position: [f64; 3],
    pub velocity: Option<[f64; 3]>,
    pub mass: f64,
    pub charge: f64,
    pub diameter: f64,
}

impl GsdParticle {
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.diameter / 2.0
    }
}

/// A GSD bond, represented with zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GsdBond {
    pub atom1: usize,
    pub atom2: usize,
}

impl GsdBond {
    #[must_use]
    pub const fn new(atom1: usize, atom2: usize) -> Self {
        Self { atom1, atom2 }
    }
}

/// A GSD angle, represented with zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GsdAngle {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
}

/// A GSD dihedral, represented with zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GsdDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

/// A GSD improper, represented with zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GsdImproper {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

/// One coordinate frame in a GSD trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct GsdFrame {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Option<Vec<[f64; 3]>>,
    pub dimensions: Option<[f64; 6]>,
    pub step: usize,
    pub time: f64,
}

impl GsdFrame {
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

/// A parsed HOOMD GSD document.
#[derive(Clone, Debug, PartialEq)]
pub struct GsdFile {
    pub schema: String,
    pub schema_version: (u16, u16),
    pub gsd_version: (u16, u16),
    pub particles: Vec<GsdParticle>,
    pub bonds: Vec<GsdBond>,
    pub angles: Vec<GsdAngle>,
    pub dihedrals: Vec<GsdDihedral>,
    pub impropers: Vec<GsdImproper>,
    pub frames: Vec<GsdFrame>,
    /// Coordinate frames in the common coordinate-file representation.
    pub coordinates: CoordinateFile,
}

/// Naming aliases matching the other format modules.
pub type GsdData = GsdFile;
pub type GsdStructure = GsdFile;
pub type GsdAtom = GsdParticle;

impl GsdFile {
    /// Parse a GSD document from a filesystem path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, GsdError> {
        let mut file = File::open(path)?;
        Self::read(&mut file)
    }

    /// Alias for [`GsdFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GsdError> {
        Self::read_file(path)
    }

    /// Parse a GSD document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, GsdError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a GSD document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GsdError> {
        let raw = RawGsd::parse(bytes)?;
        parse_document(&raw)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.particles.len()
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&GsdFrame> {
        self.frames.get(index)
    }

    /// Convert this parsed file to a [`Universe`].
    pub fn to_universe(&self) -> crate::Result<Universe> {
        if self.particles.is_empty() || self.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "GSD file has no particles or coordinate frames".to_owned(),
            ));
        }
        let atoms = self
            .particles
            .iter()
            .map(|particle| {
                let mut atom = Atom::new(particle.index, &particle.type_name, particle.position);
                atom.atom_type = Some(particle.type_name.clone());
                atom.mass = particle.mass;
                atom.charge = particle.charge;
                atom.element = crate::guesser::guess_element(&particle.type_name, None, None).ok();
                atom.resid = particle.body;
                atom.resname = particle.body.to_string();
                atom.velocity = particle.velocity;
                atom
            })
            .collect::<Vec<_>>();
        let mut topology = Topology::new(atoms);
        for bond in &self.bonds {
            topology.add_bond(Bond::new(bond.atom1, bond.atom2));
        }
        let frames = self
            .frames
            .iter()
            .map(|source| {
                let mut frame = Frame::new(source.positions.clone());
                frame.velocities = source.velocities.clone();
                frame.dimensions = source.dimensions;
                frame.step = source.step;
                frame.time = source.time;
                // Keep the original HOOMD step available through the common
                // named frame-data interface as well as the typed field.
                frame
                    .data
                    .insert("step".to_owned(), vec![source.step as f64]);
                frame
            })
            .collect();
        Ok(Universe {
            topology,
            trajectory: Trajectory::new(frames),
        })
    }
}

/// Read a GSD file from a path.
pub fn read_gsd(path: impl AsRef<Path>) -> Result<GsdFile, GsdError> {
    GsdFile::read_file(path)
}

impl Universe {
    /// Construct a universe from parsed GSD data.
    pub fn from_gsd_file(file: GsdFile) -> crate::Result<Self> {
        file.to_universe()
    }
}

impl CoordinateFile {
    /// Read a GSD document and return its coordinate frames.
    pub fn read_gsd<R: Read>(reader: R) -> Result<Self, GsdError> {
        Ok(GsdFile::read(reader)?.coordinates)
    }

    /// Parse GSD bytes and return the coordinate frames.
    pub fn from_gsd_bytes(bytes: &[u8]) -> Result<Self, GsdError> {
        Ok(GsdFile::from_bytes(bytes)?.coordinates)
    }
}

/// Errors produced while reading a GSD file.
#[derive(Debug)]
pub enum GsdError {
    Io(io::Error),
    Parse { offset: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for GsdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "GSD I/O error: {error}"),
            Self::Parse { offset, message } => {
                write!(formatter, "GSD parse error at byte {offset}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid GSD structure: {message}")
            }
        }
    }
}

impl std::error::Error for GsdError {}

impl From<io::Error> for GsdError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug)]
struct Chunk {
    frame: u64,
    rows: u64,
    columns: u32,
    location: u64,
    name_id: u16,
    data_type: u8,
}

struct RawGsd<'a> {
    bytes: &'a [u8],
    names: Vec<String>,
    chunks: Vec<Chunk>,
    n_frames: usize,
    schema: String,
    schema_version: (u16, u16),
    gsd_version: (u16, u16),
}

impl<'a> RawGsd<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, GsdError> {
        if bytes.len() < HEADER_SIZE {
            return Err(parse_error(0, "file is shorter than the 256-byte header"));
        }
        let magic = u64_at(bytes, 0)?;
        if magic != MAGIC {
            return Err(parse_error(
                0,
                format!("unexpected magic value 0x{magic:016x}"),
            ));
        }
        let index_location = usize_at(bytes, 8)?;
        let index_allocated = usize_at(bytes, 16)?;
        let name_location = usize_at(bytes, 24)?;
        let name_allocated = usize_at(bytes, 32)?;
        let schema_version_raw = u32_at(bytes, 40)?;
        let gsd_version_raw = u32_at(bytes, 44)?;
        let schema_version = (
            (schema_version_raw >> 16) as u16,
            (schema_version_raw & 0xffff) as u16,
        );
        let gsd_version = (
            (gsd_version_raw >> 16) as u16,
            (gsd_version_raw & 0xffff) as u16,
        );
        if index_allocated == 0 || name_allocated == 0 {
            return Err(invalid("header allocates no index or name entries"));
        }
        let index_bytes = index_allocated
            .checked_mul(INDEX_ENTRY_SIZE)
            .ok_or_else(|| invalid("index allocation overflows"))?;
        let index_end = index_location
            .checked_add(index_bytes)
            .ok_or_else(|| invalid("index location overflows"))?;
        if index_location < HEADER_SIZE || index_end > bytes.len() {
            return Err(invalid("index lies outside the file"));
        }
        let name_bytes = name_allocated
            .checked_mul(NAME_SIZE)
            .ok_or_else(|| invalid("name allocation overflows"))?;
        let name_end = name_location
            .checked_add(name_bytes)
            .ok_or_else(|| invalid("name-list location overflows"))?;
        if name_location < HEADER_SIZE || name_end > bytes.len() {
            return Err(invalid("name list lies outside the file"));
        }
        let names = read_names(bytes, name_location, name_allocated)?;
        if names.is_empty() {
            return Err(invalid("name list is empty"));
        }
        let mut chunks = Vec::new();
        let mut ended = false;
        let mut previous_frame = 0;
        let mut previous_name_id = 0;
        for index in 0..index_allocated {
            let offset = index_location + index * INDEX_ENTRY_SIZE;
            let frame = u64_at(bytes, offset)?;
            let rows = u64_at(bytes, offset + 8)?;
            let location = u64_at(bytes, offset + 16)?;
            let columns = u32_at(bytes, offset + 24)?;
            let name_id = u16_at(bytes, offset + 28)?;
            let data_type = bytes[offset + 30];
            let flags = bytes[offset + 31];
            if location == 0 {
                ended = true;
                continue;
            }
            if ended {
                return Err(parse_error(
                    offset,
                    "non-empty index entry follows an empty entry",
                ));
            }
            if usize::from(name_id) >= names.len() {
                return Err(parse_error(
                    offset + 28,
                    "chunk name id is outside the name list",
                ));
            }
            if flags != 0 {
                return Err(parse_error(offset + 31, "unsupported non-zero chunk flags"));
            }
            if !chunks.is_empty()
                && (frame < previous_frame
                    || (frame == previous_frame && name_id < previous_name_id))
            {
                return Err(parse_error(offset, "index entries are not sorted"));
            }
            let element_size = data_type_size(data_type).ok_or_else(|| {
                parse_error(offset + 30, format!("unsupported data type {data_type}"))
            })?;
            let value_count = rows
                .checked_mul(u64::from(columns))
                .ok_or_else(|| parse_error(offset, "chunk dimensions overflow"))?;
            let data_bytes = value_count
                .checked_mul(element_size as u64)
                .ok_or_else(|| parse_error(offset, "chunk byte length overflows"))?;
            let data_end = location
                .checked_add(data_bytes)
                .ok_or_else(|| parse_error(offset, "chunk location overflows"))?;
            if data_end > bytes.len() as u64 {
                return Err(parse_error(offset, "chunk data lies outside the file"));
            }
            previous_frame = frame;
            previous_name_id = name_id;
            chunks.push(Chunk {
                frame,
                rows,
                columns,
                location,
                name_id,
                data_type,
            });
        }
        if chunks.is_empty() {
            return Err(invalid("file contains no data chunks"));
        }
        let max_frame = chunks.last().map_or(0, |chunk| chunk.frame);
        let n_frames = usize::try_from(max_frame)
            .ok()
            .and_then(|frame| frame.checked_add(1))
            .ok_or_else(|| invalid("frame count does not fit in usize"))?;
        if chunks.first().is_none_or(|chunk| chunk.frame != 0) {
            return Err(invalid("first data frame is not frame zero"));
        }
        let schema = fixed_string(bytes, 112, 64)?;
        Ok(Self {
            bytes,
            names,
            chunks,
            n_frames,
            schema,
            schema_version,
            gsd_version,
        })
    }

    fn chunk(&self, frame: usize, name: &str) -> Option<&Chunk> {
        let name_id = self.names.iter().position(|candidate| candidate == name)? as u16;
        self.chunks
            .iter()
            .rev()
            .find(|chunk| chunk.frame <= frame as u64 && chunk.name_id == name_id)
    }

    fn data(&self, chunk: &Chunk) -> &[u8] {
        let size = data_type_size(chunk.data_type).expect("validated chunk type") as u64;
        let length = (chunk.rows * u64::from(chunk.columns) * size) as usize;
        let start = chunk.location as usize;
        &self.bytes[start..start + length]
    }
}

fn parse_document(raw: &RawGsd<'_>) -> Result<GsdFile, GsdError> {
    if raw.schema != "hoomd" {
        return Err(invalid(format!("unsupported schema {:?}", raw.schema)));
    }
    let first_n = scalar_usize(raw, 0, "particles/N")?
        .ok_or_else(|| invalid("frame zero is missing required particles/N chunk"))?;
    if first_n == 0 {
        return Err(invalid("particles/N must be positive"));
    }
    let types = read_types(raw, 0, first_n)?;
    let type_ids =
        read_optional_ints(raw, 0, "particles/typeid")?.unwrap_or_else(|| vec![0; first_n]);
    if type_ids.len() != first_n {
        return Err(invalid(
            "particles/typeid length does not match particles/N",
        ));
    }
    let bodies = read_optional_ints(raw, 0, "particles/body")?.unwrap_or_else(|| vec![-1; first_n]);
    if bodies.len() != first_n {
        return Err(invalid("particles/body length does not match particles/N"));
    }
    let mut body_values = bodies
        .into_iter()
        .map(|value| i32::try_from(value).map_err(|_| invalid("particle body does not fit i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let min_body = body_values.iter().copied().min().unwrap_or(0);
    if min_body < 0 {
        let offset = min_body
            .checked_neg()
            .ok_or_else(|| invalid("particle body normalization overflows"))?;
        for body in &mut body_values {
            *body = body
                .checked_add(offset)
                .ok_or_else(|| invalid("particle body adjustment overflows"))?;
        }
    }
    let positions = read_positions(raw, 0, first_n)?;
    let velocities = read_optional_positions(raw, 0, "particles/velocity", first_n)?;
    let masses = read_optional_floats(raw, 0, "particles/mass", first_n)?
        .unwrap_or_else(|| vec![1.0; first_n]);
    let charges = read_optional_floats(raw, 0, "particles/charge", first_n)?
        .unwrap_or_else(|| vec![0.0; first_n]);
    let diameters = read_optional_floats(raw, 0, "particles/diameter", first_n)?
        .unwrap_or_else(|| vec![1.0; first_n]);
    if masses.len() != first_n || charges.len() != first_n || diameters.len() != first_n {
        return Err(invalid(
            "particle metadata length does not match particles/N",
        ));
    }
    let mut particles = Vec::with_capacity(first_n);
    for index in 0..first_n {
        let type_id =
            usize::try_from(type_ids[index]).map_err(|_| invalid("negative particle type id"))?;
        let type_name = types.get(type_id).cloned().ok_or_else(|| {
            invalid(format!(
                "particle {index} refers to missing type id {type_id}"
            ))
        })?;
        let mass = masses[index];
        let charge = charges[index];
        let diameter = diameters[index];
        if !mass.is_finite()
            || mass < 0.0
            || !charge.is_finite()
            || !diameter.is_finite()
            || diameter < 0.0
        {
            return Err(invalid(format!(
                "particle {index} has invalid mass, charge, or diameter"
            )));
        }
        particles.push(GsdParticle {
            index,
            type_id,
            type_name,
            body: body_values[index],
            position: positions[index],
            velocity: velocities.as_ref().map(|values| values[index]),
            mass,
            charge,
            diameter,
        });
    }
    let bonds = read_groups::<2, GsdBond>(raw, "bonds", first_n, |group| {
        GsdBond::new(group[0], group[1])
    })?;
    let mut angles = read_groups::<3, GsdAngle>(raw, "angles", first_n, |group| GsdAngle {
        atom1: group[0],
        atom2: group[1],
        atom3: group[2],
    })?;
    let mut dihedrals =
        read_groups::<4, GsdDihedral>(raw, "dihedrals", first_n, |group| GsdDihedral {
            atom1: group[0],
            atom2: group[1],
            atom3: group[2],
            atom4: group[3],
        })?;
    let impropers =
        read_groups::<4, GsdImproper>(raw, "impropers", first_n, |group| GsdImproper {
            atom1: group[0],
            atom2: group[1],
            atom3: group[2],
            atom4: group[3],
        })?;
    if angles.is_empty() && !bonds.is_empty() {
        angles = infer_angles(first_n, &bonds);
    }
    if dihedrals.is_empty() && !bonds.is_empty() {
        dihedrals = infer_dihedrals(first_n, &bonds);
    }
    let mut frames = Vec::with_capacity(raw.n_frames);
    let mut coordinate_frames = Vec::with_capacity(raw.n_frames);
    for frame_index in 0..raw.n_frames {
        let n_atoms = scalar_usize(raw, frame_index, "particles/N")?
            .ok_or_else(|| invalid(format!("frame {frame_index} is missing particles/N")))?;
        if n_atoms != first_n {
            return Err(invalid(format!(
                "frame {frame_index} has {n_atoms} atoms; expected {first_n}"
            )));
        }
        let positions = read_positions(raw, frame_index, first_n)?;
        let velocities = read_optional_positions(raw, frame_index, "particles/velocity", first_n)?;
        let dimensions = read_box(raw, frame_index)?;
        let step = scalar_usize(raw, frame_index, "configuration/step")?.unwrap_or(0);
        let source = GsdFrame {
            positions: positions.clone(),
            velocities: velocities.clone(),
            dimensions,
            step,
            time: 0.0,
        };
        let mut coordinate = CoordinateFrame::new(positions);
        coordinate.names = particles
            .iter()
            .map(|particle| particle.type_name.clone())
            .collect();
        coordinate.residue_names = particles
            .iter()
            .map(|particle| particle.body.to_string())
            .collect();
        coordinate.residue_ids = particles.iter().map(|particle| particle.body).collect();
        coordinate.atom_ids = (1..=first_n).collect();
        coordinate.velocities = velocities;
        coordinate.dimensions = dimensions;
        coordinate.step = step;
        frames.push(source);
        coordinate_frames.push(coordinate);
    }
    Ok(GsdFile {
        schema: raw.schema.clone(),
        schema_version: raw.schema_version,
        gsd_version: raw.gsd_version,
        particles,
        bonds,
        angles,
        dihedrals,
        impropers,
        frames,
        coordinates: CoordinateFile::new(coordinate_frames),
    })
}

fn read_names(bytes: &[u8], start: usize, count: usize) -> Result<Vec<String>, GsdError> {
    let mut names = Vec::new();
    let mut ended = false;
    for index in 0..count {
        let offset = start + index * NAME_SIZE;
        let slot = &bytes[offset..offset + NAME_SIZE];
        let end = slot.iter().position(|byte| *byte == 0).unwrap_or(NAME_SIZE);
        if end == 0 {
            ended = true;
            continue;
        }
        if ended {
            return Err(parse_error(
                offset,
                "non-empty name follows an empty name slot",
            ));
        }
        let name = std::str::from_utf8(&slot[..end])
            .map_err(|_| parse_error(offset, "chunk name is not UTF-8"))?;
        names.push(name.to_owned());
    }
    Ok(names)
}

fn read_types(raw: &RawGsd<'_>, frame: usize, n_atoms: usize) -> Result<Vec<String>, GsdError> {
    let Some(chunk) = raw.chunk(frame, "particles/types") else {
        return Ok(vec!["A".to_owned()]);
    };
    if chunk.rows == 0 || chunk.columns == 0 {
        return Err(invalid("particles/types has no string width"));
    }
    let bytes = raw.data(chunk);
    let width =
        usize::try_from(chunk.columns).map_err(|_| invalid("type string width overflows"))?;
    let mut types = Vec::with_capacity(chunk.rows as usize);
    for row in 0..chunk.rows as usize {
        let slot = &bytes[row * width..(row + 1) * width];
        let end = slot.iter().position(|byte| *byte == 0).unwrap_or(width);
        let name = std::str::from_utf8(&slot[..end])
            .map_err(|_| invalid("particle type is not UTF-8"))?
            .trim()
            .to_owned();
        if name.is_empty() {
            return Err(invalid("particle type name is empty"));
        }
        types.push(name);
    }
    if types.is_empty() || n_atoms == 0 {
        return Err(invalid("particle types are empty"));
    }
    Ok(types)
}

fn read_positions(
    raw: &RawGsd<'_>,
    frame: usize,
    n_atoms: usize,
) -> Result<Vec<[f64; 3]>, GsdError> {
    read_optional_positions(raw, frame, "particles/position", n_atoms)?
        .ok_or_else(|| invalid(format!("frame {frame} is missing particles/position")))
}

fn read_optional_positions(
    raw: &RawGsd<'_>,
    frame: usize,
    name: &str,
    n_atoms: usize,
) -> Result<Option<Vec<[f64; 3]>>, GsdError> {
    let Some(chunk) = raw.chunk(frame, name) else {
        return Ok(None);
    };
    if chunk.rows as usize != n_atoms || chunk.columns != 3 || !matches!(chunk.data_type, 9 | 10) {
        return Err(invalid(format!(
            "{name} has inconsistent dimensions or data type"
        )));
    }
    let values = read_floats(raw, chunk)?;
    let positions = values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| [value[0], value[1], value[2]])
        .collect();
    Ok(Some(positions))
}

fn read_optional_floats(
    raw: &RawGsd<'_>,
    frame: usize,
    name: &str,
    expected: usize,
) -> Result<Option<Vec<f64>>, GsdError> {
    let Some(chunk) = raw.chunk(frame, name) else {
        return Ok(None);
    };
    if chunk.rows as usize != expected || chunk.columns != 1 || !matches!(chunk.data_type, 9 | 10) {
        return Err(invalid(format!(
            "{name} has inconsistent dimensions or data type"
        )));
    }
    Ok(Some(read_floats(raw, chunk)?))
}

fn read_box(raw: &RawGsd<'_>, frame: usize) -> Result<Option<[f64; 6]>, GsdError> {
    let Some(chunk) = raw.chunk(frame, "configuration/box") else {
        return Ok(None);
    };
    if chunk.rows != 6 || chunk.columns != 1 || !matches!(chunk.data_type, 9 | 10) {
        return Err(invalid(
            "configuration/box must contain six floating-point values",
        ));
    }
    let values = read_floats(raw, chunk)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("configuration/box contains non-finite values"));
    }
    if values[..3].iter().all(|value| *value == 0.0) {
        return Ok(None);
    }
    if values[..3].iter().any(|value| *value <= 0.0) {
        return Err(invalid("configuration/box contains a non-positive length"));
    }
    let dimensions = triclinic_box([
        [values[0], 0.0, 0.0],
        [values[3], values[1], 0.0],
        [values[4], values[5], values[2]],
    ]);
    Ok(Some(dimensions))
}

fn scalar_usize(raw: &RawGsd<'_>, frame: usize, name: &str) -> Result<Option<usize>, GsdError> {
    let Some(chunk) = raw.chunk(frame, name) else {
        return Ok(None);
    };
    if chunk.rows != 1 || chunk.columns != 1 {
        return Err(invalid(format!("{name} must be a scalar")));
    }
    let value = read_chunk_ints(raw, name, chunk)?
        .first()
        .copied()
        .ok_or_else(|| invalid(format!("{name} has no scalar value")))?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| invalid(format!("{name} contains a negative or oversized value")))
}

fn read_optional_ints(
    raw: &RawGsd<'_>,
    frame: usize,
    name: &str,
) -> Result<Option<Vec<i64>>, GsdError> {
    let Some(chunk) = raw.chunk(frame, name) else {
        return Ok(None);
    };
    Ok(Some(read_chunk_ints(raw, name, chunk)?))
}

fn read_chunk_ints(raw: &RawGsd<'_>, name: &str, chunk: &Chunk) -> Result<Vec<i64>, GsdError> {
    let bytes = raw.data(chunk);
    let count = usize::try_from(
        chunk
            .rows
            .checked_mul(u64::from(chunk.columns))
            .ok_or_else(|| invalid("integer chunk dimensions overflow"))?,
    )
    .map_err(|_| invalid("integer chunk is too large"))?;
    let mut result = Vec::with_capacity(count);
    let width = data_type_size(chunk.data_type)
        .ok_or_else(|| invalid(format!("unsupported {name} data type")))?;
    for i in 0..count {
        let offset = i * width;
        let value = match chunk.data_type {
            1 => i64::from(bytes[offset]),
            2 => i64::from(u16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            )),
            3 => i64::from(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            )),
            4 => i64::try_from(u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap(),
            ))
            .map_err(|_| invalid(format!("{name} contains an oversized integer")))?,
            5 => i64::from(i8::from_le_bytes([bytes[offset]])),
            6 => i64::from(i16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            )),
            7 => i64::from(i32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            )),
            8 => i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()),
            _ => return Err(invalid(format!("{name} is not an integer chunk"))),
        };
        result.push(value);
    }
    Ok(result)
}

fn read_floats(raw: &RawGsd<'_>, chunk: &Chunk) -> Result<Vec<f64>, GsdError> {
    let bytes = raw.data(chunk);
    let count = usize::try_from(
        chunk
            .rows
            .checked_mul(u64::from(chunk.columns))
            .ok_or_else(|| invalid("floating chunk dimensions overflow"))?,
    )
    .map_err(|_| invalid("floating chunk is too large"))?;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let value = match chunk.data_type {
            9 => f64::from(f32::from_le_bytes(
                bytes[i * 4..i * 4 + 4].try_into().unwrap(),
            )),
            10 => f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()),
            _ => return Err(invalid("chunk is not floating-point")),
        };
        if !value.is_finite() {
            return Err(invalid("chunk contains a non-finite floating-point value"));
        }
        result.push(value);
    }
    Ok(result)
}

fn read_groups<const M: usize, T>(
    raw: &RawGsd<'_>,
    prefix: &str,
    n_atoms: usize,
    make: impl Fn([usize; M]) -> T,
) -> Result<Vec<T>, GsdError> {
    let group_name = format!("{prefix}/group");
    let count_name = format!("{prefix}/N");
    let Some(group_chunk) = raw.chunk(0, &group_name) else {
        return Ok(Vec::new());
    };
    if group_chunk.columns as usize != M || !matches!(group_chunk.data_type, 3 | 4) {
        return Err(invalid(format!(
            "{group_name} has inconsistent dimensions or data type"
        )));
    }
    let declared = scalar_usize(raw, 0, &count_name)?.unwrap_or(group_chunk.rows as usize);
    if declared != group_chunk.rows as usize {
        return Err(invalid(format!(
            "{group_name} row count disagrees with {count_name}"
        )));
    }
    let values = read_chunk_ints(raw, &group_name, group_chunk)?;
    let mut groups = Vec::with_capacity(declared);
    let mut seen = HashSet::new();
    for row in 0..declared {
        let mut group = [0usize; M];
        for column in 0..M {
            let index = usize::try_from(values[row * M + column])
                .map_err(|_| invalid(format!("{group_name} contains a negative atom index")))?;
            if index >= n_atoms {
                return Err(invalid(format!(
                    "{group_name} references atom {index} outside 0..{n_atoms}"
                )));
            }
            group[column] = index;
        }
        let canonical = canonical_group(group);
        if seen.insert(canonical) {
            groups.push(make(group));
        }
    }
    Ok(groups)
}

fn canonical_group<const M: usize>(group: [usize; M]) -> [usize; M] {
    let mut reversed = group;
    reversed.reverse();
    if reversed < group { reversed } else { group }
}

fn infer_angles(n_atoms: usize, bonds: &[GsdBond]) -> Vec<GsdAngle> {
    let adjacency = adjacency(n_atoms, bonds);
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for (center, neighbors) in adjacency.iter().enumerate().take(n_atoms) {
        for left in 0..neighbors.len() {
            for right in left + 1..neighbors.len() {
                let group = [neighbors[left], center, neighbors[right]];
                if seen.insert(canonical_group(group)) {
                    result.push(GsdAngle {
                        atom1: group[0],
                        atom2: group[1],
                        atom3: group[2],
                    });
                }
            }
        }
    }
    result
}

fn infer_dihedrals(n_atoms: usize, bonds: &[GsdBond]) -> Vec<GsdDihedral> {
    let adjacency = adjacency(n_atoms, bonds);
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for bond in bonds {
        for &left in &adjacency[bond.atom1] {
            if left == bond.atom2 {
                continue;
            }
            for &right in &adjacency[bond.atom2] {
                if right == bond.atom1 {
                    continue;
                }
                let group = [left, bond.atom1, bond.atom2, right];
                if seen.insert(canonical_group(group)) {
                    result.push(GsdDihedral {
                        atom1: group[0],
                        atom2: group[1],
                        atom3: group[2],
                        atom4: group[3],
                    });
                }
            }
        }
    }
    result
}

fn adjacency(n_atoms: usize, bonds: &[GsdBond]) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); n_atoms];
    for bond in bonds {
        result[bond.atom1].push(bond.atom2);
        result[bond.atom2].push(bond.atom1);
    }
    result
}

fn data_type_size(data_type: u8) -> Option<usize> {
    match data_type {
        1 | 5 => Some(1),
        2 | 6 => Some(2),
        3 | 7 => Some(4),
        4 | 8 => Some(8),
        9 => Some(4),
        10 => Some(8),
        11 => Some(1),
        _ => None,
    }
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, GsdError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| parse_error(offset, "offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| parse_error(offset, "truncated u64"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, GsdError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| parse_error(offset, "offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| parse_error(offset, "truncated u32"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, GsdError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| parse_error(offset, "offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| parse_error(offset, "truncated u16"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn usize_at(bytes: &[u8], offset: usize) -> Result<usize, GsdError> {
    usize::try_from(u64_at(bytes, offset)?)
        .map_err(|_| parse_error(offset, "value does not fit usize"))
}

fn fixed_string(bytes: &[u8], offset: usize, width: usize) -> Result<String, GsdError> {
    let slot = bytes
        .get(offset..offset + width)
        .ok_or_else(|| parse_error(offset, "truncated fixed string"))?;
    let end = slot.iter().position(|byte| *byte == 0).unwrap_or(width);
    std::str::from_utf8(&slot[..end])
        .map(str::to_owned)
        .map_err(|_| parse_error(offset, "fixed string is not UTF-8"))
}

fn parse_error(offset: usize, message: impl Into<String>) -> GsdError {
    GsdError::Parse {
        offset,
        message: message.into(),
    }
}

fn invalid(message: impl Into<String>) -> GsdError {
    GsdError::InvalidStructure(message.into())
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
    fn reads_multiframe_fixture() {
        let file = read_gsd(fixture("example.gsd")).unwrap();
        assert_eq!(file.n_atoms(), 5832);
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.frames[0].step, 0);
        assert_eq!(file.frames[1].step, 500);
        assert!((file.frames[0].positions[0][0] + 5.400000095).abs() < 1e-6);
        assert_eq!(file.frames[0].dimensions.unwrap()[0], 21.600000381469727);
        assert_eq!(file.particles.len(), 5832);
        assert_eq!(file.particles[0].body, 0);
        assert_eq!(file.angles.len(), 0);
    }

    #[test]
    fn reads_bond_groups_and_infers_when_missing() {
        let file = read_gsd(fixture("example_bonds.gsd")).unwrap();
        assert_eq!(file.n_atoms(), 490);
        assert_eq!(file.n_frames(), 3);
        assert_eq!(file.bonds.len(), 441);
        assert_eq!(file.angles.len(), 392);
        assert_eq!(file.dihedrals.len(), 343);
        assert!(
            file.bonds
                .iter()
                .any(|bond| bond.atom1 == 0 && bond.atom2 == 1)
        );
        assert_eq!(file.frames[2].step, 200);
    }

    #[test]
    fn universe_constructor_preserves_frames() {
        let file = read_gsd(fixture("example.gsd")).unwrap();
        let universe = Universe::from_gsd_file(file).unwrap();
        assert_eq!(universe.n_atoms(), 5832);
        assert_eq!(universe.n_frames(), 2);
        assert_eq!(universe.trajectory.frames[1].step, 500);
        assert_eq!(universe.trajectory.frames[1].data["step"], vec![500.0]);
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(matches!(
            GsdFile::from_bytes(&[0; 8]),
            Err(GsdError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_malformed_optional_integer_chunks() {
        let mut bytes = std::fs::read(fixture("example.gsd")).expect("fixture bytes");
        let raw = RawGsd::parse(&bytes).expect("fixture should parse");
        let name_id = raw
            .names
            .iter()
            .position(|name| name == "particles/typeid")
            .expect("typeid chunk name");
        let index_location = usize_at(&bytes, 8).expect("index location");
        let index_allocated = usize_at(&bytes, 16).expect("index count");
        let entry = (0..index_allocated)
            .map(|index| index_location + index * INDEX_ENTRY_SIZE)
            .find(|&offset| {
                u64_at(&bytes, offset).ok() == Some(0)
                    && u16_at(&bytes, offset + 28).ok() == Some(name_id as u16)
            })
            .expect("typeid index entry");
        // Type 11 is a byte/string value in GSD, not an integer.  The index
        // remains structurally valid, so this exercises parse_document's
        // handling of an invalid optional chunk rather than header parsing.
        bytes[entry + 30] = 11;
        assert!(matches!(
            GsdFile::from_bytes(&bytes),
            Err(GsdError::InvalidStructure(message)) if message.contains("particles/typeid")
        ));
    }

    #[test]
    fn rejects_body_normalization_overflow() {
        let mut bytes = std::fs::read(fixture("example.gsd")).expect("fixture bytes");
        let raw = RawGsd::parse(&bytes).expect("fixture should parse");
        let body_chunk = raw.chunk(0, "particles/body").expect("body chunk").clone();
        assert_eq!(
            body_chunk.data_type, 7,
            "fixture body chunk should be signed i32"
        );
        let location = usize::try_from(body_chunk.location).expect("body location");
        bytes[location..location + 4].copy_from_slice(&i32::MIN.to_le_bytes());
        assert!(matches!(
            GsdFile::from_bytes(&bytes),
            Err(GsdError::InvalidStructure(message)) if message.contains("normalization overflows")
        ));
    }
}
