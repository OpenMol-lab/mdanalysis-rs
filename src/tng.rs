//! TNG trajectory support backed by the pure-Rust `tng-rs` reader.
//!
//! TNG stores trajectory data at integrator steps and commonly uses sparse
//! blocks (for example, one coordinate sample every 5000 steps).  This module
//! materializes the samples in logical trajectory order and records the
//! integrator step on every frame.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use flate2::read::GzDecoder;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tng_rs::data::DataType;
use tng_rs::gen_block::BlockID;
use tng_rs::trajectory::Trajectory as TngTrajectory;

const SPECIAL_BLOCKS: [BlockID; 4] = [
    BlockID::TrajPositions,
    BlockID::TrajBoxShape,
    BlockID::TrajVelocities,
    BlockID::TrajForces,
];

/// Parsed TNG trajectory data, with values converted to MDAnalysis base units.
#[derive(Clone, Debug, PartialEq)]
pub struct TngFile {
    pub n_atoms: usize,
    pub coordinates: CoordinateFile,
    /// Optional per-frame force vectors.
    pub forces: Vec<Option<Vec<[f64; 3]>>>,
    /// Additional non-special blocks keyed by their TNG block name.  Each
    /// entry contains one flattened value vector per logical frame.
    pub additional_blocks: BTreeMap<String, Vec<Vec<f64>>>,
    /// Names of every data block found in the trajectory.
    pub blocks: Vec<String>,
    /// Names of the present special blocks (positions, box, velocities,
    /// forces).
    pub special_blocks: Vec<String>,
    /// Integrator-step interval shared by the special blocks.
    pub global_stride: usize,
}

pub type TngData = TngFile;
pub type TngStructure = TngFile;

/// Errors produced while reading a TNG trajectory.
#[derive(Debug)]
pub enum TngError {
    Io(io::Error),
    Tng(String),
    InvalidStructure(String),
}

impl fmt::Display for TngError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "TNG I/O error: {error}"),
            Self::Tng(error) => write!(formatter, "TNG reader error: {error}"),
            Self::InvalidStructure(error) => write!(formatter, "invalid TNG structure: {error}"),
        }
    }
}

impl std::error::Error for TngError {}

impl From<io::Error> for TngError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<tng_rs::TngError> for TngError {
    fn from(error: tng_rs::TngError) -> Self {
        Self::Tng(error.to_string())
    }
}

impl TngFile {
    /// Read a TNG trajectory from a path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, TngError> {
        Self::read_file_with_options(path, true)
    }

    /// Alias for [`TngFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TngError> {
        Self::read_file(path)
    }

    pub fn read_file_unconverted(path: impl AsRef<Path>) -> Result<Self, TngError> {
        Self::read_file_with_options(path, false)
    }

    pub fn read_file_with_options(
        path: impl AsRef<Path>,
        convert_units: bool,
    ) -> Result<Self, TngError> {
        let path = path.as_ref();
        File::open(path)?;
        let mut trajectory = TngTrajectory::new();
        trajectory.util_trajectory_open(path, 'r')?;
        parse_trajectory(&mut trajectory, convert_units)
    }

    /// Parse a TNG document supplied in memory.  `tng-rs` is path based, so a
    /// uniquely named temporary file is used for the duration of the parse.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TngError> {
        Self::from_bytes_with_options(bytes, true)
    }

    pub fn from_bytes_unconverted(bytes: &[u8]) -> Result<Self, TngError> {
        Self::from_bytes_with_options(bytes, false)
    }

    pub fn from_bytes_with_options(bytes: &[u8], convert_units: bool) -> Result<Self, TngError> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| TngError::InvalidStructure(format!("system clock error: {error}")))?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("mdanalysis-rs-{}-{stamp}.tng", std::process::id()));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(bytes)?;
            drop(file);
            Self::read_file_with_options(&path, convert_units)
        })();
        let _ = std::fs::remove_file(&path);
        result
    }

    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    pub const fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    pub fn to_universe(&self) -> crate::Result<Universe> {
        universe_from_tng(self.clone())
    }
}

/// Read a TNG trajectory from a path.
pub fn read_tng(path: impl AsRef<Path>) -> Result<TngFile, TngError> {
    TngFile::read_file(path)
}

impl CoordinateFile {
    /// Read a TNG trajectory into the common coordinate container.
    pub fn read_tng(path: impl AsRef<Path>) -> Result<Self, TngError> {
        Ok(TngFile::read_file(path)?.coordinates)
    }

    pub fn from_tng_bytes(bytes: &[u8]) -> Result<Self, TngError> {
        Ok(TngFile::from_bytes(bytes)?.coordinates)
    }
}

impl Universe {
    pub fn from_tng(path: impl AsRef<Path>) -> crate::Result<Self> {
        TngFile::read_file(path)?.to_universe()
    }

    pub fn from_tng_file(file: TngFile) -> crate::Result<Self> {
        file.to_universe()
    }

    pub fn from_psf_and_tng(
        psf_path: impl AsRef<Path>,
        tng_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf(psf_path)?;
        attach_tng(&mut universe, TngFile::read_file(tng_path)?)?;
        Ok(universe)
    }

    pub fn from_psf_and_tng_bytes(psf: &str, tng: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_psf_str(psf)?;
        attach_tng(&mut universe, TngFile::from_bytes(tng)?)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_tng(
        topology_path: impl AsRef<Path>,
        tng_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop(topology_path)?;
        attach_tng(&mut universe, TngFile::read_file(tng_path)?)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_tng_bytes(topology: &str, tng: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop_str(topology)?;
        attach_tng(&mut universe, TngFile::from_bytes(tng)?)?;
        Ok(universe)
    }

    /// Attach a TNG trajectory to a GROMACS GRO topology.  Gzip-compressed
    /// GRO files are accepted as they are common alongside TNG trajectories.
    pub fn from_gro_and_tng(
        gro_path: impl AsRef<Path>,
        tng_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let gro = read_text_maybe_gzip(gro_path)?;
        let mut universe = Self::from_gro_str(&gro)?;
        attach_tng(&mut universe, TngFile::read_file(tng_path)?)?;
        Ok(universe)
    }

    pub fn from_gro_and_tng_bytes(gro: &str, tng: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_gro_str(gro)?;
        attach_tng(&mut universe, TngFile::from_bytes(tng)?)?;
        Ok(universe)
    }
}

fn read_text_maybe_gzip(path: impl AsRef<Path>) -> Result<String, TngError> {
    let bytes = std::fs::read(path)?;
    let mut text = String::new();
    if bytes.starts_with(&[0x1f, 0x8b]) {
        GzDecoder::new(bytes.as_slice()).read_to_string(&mut text)?;
    } else {
        text = String::from_utf8(bytes)
            .map_err(|error| TngError::InvalidStructure(format!("GRO is not UTF-8: {error}")))?;
    }
    Ok(text)
}

#[derive(Default)]
struct BlockSamples {
    values: Vec<Vec<f64>>,
    steps: Vec<i64>,
    stride: Option<i64>,
    n_values: Option<i64>,
    data_type: Option<DataType>,
}

fn parse_trajectory(
    trajectory: &mut TngTrajectory,
    convert_units: bool,
) -> Result<TngFile, TngError> {
    let n_atoms = usize::try_from(trajectory.num_particles_get())
        .map_err(|_| TngError::InvalidStructure("negative atom count".to_owned()))?;
    if n_atoms == 0 {
        return Err(TngError::InvalidStructure(
            "trajectory has no atoms".to_owned(),
        ));
    }
    let distance_exponent = trajectory.distance_unit_exponential_get();
    let length_factor = if convert_units {
        let length_exponent = distance_exponent
            .checked_add(10)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| {
                TngError::InvalidStructure("invalid distance-unit exponent".to_owned())
            })?;
        10f64.powi(length_exponent)
    } else {
        1.0
    };
    let time_per_frame = trajectory.time_per_frame;
    let mut samples: BTreeMap<BlockID, BlockSamples> = BTreeMap::new();
    let mut block_names = BTreeMap::<BlockID, String>::new();
    let mut additional_blocks = BTreeMap::new();
    let mut static_additional_names = BTreeSet::new();
    let mut position_times = Vec::new();
    let mut first_set = true;
    let mut global_stride = None;

    // Header blocks are not attached to a frame set. Read them before the
    // first frame set, while tng-rs still resolves particle data globally.
    let non_tr_blocks: Vec<(BlockID, bool, i64, String)> = trajectory
        .non_tr_particle_data
        .iter()
        .map(|data| {
            (
                data.block_id,
                true,
                data.stride_length,
                data.block_name.clone(),
            )
        })
        .chain(trajectory.non_tr_data.iter().map(|data| {
            (
                data.block_id,
                false,
                data.stride_length,
                data.block_name.clone(),
            )
        }))
        .collect();
    for (block_id, particle_dependent, _metadata_stride, metadata_name) in non_tr_blocks {
        block_names
            .entry(block_id)
            .or_insert_with(|| canonical_block_name(block_id, &metadata_name));
        let (values, n_values, data_type) = if particle_dependent {
            let (values, _frames, _particles, n_values, data_type) =
                trajectory.particle_data_get(block_id)?;
            (values, n_values, data_type)
        } else {
            let (values, _frames, n_values, data_type) = trajectory.data_get(block_id)?;
            (values, n_values, data_type)
        };
        if data_type == DataType::Char {
            return Err(TngError::InvalidStructure(format!(
                "block {:?} contains character data, which is not supported",
                block_id
            )));
        }
        let values_per_sample = if particle_dependent {
            n_atoms
                .checked_mul(usize::try_from(n_values).map_err(|_| {
                    TngError::InvalidStructure("negative values-per-frame".to_owned())
                })?)
                .ok_or_else(|| {
                    TngError::InvalidStructure("block size overflows usize".to_owned())
                })?
        } else {
            usize::try_from(n_values)
                .map_err(|_| TngError::InvalidStructure("negative values-per-frame".to_owned()))?
        };
        if values_per_sample == 0 || values.len() % values_per_sample != 0 {
            return Err(TngError::InvalidStructure(format!(
                "block {:?} has {} values, not a multiple of {}",
                block_id,
                values.len(),
                values_per_sample
            )));
        }
        let name = block_names
            .get(&block_id)
            .cloned()
            .unwrap_or_else(|| format!("{block_id:?}"));
        let block_values: Vec<Vec<f64>> = values
            .chunks(values_per_sample)
            .map(ToOwned::to_owned)
            .collect();
        static_additional_names.insert(name.clone());
        additional_blocks.insert(name, block_values);
    }

    loop {
        if !first_set {
            if trajectory
                .current_trajectory_frame_set
                .next_frame_set_file_pos
                <= 0
            {
                break;
            }
            trajectory.frame_set_read_next(false)?;
        } else {
            trajectory.frame_set_read_next(false)?;
            first_set = false;
        }

        let frame_set_first = trajectory.current_trajectory_frame_set.first_frame;
        let frame_set_time = trajectory.current_trajectory_frame_set.first_frame_time;
        let blocks: Vec<(BlockID, bool, i64, i64, String)> = trajectory
            .current_trajectory_frame_set
            .tr_particle_data
            .iter()
            .map(|data| {
                (
                    data.block_id,
                    true,
                    data.stride_length,
                    data.first_frame_with_data,
                    data.block_name.clone(),
                )
            })
            .chain(
                trajectory
                    .current_trajectory_frame_set
                    .tr_data
                    .iter()
                    .map(|data| {
                        (
                            data.block_id,
                            false,
                            data.stride_length,
                            data.first_frame_with_data,
                            data.block_name.clone(),
                        )
                    }),
            )
            .collect();
        if blocks.is_empty() {
            return Err(TngError::InvalidStructure(
                "frame set contains no data blocks".to_owned(),
            ));
        }

        for (block_id, particle_dependent, metadata_stride, metadata_first, metadata_name) in blocks
        {
            let stride = metadata_stride.max(1);
            if SPECIAL_BLOCKS.contains(&block_id) {
                if let Some(expected) = global_stride {
                    if expected != stride {
                        return Err(TngError::InvalidStructure(format!(
                            "special block {:?} has stride {stride}; expected {expected}",
                            block_id
                        )));
                    }
                } else {
                    global_stride = Some(stride);
                }
            }
            block_names
                .entry(block_id)
                .or_insert_with(|| canonical_block_name(block_id, &metadata_name));
            let (values, n_values, data_type) = if particle_dependent {
                let (values, _frames, _particles, n_values, data_type) =
                    trajectory.particle_data_get(block_id)?;
                (values, n_values, data_type)
            } else {
                let (values, _frames, n_values, data_type) = trajectory.data_get(block_id)?;
                (values, n_values, data_type)
            };
            let entry = samples.entry(block_id).or_default();
            if entry
                .stride
                .replace(stride)
                .is_some_and(|old| old != stride)
            {
                return Err(TngError::InvalidStructure(format!(
                    "block {:?} changes stride within the trajectory",
                    block_id
                )));
            }
            if entry
                .n_values
                .replace(n_values)
                .is_some_and(|old| old != n_values)
            {
                return Err(TngError::InvalidStructure(format!(
                    "block {:?} changes values per frame",
                    block_id
                )));
            }
            if entry
                .data_type
                .replace(data_type)
                .is_some_and(|old| old != data_type)
            {
                return Err(TngError::InvalidStructure(format!(
                    "block {:?} changes data type",
                    block_id
                )));
            }
            if data_type == DataType::Char {
                return Err(TngError::InvalidStructure(format!(
                    "block {:?} contains character data, which is not supported",
                    block_id
                )));
            }
            let values_per_sample = if particle_dependent {
                n_atoms
                    .checked_mul(usize::try_from(n_values).map_err(|_| {
                        TngError::InvalidStructure("negative values-per-frame".to_owned())
                    })?)
                    .ok_or_else(|| {
                        TngError::InvalidStructure("block size overflows usize".to_owned())
                    })?
            } else {
                usize::try_from(n_values).map_err(|_| {
                    TngError::InvalidStructure("negative values-per-frame".to_owned())
                })?
            };
            if values_per_sample == 0 || values.len() % values_per_sample != 0 {
                return Err(TngError::InvalidStructure(format!(
                    "block {:?} has {} values, not a multiple of {}",
                    block_id,
                    values.len(),
                    values_per_sample
                )));
            }
            let first = metadata_first.max(frame_set_first);
            for (sample, chunk) in values.chunks(values_per_sample).enumerate() {
                entry.values.push(chunk.to_vec());
                entry
                    .steps
                    .push(first + i64::try_from(sample).unwrap_or(i64::MAX) * stride);
            }
            if block_id == BlockID::TrajPositions {
                let sample_count = values.len() / values_per_sample;
                for sample in 0..sample_count {
                    let step = first + i64::try_from(sample).unwrap_or(i64::MAX) * stride;
                    position_times.push(
                        frame_set_time + time_per_frame.max(0.0) * (step - frame_set_first) as f64,
                    );
                }
            }
        }
    }

    let global_stride = usize::try_from(global_stride.ok_or_else(|| {
        TngError::InvalidStructure("trajectory has no positions or box block".to_owned())
    })?)
    .map_err(|_| TngError::InvalidStructure("invalid special-block stride".to_owned()))?;
    let positions = samples.remove(&BlockID::TrajPositions).ok_or_else(|| {
        TngError::InvalidStructure("trajectory is missing the positions block".to_owned())
    })?;
    let frame_count = positions.values.len();
    if frame_count == 0 {
        return Err(TngError::InvalidStructure(
            "positions block contains no frames".to_owned(),
        ));
    }
    for name in static_additional_names {
        if let Some(values) = additional_blocks.get_mut(&name)
            && let Some(value) = values.first().cloned()
        {
            values.resize(frame_count, value);
        }
    }
    let mut boxes = samples.remove(&BlockID::TrajBoxShape);
    if let Some(values) = &mut boxes {
        trim_special(values, frame_count, "box shape")?;
    }
    let mut velocities = samples.remove(&BlockID::TrajVelocities);
    let mut forces = samples.remove(&BlockID::TrajForces);
    if let Some(values) = &mut velocities {
        trim_special(values, frame_count, "velocities")?;
    }
    if let Some(values) = &mut forces {
        trim_special(values, frame_count, "forces")?;
    }

    let mut frames = Vec::with_capacity(frame_count);
    let mut force_frames = Vec::with_capacity(frame_count);
    for index in 0usize..frame_count {
        let position_values = &positions.values[index];
        if position_values.len() != n_atoms * 3 {
            return Err(TngError::InvalidStructure(
                "positions block is not 3D".to_owned(),
            ));
        }
        let mut frame = CoordinateFrame::new(triples_scaled(
            position_values,
            length_factor,
            n_atoms,
            "positions",
        )?);
        frame.velocities = velocities
            .as_ref()
            .map(|values| {
                triples_scaled(&values.values[index], length_factor, n_atoms, "velocities")
            })
            .transpose()?;
        force_frames.push(
            forces
                .as_ref()
                .map(|values| {
                    triples_scaled(
                        &values.values[index],
                        if convert_units {
                            1.0 / length_factor
                        } else {
                            1.0
                        },
                        n_atoms,
                        "forces",
                    )
                })
                .transpose()?,
        );
        frame.dimensions = boxes
            .as_ref()
            .map(|values| box_dimensions(&values.values[index], length_factor))
            .transpose()?;
        let step = index.checked_mul(global_stride).ok_or_else(|| {
            TngError::InvalidStructure("integrator step overflows usize".to_owned())
        })?;
        frame.step = step;
        frame.time = position_times.get(index).copied().unwrap_or(0.0)
            * if convert_units { 1.0e12 } else { 1.0 };
        frames.push(frame);
    }

    for (block_id, mut block) in samples {
        let Some(stride) = block.stride else {
            continue;
        };
        if stride <= 0 || stride % global_stride as i64 != 0 {
            continue;
        }
        let name = block_names
            .get(&block_id)
            .cloned()
            .unwrap_or_else(|| format!("{block_id:?}"));
        let mut per_frame = vec![Vec::new(); frame_count];
        for (step, values) in block.steps.into_iter().zip(block.values.drain(..)) {
            if step >= 0 && step % global_stride as i64 == 0 {
                let index = usize::try_from(step / global_stride as i64).unwrap_or(usize::MAX);
                if index < frame_count {
                    per_frame[index] = values;
                }
            }
        }
        if per_frame.iter().any(|values| !values.is_empty()) {
            additional_blocks.insert(name, per_frame);
        }
    }
    let special_blocks = SPECIAL_BLOCKS
        .iter()
        .filter_map(|block_id| block_names.get(block_id).cloned())
        .collect();
    let blocks = block_names.into_values().collect();
    Ok(TngFile {
        n_atoms,
        coordinates: CoordinateFile::new(frames),
        forces: force_frames,
        additional_blocks,
        blocks,
        special_blocks,
        global_stride,
    })
}

fn canonical_block_name(block_id: BlockID, source_name: &str) -> String {
    match block_id {
        BlockID::TrajPositions => "TNG_TRAJ_POSITIONS".to_owned(),
        BlockID::TrajBoxShape => "TNG_TRAJ_BOX_SHAPE".to_owned(),
        BlockID::TrajVelocities => "TNG_TRAJ_VELOCITIES".to_owned(),
        BlockID::TrajForces => "TNG_TRAJ_FORCES".to_owned(),
        BlockID::GmxLambda => "TNG_GMX_LAMBDA".to_owned(),
        _ => source_name.to_owned(),
    }
}

fn trim_special(candidate: &mut BlockSamples, expected: usize, name: &str) -> Result<(), TngError> {
    if candidate.values.len() < expected {
        return Err(TngError::InvalidStructure(format!(
            "special block {name} has fewer data frames than positions"
        )));
    }
    candidate.values.truncate(expected);
    candidate.steps.truncate(expected);
    Ok(())
}

fn triples_scaled(
    values: &[f64],
    factor: f64,
    n_atoms: usize,
    name: &str,
) -> Result<Vec<[f64; 3]>, TngError> {
    if values.len() != n_atoms * 3 {
        return Err(TngError::InvalidStructure(format!(
            "{name} block is not 3D"
        )));
    }
    Ok(values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|chunk| [chunk[0] * factor, chunk[1] * factor, chunk[2] * factor])
        .collect())
}

fn box_dimensions(values: &[f64], length_factor: f64) -> Result<[f64; 6], TngError> {
    if values.len() != 9 {
        return Err(TngError::InvalidStructure(format!(
            "box shape contains {} values; expected 9",
            values.len()
        )));
    }
    Ok(triclinic_box([
        [
            values[0] * length_factor,
            values[1] * length_factor,
            values[2] * length_factor,
        ],
        [
            values[3] * length_factor,
            values[4] * length_factor,
            values[5] * length_factor,
        ],
        [
            values[6] * length_factor,
            values[7] * length_factor,
            values[8] * length_factor,
        ],
    ]))
}

fn universe_from_tng(file: TngFile) -> crate::Result<Universe> {
    let first = file
        .coordinates
        .frames
        .first()
        .ok_or_else(|| crate::Error::InvalidInput("TNG trajectory has no frames".to_owned()))?;
    let atoms = first
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| Atom::new(index, "X", *position))
        .collect();
    let mut universe = Universe {
        topology: Topology::new(atoms),
        trajectory: Trajectory::default(),
    };
    attach_tng(&mut universe, file)?;
    Ok(universe)
}

fn attach_tng(universe: &mut Universe, file: TngFile) -> crate::Result<()> {
    if file.n_atoms != universe.n_atoms() {
        return Err(crate::Error::InvalidInput(format!(
            "TNG contains {} atoms, topology contains {}",
            file.n_atoms,
            universe.n_atoms()
        )));
    }
    if file.coordinates.frames.is_empty() {
        return Err(crate::Error::InvalidInput(
            "TNG trajectory has no frames".to_owned(),
        ));
    }
    let forces = file.forces;
    let additional_blocks = file.additional_blocks;
    universe.trajectory = Trajectory::new(
        file.coordinates
            .frames
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let mut frame = Frame::new(source.positions);
                frame.velocities = source.velocities;
                frame.dimensions = source.dimensions;
                frame.step = source.step;
                frame.time = source.time;
                frame.forces = forces.get(index).cloned().flatten();
                for (name, values) in &additional_blocks {
                    if let Some(data) = values.get(index).filter(|data| !data.is_empty()) {
                        frame.data.insert(name.clone(), data.clone());
                    }
                }
                frame
            })
            .collect(),
    );
    Ok(())
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
    fn reads_compressed_tng_positions_and_box() {
        let file = read_tng(fixture("argon_npt_compressed.tng")).unwrap();
        assert_eq!(file.n_atoms(), 1000);
        assert_eq!(file.n_frames(), 101);
        assert_eq!(file.global_stride, 5000);
        assert!(
            file.special_blocks
                .iter()
                .any(|name| name == "TNG_TRAJ_POSITIONS")
        );
        assert!(
            file.to_universe()
                .unwrap()
                .current_frame()
                .unwrap()
                .data
                .contains_key("TNG_GMX_LAMBDA")
        );
        assert!((file.coordinates.frames[0].positions[0][0] - 25.3299975).abs() < 1e-4);
        assert!((file.coordinates.frames[100].positions[0][0] - 4.4).abs() < 1e-4);
        assert_eq!(file.coordinates.frames[100].step, 500_000);
        assert!((file.coordinates.frames[0].dimensions.unwrap()[0] - 36.014).abs() < 1e-4);
        assert!(file.blocks.iter().any(|name| name == "TNG_GMX_LAMBDA"));
        assert!(file.additional_blocks.contains_key("TNG_GMX_LAMBDA"));
        assert!(file.blocks.iter().any(|name| name == "ATOM MASSES"));
        assert!(file.blocks.iter().any(|name| name == "PARTIAL CHARGES"));
        assert_eq!(file.additional_blocks["ATOM MASSES"].len(), file.n_frames());
        let universe = file.to_universe().unwrap();
        assert!(
            universe.trajectory.frames[100]
                .data
                .contains_key("ATOM MASSES")
        );
    }

    #[test]
    fn converts_synthetic_units_and_timing() {
        let file = read_tng(fixture("coordinates/test.tng")).unwrap();
        let first = &file.coordinates.frames[0];
        for (actual, expected) in first.positions[0].into_iter().zip([0.0, 1.0, 2.0]) {
            assert!((actual - expected).abs() < 1e-4);
        }
        for (actual, expected) in first.velocities.as_ref().unwrap()[0]
            .into_iter()
            .zip([0.0, 0.1, 0.2])
        {
            assert!((actual - expected).abs() < 1e-4);
        }
        assert!((file.forces[0].as_ref().unwrap()[0][1] - 0.01).abs() < 1e-6);
        assert_eq!(first.step, 0);
        assert!((file.coordinates.frames[1].time - 1.0).abs() < 1e-12);
        let dimensions = first.dimensions.unwrap();
        assert!((dimensions[0] - 81.1).abs() < 1e-4);
        assert!((dimensions[3] - 75.0).abs() < 1e-4);
    }

    #[test]
    fn can_retain_native_units() {
        let native = TngFile::read_file_unconverted(fixture("coordinates/test.tng")).unwrap();
        assert!((native.coordinates.frames[0].positions[0][1] - 0.1).abs() < 1e-6);
        assert!(
            (native.coordinates.frames[0].velocities.as_ref().unwrap()[0][1] - 0.01).abs() < 1e-6
        );
        assert!((native.forces[0].as_ref().unwrap()[0][1] - 0.1).abs() < 1e-6);
        assert!((native.coordinates.frames[1].time - 1.0e-12).abs() < 1e-24);
    }

    #[test]
    fn reads_velocities_and_forces() {
        let file = read_tng(fixture("argon_npt_compressed_vels_forces.tng")).unwrap();
        let frame = &file.coordinates.frames[0];
        assert_eq!(file.n_frames(), 51);
        assert!(frame.velocities.is_some());
        assert!(
            file.to_universe().unwrap().trajectory.frames[0]
                .forces
                .is_some()
        );
    }

    #[test]
    fn bytes_and_gro_constructors_attach_trajectory() {
        let tng_bytes = std::fs::read(fixture("coordinates/test.tng")).unwrap();
        let from_bytes = TngFile::from_bytes(&tng_bytes).unwrap();
        let from_path = read_tng(fixture("coordinates/test.tng")).unwrap();
        assert_eq!(from_bytes, from_path);
        let universe = Universe::from_gro_and_tng(
            fixture("argon_npt_compressed.gro.gz"),
            fixture("argon_npt_compressed.tng"),
        )
        .unwrap();
        assert_eq!(universe.n_atoms(), 1000);
        assert_eq!(universe.trajectory.n_frames(), 101);
    }

    #[test]
    fn rejects_uneven_special_blocks() {
        let error = read_tng(fixture("argon_npt_compressed_uneven.tng")).unwrap_err();
        assert!(error.to_string().contains("special block"));
    }
}
