//! H5MD trajectory support backed by the pure-Rust `hdf5-pure` reader.
//!
//! The reader consumes the standard `particles/<group>` hierarchy (the
//! conventional group name is `trajectory`) and converts recognized H5MD units
//! into the crate's Angstrom, picosecond, and kJ/(mol A) base units.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use crate::units::{Unit, UnitKind};
use hdf5_pure::{AttrValue, File};
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::str::FromStr;

/// Parsed H5MD trajectory data.
#[derive(Clone, Debug, PartialEq)]
pub struct H5mdFile {
    pub n_atoms: usize,
    pub coordinates: CoordinateFile,
    pub forces: Vec<Option<Vec<[f64; 3]>>>,
}

pub type H5mdData = H5mdFile;
pub type H5mdStructure = H5mdFile;

/// Errors produced while reading H5MD files.
#[derive(Debug)]
pub enum H5mdError {
    Io(io::Error),
    Hdf5(String),
    InvalidStructure(String),
    UnknownUnit { kind: UnitKind, unit: String },
}

impl fmt::Display for H5mdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "H5MD I/O error: {error}"),
            Self::Hdf5(error) => write!(formatter, "H5MD HDF5 error: {error}"),
            Self::InvalidStructure(error) => write!(formatter, "invalid H5MD structure: {error}"),
            Self::UnknownUnit { kind, unit } => {
                write!(formatter, "unrecognized H5MD {kind} unit {unit:?}")
            }
        }
    }
}

impl std::error::Error for H5mdError {}

impl From<io::Error> for H5mdError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl H5mdFile {
    /// Read an H5MD file and convert recognized units to base units.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, H5mdError> {
        Self::read_file_with_options(path, true)
    }

    /// Alias for [`H5mdFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, H5mdError> {
        Self::read_file(path)
    }

    /// Read an H5MD document from any reader with unit conversion enabled.
    pub fn read<R: io::Read>(mut reader: R) -> Result<Self, H5mdError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Read an H5MD file while retaining its native numeric units.
    pub fn read_file_unconverted(path: impl AsRef<Path>) -> Result<Self, H5mdError> {
        Self::read_file_with_options(path, false)
    }

    /// Read an H5MD file with explicit unit-conversion behavior.
    pub fn read_file_with_options(
        path: impl AsRef<Path>,
        convert_units: bool,
    ) -> Result<Self, H5mdError> {
        let bytes = fs::read(path)?;
        Self::from_bytes_with_options(&bytes, convert_units)
    }

    /// Parse H5MD bytes with unit conversion enabled.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, H5mdError> {
        Self::from_bytes_with_options(bytes, true)
    }

    /// Parse H5MD bytes while retaining native units.
    pub fn from_bytes_unconverted(bytes: &[u8]) -> Result<Self, H5mdError> {
        Self::from_bytes_with_options(bytes, false)
    }

    /// Parse H5MD bytes with explicit unit-conversion behavior.
    pub fn from_bytes_with_options(bytes: &[u8], convert_units: bool) -> Result<Self, H5mdError> {
        let file =
            File::from_bytes(bytes.to_vec()).map_err(|error| H5mdError::Hdf5(error.to_string()))?;
        parse_file(&file, convert_units)
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
        universe_from_h5md(self.clone())
    }
}

/// Read an H5MD trajectory from a path.
pub fn read_h5md(path: impl AsRef<Path>) -> Result<H5mdFile, H5mdError> {
    H5mdFile::read_file(path)
}

impl CoordinateFile {
    pub fn read_h5md<R: io::Read>(mut reader: R) -> Result<Self, H5mdError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(H5mdFile::from_bytes(&bytes)?.coordinates)
    }

    pub fn from_h5md_bytes(bytes: &[u8]) -> Result<Self, H5mdError> {
        Ok(H5mdFile::from_bytes(bytes)?.coordinates)
    }
}

impl Universe {
    pub fn from_h5md(path: impl AsRef<Path>) -> crate::Result<Self> {
        H5mdFile::read_file(path)?.to_universe()
    }

    pub fn from_h5md_bytes(bytes: &[u8]) -> crate::Result<Self> {
        H5mdFile::from_bytes(bytes)?.to_universe()
    }

    pub fn from_h5md_file(file: H5mdFile) -> crate::Result<Self> {
        file.to_universe()
    }

    pub fn from_psf_and_h5md(
        psf_path: impl AsRef<Path>,
        h5md_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf(psf_path)?;
        attach_h5md(&mut universe, H5mdFile::read_file(h5md_path)?)?;
        Ok(universe)
    }

    pub fn from_psf_and_h5md_bytes(psf: &str, h5md: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_psf_str(psf)?;
        attach_h5md(&mut universe, H5mdFile::from_bytes(h5md)?)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_h5md(
        topology_path: impl AsRef<Path>,
        h5md_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop(topology_path)?;
        attach_h5md(&mut universe, H5mdFile::read_file(h5md_path)?)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_h5md_bytes(topology: &str, h5md: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop_str(topology)?;
        attach_h5md(&mut universe, H5mdFile::from_bytes(h5md)?)?;
        Ok(universe)
    }
}

fn parse_file(file: &File, convert_units: bool) -> Result<H5mdFile, H5mdError> {
    let particles = file.group("particles").map_err(|error| {
        H5mdError::InvalidStructure(format!("missing particles group: {error}"))
    })?;
    let group_name = particles
        .groups()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?
        .into_iter()
        .find(|name| {
            file.dataset(&format!("particles/{name}/position/value"))
                .is_ok()
        })
        .ok_or_else(|| {
            H5mdError::InvalidStructure(
                "particles contains no group with position/value data".to_owned(),
            )
        })?;
    let base = format!("particles/{group_name}");
    let position_path = format!("{base}/position/value");
    let position = read_payload(file, &position_path, UnitKind::Length, convert_units)?;
    let (n_frames, n_atoms) = payload_shape(&position.shape, "position")?;
    let expected = n_frames
        .checked_mul(n_atoms)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| {
            H5mdError::InvalidStructure("position dimensions overflow usize".to_owned())
        })?;
    if position.values.len() != expected {
        return Err(H5mdError::InvalidStructure(format!(
            "position contains {} values; expected {expected}",
            position.values.len()
        )));
    }
    let velocity = optional_payload(
        file,
        &format!("{base}/velocity/value"),
        UnitKind::Velocity,
        convert_units,
        &position.shape,
        expected,
    )?;
    let force = optional_payload(
        file,
        &format!("{base}/force/value"),
        UnitKind::Force,
        convert_units,
        &position.shape,
        expected,
    )?;
    let times = first_scalar_series(file, &base, "time", UnitKind::Time, convert_units, n_frames)?
        .unwrap_or_else(|| (0..n_frames).map(|index| index as f64).collect());
    let steps = first_integer_series(file, &base, "step", n_frames)?
        .unwrap_or_else(|| (0..n_frames).collect());
    let dimensions = read_box(file, &base, n_frames, convert_units)?;

    let mut frames = Vec::with_capacity(n_frames);
    let mut forces = Vec::with_capacity(n_frames);
    for frame_index in 0..n_frames {
        let start = frame_index * n_atoms * 3;
        let mut frame = CoordinateFrame::new(triples(&position.values[start..start + n_atoms * 3]));
        frame.velocities = velocity
            .as_ref()
            .map(|values| triples(&values[start..start + n_atoms * 3]));
        frame.dimensions = dimensions
            .as_ref()
            .and_then(|values| values.get(frame_index).copied());
        frame.step = steps[frame_index];
        frame.time = times[frame_index];
        frames.push(frame);
        forces.push(
            force
                .as_ref()
                .map(|values| triples(&values[start..start + n_atoms * 3])),
        );
    }
    Ok(H5mdFile {
        n_atoms,
        coordinates: CoordinateFile::new(frames),
        forces,
    })
}

struct Payload {
    values: Vec<f64>,
    shape: Vec<u64>,
}

fn read_payload(
    file: &File,
    path: &str,
    kind: UnitKind,
    convert_units: bool,
) -> Result<Payload, H5mdError> {
    let dataset = file.dataset(path).map_err(|error| {
        H5mdError::InvalidStructure(format!("missing dataset {path:?}: {error}"))
    })?;
    let shape = dataset
        .shape()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    let values = dataset
        .read_f64()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    let factor = unit_factor(
        &dataset
            .attrs()
            .map_err(|error| H5mdError::Hdf5(error.to_string()))?,
        kind,
        convert_units,
    )?;
    Ok(Payload {
        values: values.into_iter().map(|value| value * factor).collect(),
        shape,
    })
}

fn optional_payload(
    file: &File,
    path: &str,
    kind: UnitKind,
    convert_units: bool,
    expected_shape: &[u64],
    expected_len: usize,
) -> Result<Option<Vec<f64>>, H5mdError> {
    let Ok(dataset) = file.dataset(path) else {
        return Ok(None);
    };
    let shape = dataset
        .shape()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    if shape != expected_shape {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} shape {shape:?} does not match position shape {expected_shape:?}"
        )));
    }
    let factor = unit_factor(
        &dataset
            .attrs()
            .map_err(|error| H5mdError::Hdf5(error.to_string()))?,
        kind,
        convert_units,
    )?;
    let values = dataset
        .read_f64()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?
        .into_iter()
        .map(|value| value * factor)
        .collect::<Vec<_>>();
    if values.len() != expected_len {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} contains {} values; expected {expected_len}",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn optional_scalar_series(
    file: &File,
    path: &str,
    kind: UnitKind,
    convert_units: bool,
    n_frames: usize,
) -> Result<Option<Vec<f64>>, H5mdError> {
    let Ok(dataset) = file.dataset(path) else {
        return Ok(None);
    };
    let shape = dataset
        .shape()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    if shape != [n_frames as u64] {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} shape {shape:?}; expected [{n_frames}]"
        )));
    }
    let values = dataset
        .read_f64()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    if values.len() != n_frames {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} shape {shape:?} contains {}; expected {n_frames}",
            values.len()
        )));
    }
    let factor = unit_factor(
        &dataset
            .attrs()
            .map_err(|error| H5mdError::Hdf5(error.to_string()))?,
        kind,
        convert_units,
    )?;
    Ok(Some(
        values.into_iter().map(|value| value * factor).collect(),
    ))
}

fn first_scalar_series(
    file: &File,
    base: &str,
    name: &str,
    kind: UnitKind,
    convert_units: bool,
    n_frames: usize,
) -> Result<Option<Vec<f64>>, H5mdError> {
    for group in ["position", "velocity", "force"] {
        let path = format!("{base}/{group}/{name}");
        if file.dataset(&path).is_ok() {
            return optional_scalar_series(file, &path, kind, convert_units, n_frames);
        }
    }
    Ok(None)
}

fn optional_integer_series(
    file: &File,
    path: &str,
    n_frames: usize,
) -> Result<Option<Vec<usize>>, H5mdError> {
    let Ok(dataset) = file.dataset(path) else {
        return Ok(None);
    };
    let shape = dataset
        .shape()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    if shape != [n_frames as u64] {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} shape {shape:?}; expected [{n_frames}]"
        )));
    }
    let values = dataset
        .read_i64()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    if values.len() != n_frames {
        return Err(H5mdError::InvalidStructure(format!(
            "{path:?} contains {}; expected {n_frames}",
            values.len()
        )));
    }
    values
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                H5mdError::InvalidStructure(format!("{path:?} contains invalid step {value}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn first_integer_series(
    file: &File,
    base: &str,
    name: &str,
    n_frames: usize,
) -> Result<Option<Vec<usize>>, H5mdError> {
    for group in ["position", "velocity", "force"] {
        let path = format!("{base}/{group}/{name}");
        if file.dataset(&path).is_ok() {
            return optional_integer_series(file, &path, n_frames);
        }
    }
    Ok(None)
}

fn read_box(
    file: &File,
    base: &str,
    n_frames: usize,
    convert_units: bool,
) -> Result<Option<Vec<[f64; 6]>>, H5mdError> {
    let box_path = format!("{base}/box");
    let Ok(box_group) = file.group(&box_path) else {
        return Ok(None);
    };
    let box_attrs = box_group
        .attrs()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    let dimension = box_attrs
        .get("dimension")
        .and_then(AttrValue::as_i64)
        .or_else(|| {
            box_attrs
                .get("dimension")
                .and_then(AttrValue::as_u64)
                .and_then(|v| i64::try_from(v).ok())
        })
        .ok_or_else(|| {
            H5mdError::InvalidStructure(format!("{box_path:?} is missing integer dimension"))
        })?;
    if dimension != 3 {
        return Err(H5mdError::InvalidStructure(format!(
            "H5MD box dimension {dimension}; only 3-dimensional boxes are supported"
        )));
    }
    let path = format!("{base}/box/edges/value");
    let Ok(dataset) = file.dataset(&path) else {
        return Ok(None);
    };
    let shape = dataset
        .shape()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    let values = dataset
        .read_f64()
        .map_err(|error| H5mdError::Hdf5(error.to_string()))?;
    let factor = unit_factor(
        &dataset
            .attrs()
            .map_err(|error| H5mdError::Hdf5(error.to_string()))?,
        UnitKind::Length,
        convert_units,
    )?;
    let values = values
        .into_iter()
        .map(|value| value * factor)
        .collect::<Vec<_>>();
    let frames = match shape.as_slice() {
        [frames, 3, 3] => {
            let frames = usize::try_from(*frames).map_err(|_| {
                H5mdError::InvalidStructure("box frame count overflows usize".to_owned())
            })?;
            let expected_values = frames.checked_mul(9).ok_or_else(|| {
                H5mdError::InvalidStructure("box dimensions overflow usize".to_owned())
            })?;
            if frames != n_frames || values.len() != expected_values {
                return Err(H5mdError::InvalidStructure(
                    "box edge frame count does not match positions".to_owned(),
                ));
            }
            (0..frames)
                .map(|index| {
                    let start = index * 9;
                    triclinic_box([
                        [values[start], values[start + 1], values[start + 2]],
                        [values[start + 3], values[start + 4], values[start + 5]],
                        [values[start + 6], values[start + 7], values[start + 8]],
                    ])
                })
                .collect()
        }
        [frames, 3] if *frames == n_frames as u64 => {
            let frames = usize::try_from(*frames).map_err(|_| {
                H5mdError::InvalidStructure("box frame count overflows usize".to_owned())
            })?;
            let expected_values = frames.checked_mul(3).ok_or_else(|| {
                H5mdError::InvalidStructure("box dimensions overflow usize".to_owned())
            })?;
            if values.len() != expected_values {
                return Err(H5mdError::InvalidStructure(
                    "cuboid box frame count does not match positions".to_owned(),
                ));
            }
            (0..frames)
                .map(|index| {
                    [
                        values[index * 3],
                        values[index * 3 + 1],
                        values[index * 3 + 2],
                        90.0,
                        90.0,
                        90.0,
                    ]
                })
                .collect()
        }
        [3, 3] => {
            if values.len() != 9 {
                return Err(H5mdError::InvalidStructure(
                    "box edge matrix must contain 9 values".to_owned(),
                ));
            }
            let dimensions = triclinic_box([
                [values[0], values[1], values[2]],
                [values[3], values[4], values[5]],
                [values[6], values[7], values[8]],
            ]);
            vec![dimensions; n_frames]
        }
        [3] => {
            if values.len() != 3 {
                return Err(H5mdError::InvalidStructure(
                    "cuboid box must contain 3 values".to_owned(),
                ));
            }
            vec![[values[0], values[1], values[2], 90.0, 90.0, 90.0]; n_frames]
        }
        _ => {
            return Err(H5mdError::InvalidStructure(format!(
                "unsupported box edge shape {shape:?}"
            )));
        }
    };
    Ok(Some(frames))
}

fn payload_shape(shape: &[u64], name: &str) -> Result<(usize, usize), H5mdError> {
    let result = match shape {
        [frames, atoms, 3] => (
            usize::try_from(*frames).map_err(|_| {
                H5mdError::InvalidStructure(format!("{name} frame count overflows usize"))
            })?,
            usize::try_from(*atoms).map_err(|_| {
                H5mdError::InvalidStructure(format!("{name} atom count overflows usize"))
            })?,
        ),
        [atoms, 3] => (
            1,
            usize::try_from(*atoms).map_err(|_| {
                H5mdError::InvalidStructure(format!("{name} atom count overflows usize"))
            })?,
        ),
        _ => {
            return Err(H5mdError::InvalidStructure(format!(
                "{name} values must have shape (frame, atom, 3) or (atom, 3), got {shape:?}"
            )));
        }
    };
    if result.0 == 0 || result.1 == 0 {
        return Err(H5mdError::InvalidStructure(format!(
            "{name} must contain at least one frame and atom"
        )));
    }
    Ok(result)
}

fn unit_factor(
    attrs: &std::collections::HashMap<String, AttrValue>,
    kind: UnitKind,
    convert_units: bool,
) -> Result<f64, H5mdError> {
    if !convert_units {
        return Ok(1.0);
    }
    let unit = attrs
        .get("unit")
        .and_then(AttrValue::as_str)
        .ok_or_else(|| H5mdError::UnknownUnit {
            kind,
            unit: "<missing>".to_owned(),
        })?;
    let translated = match (kind, unit) {
        (UnitKind::Time, "second" | "sec") => "s",
        (UnitKind::Velocity, "Angstrom ps-1") => "A/ps",
        (UnitKind::Velocity, "A ps-1") => "A/ps",
        (UnitKind::Velocity, "Angstrom fs-1") => "A/fs",
        (UnitKind::Velocity, "A fs-1") => "A/fs",
        (UnitKind::Velocity, "Angstrom AKMA-1") => "A/AKMA",
        (UnitKind::Velocity, "A AKMA-1") => "A/AKMA",
        (UnitKind::Velocity, "nm ps-1") => "nm/ps",
        (UnitKind::Velocity, "nm ns-1") => "nm/ns",
        (UnitKind::Velocity, "pm ps-1") => "pm/ps",
        (UnitKind::Velocity, "m s-1") => "m/s",
        (UnitKind::Force, "kJ mol-1 Angstrom-1" | "kcal mol-1 Angstrom-1" | "kcal mol-1 A-1") => {
            if unit.starts_with("kcal") {
                "kcal/(mol*Angstrom)"
            } else {
                "kJ/(mol*Angstrom)"
            }
        }
        (UnitKind::Force, "kJ mol-1 nm-1") => "kJ/(mol*nm)",
        (UnitKind::Force, "Newton") => "N",
        (UnitKind::Force, "J m-1") => "J/m",
        _ => unit,
    };
    let parsed = Unit::from_str(translated).map_err(|_| H5mdError::UnknownUnit {
        kind,
        unit: unit.to_owned(),
    })?;
    if parsed.kind() != kind {
        return Err(H5mdError::UnknownUnit {
            kind,
            unit: unit.to_owned(),
        });
    }
    Ok(parsed.factor_to_base())
}

fn triples(values: &[f64]) -> Vec<[f64; 3]> {
    values.as_chunks::<3>().0.to_vec()
}

fn universe_from_h5md(file: H5mdFile) -> crate::Result<Universe> {
    let first =
        file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("H5MD trajectory has no frames".to_owned())
        })?;
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
    attach_h5md(&mut universe, file)?;
    Ok(universe)
}

fn attach_h5md(universe: &mut Universe, file: H5mdFile) -> crate::Result<()> {
    if file.n_atoms != universe.n_atoms() {
        return Err(crate::Error::InvalidInput(format!(
            "H5MD contains {} atoms, topology contains {}",
            file.n_atoms,
            universe.n_atoms()
        )));
    }
    let frames = file
        .coordinates
        .frames
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            let mut frame = Frame::new(source.positions);
            frame.velocities = source.velocities;
            frame.dimensions = source.dimensions;
            frame.step = source.step;
            frame.time = source.time;
            frame.forces = file.forces.get(index).cloned().flatten();
            frame
        })
        .collect();
    universe.trajectory = Trajectory::new(frames);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_pure::{AttrValue, FileBuilder};
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data")
            .join(name)
    }

    #[test]
    fn reads_h5md_trajectory() {
        let file = read_h5md(fixture("coordinates/test.h5md")).unwrap();
        assert_eq!(file.n_atoms(), 5);
        assert_eq!(file.n_frames(), 5);
        assert_eq!(file.coordinates.frames[0].positions[0], [0.0, 1.0, 2.0]);
        let velocity = file.coordinates.frames[0].velocities.as_ref().unwrap()[0];
        assert!((velocity[0] - 0.0).abs() < 1e-6);
        assert!((velocity[1] - 0.1).abs() < 1e-6);
        assert!((velocity[2] - 0.2).abs() < 1e-6);
        assert!(file.forces[0].is_some());
        assert_eq!(file.coordinates.frames[0].step, 0);
        assert_eq!(file.coordinates.frames[4].time, 4.0);
        assert!((file.coordinates.frames[0].dimensions.unwrap()[0] - 81.1).abs() < 1e-5);
        let force = file.forces[0].as_ref().unwrap()[0];
        assert!((force[1] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn bytes_and_universe_constructors_match() {
        let bytes = fs::read(fixture("coordinates/test.h5md")).unwrap();
        let file = H5mdFile::from_bytes(&bytes).unwrap();
        let coordinates = CoordinateFile::from_h5md_bytes(&bytes).unwrap();
        assert_eq!(coordinates, file.coordinates);
        let universe = Universe::from_h5md_bytes(&bytes).unwrap();
        assert_eq!(universe.n_atoms(), 5);
        assert!(universe.current_frame().unwrap().forces.is_some());
    }

    #[test]
    fn converts_units_and_supports_native_values() {
        let bytes = fs::read(fixture("coordinates/test.h5md")).unwrap();
        let converted = H5mdFile::from_bytes(&bytes).unwrap();
        let native = H5mdFile::from_bytes_unconverted(&bytes).unwrap();
        assert_eq!(native.coordinates.frames[0].positions[0], [0.0, 1.0, 2.0]);
        assert_eq!(
            converted.coordinates.frames[0].positions[0],
            [0.0, 1.0, 2.0]
        );
        assert!(
            (native.coordinates.frames[0].velocities.as_ref().unwrap()[0][1] - 0.1).abs() < 1e-6
        );
        assert!(
            (converted.coordinates.frames[0].velocities.as_ref().unwrap()[0][1] - 0.1).abs() < 1e-6
        );
    }

    #[test]
    fn missing_unit_is_rejected_when_conversion_is_enabled() {
        let attrs = std::collections::HashMap::new();
        assert!(matches!(
            unit_factor(&attrs, UnitKind::Length, true),
            Err(H5mdError::UnknownUnit {
                kind: UnitKind::Length,
                ..
            })
        ));
        assert_eq!(unit_factor(&attrs, UnitKind::Length, false).unwrap(), 1.0);
    }

    #[test]
    fn reads_cobrotoxin_fixture() {
        let file = read_h5md(fixture("cobrotoxin.h5md")).unwrap();
        assert_eq!(file.n_atoms(), 19_385);
        assert_eq!(file.n_frames(), 3);
        let first = &file.coordinates.frames[0];
        assert!((first.positions[0][0] - 32.309906).abs() < 1e-5);
        assert!((first.velocities.as_ref().unwrap()[0][0] + 2.697732).abs() < 1e-5);
        assert!((file.forces[0].as_ref().unwrap()[0][0] - 20.071287).abs() < 1e-5);
        let dimensions = first.dimensions.unwrap();
        assert!((dimensions[0] - 52.763).abs() < 1e-4);
        assert_eq!(first.step, 0);
        assert_eq!(file.coordinates.frames[2].step, 50_000);
    }

    #[test]
    fn reads_per_frame_cuboid_edges() {
        let mut builder = FileBuilder::new();
        let mut particles = builder.create_group("particles");
        let mut trajectory = particles.create_group("trajectory");

        let mut position = trajectory.create_group("position");
        position
            .create_dataset("value")
            .with_f64_data(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
            .with_shape(&[3, 1, 3])
            .set_attr("unit", AttrValue::VarLenString("Angstrom".to_owned()));
        trajectory.add_group(position.finish());

        let mut box_group = trajectory.create_group("box");
        box_group.set_attr("dimension", AttrValue::I32(3));
        let mut edges = box_group.create_group("edges");
        edges
            .create_dataset("value")
            .with_f64_data(&[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0])
            .with_shape(&[3, 3])
            .set_attr("unit", AttrValue::VarLenString("Angstrom".to_owned()));
        box_group.add_group(edges.finish());
        trajectory.add_group(box_group.finish());

        particles.add_group(trajectory.finish());
        builder.add_group(particles.finish());
        let bytes = builder.finish().unwrap();
        let file = H5mdFile::from_bytes(&bytes).unwrap();
        assert_eq!(file.n_frames(), 3);
        assert_eq!(
            file.coordinates.frames[0].dimensions.unwrap()[..3],
            [10.0, 11.0, 12.0]
        );
        assert_eq!(
            file.coordinates.frames[2].dimensions.unwrap()[..3],
            [30.0, 31.0, 32.0]
        );
    }
}
