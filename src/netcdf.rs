//! Amber NetCDF-3 trajectory support.
//!
//! Amber's NetCDF trajectory convention stores coordinates in Angstroms and
//! uses an unlimited `frame` dimension.  This reader accepts both NetCDF-3
//! classic and 64-bit-offset files through the pure-Rust `netcdf3` crate.
//! Force values are retained in their source units (normally
//! kcal/(mol Angstrom)); variable `scale_factor` attributes are applied to
//! all numeric payloads before they are exposed.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Frame, Topology, Trajectory, Universe};
use flate2::read::GzDecoder;
use netcdf3::{DataVector, FileReader};
use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// Parsed Amber NetCDF trajectory data.
#[derive(Clone, Debug, PartialEq)]
pub struct NetcdfFile {
    /// Global title, when present.
    pub title: String,
    /// Global `Conventions` attribute.
    pub conventions: String,
    /// Global `ConventionVersion` attribute.
    pub convention_version: String,
    /// Number of atoms in each frame.
    pub n_atoms: usize,
    /// Coordinate frames, including positions, velocities, dimensions, time,
    /// and step metadata.
    pub coordinates: CoordinateFile,
    /// Optional per-frame forces.
    pub forces: Vec<Option<Vec<[f64; 3]>>>,
    /// Integration steps, when a `step`/`steps` variable exists; otherwise
    /// frame indices are used.
    pub steps: Vec<usize>,
    /// Frame times in the source units (normally ps).
    pub times: Vec<f64>,
}

/// Conventional alias used by other format modules.
pub type NetcdfData = NetcdfFile;
/// Conventional alias for APIs that use `Structure` terminology.
pub type NetcdfStructure = NetcdfFile;

/// Errors produced while reading an Amber NetCDF trajectory.
#[derive(Debug)]
pub enum NetcdfError {
    Io(io::Error),
    Netcdf(String),
    Parse(String),
    InvalidStructure(String),
}

impl fmt::Display for NetcdfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "NetCDF I/O error: {error}"),
            Self::Netcdf(error) => write!(formatter, "NetCDF reader error: {error}"),
            Self::Parse(error) => write!(formatter, "NetCDF parse error: {error}"),
            Self::InvalidStructure(error) => {
                write!(formatter, "invalid Amber NetCDF structure: {error}")
            }
        }
    }
}

impl std::error::Error for NetcdfError {}

impl From<io::Error> for NetcdfError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<netcdf3::ReadError> for NetcdfError {
    fn from(error: netcdf3::ReadError) -> Self {
        Self::Netcdf(error.to_string())
    }
}

impl NetcdfFile {
    /// Parse a NetCDF trajectory from a path. Gzip-compressed input is
    /// accepted when the path contains a gzip stream.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, NetcdfError> {
        Self::read(File::open(path)?)
    }

    /// Alias for [`NetcdfFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NetcdfError> {
        Self::read_file(path)
    }

    /// Parse a NetCDF trajectory from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, NetcdfError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse an in-memory NetCDF-3 document. Gzip-compressed bytes are
    /// transparently decompressed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetcdfError> {
        let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut decoder = GzDecoder::new(Cursor::new(bytes));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            bytes.to_vec()
        };
        let mut reader = FileReader::open_seek_read("<memory>", Box::new(Cursor::new(bytes)))?;
        parse_reader(&mut reader)
    }

    /// Number of frames in this trajectory.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    /// Number of atoms in each frame.
    #[must_use]
    pub const fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Return one frame by zero-based index.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    /// Convert this trajectory to a coordinate-only universe, retaining
    /// velocities, forces, dimensions, timing, and steps.
    pub fn to_universe(&self) -> crate::Result<Universe> {
        universe_from_netcdf(self.clone())
    }
}

/// Read an Amber NetCDF trajectory from a path.
pub fn read_netcdf(path: impl AsRef<Path>) -> Result<NetcdfFile, NetcdfError> {
    NetcdfFile::read_file(path)
}

impl CoordinateFile {
    /// Read NetCDF coordinates and frame metadata, discarding force arrays.
    pub fn read_netcdf<R: Read>(reader: R) -> Result<Self, NetcdfError> {
        Ok(NetcdfFile::read(reader)?.coordinates)
    }

    /// Parse NetCDF coordinates from bytes, discarding force arrays.
    pub fn from_netcdf_bytes(bytes: &[u8]) -> Result<Self, NetcdfError> {
        Ok(NetcdfFile::from_bytes(bytes)?.coordinates)
    }
}

impl Universe {
    /// Construct a coordinate-only universe from an Amber NetCDF trajectory.
    pub fn from_netcdf(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_netcdf_file(read_netcdf(path)?)
    }

    /// Construct a coordinate-only universe from NetCDF bytes.
    pub fn from_netcdf_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_netcdf_file(NetcdfFile::from_bytes(bytes)?)
    }

    /// Construct a universe from parsed Amber NetCDF data.
    pub fn from_netcdf_file(file: NetcdfFile) -> crate::Result<Self> {
        universe_from_netcdf(file)
    }

    /// Attach an Amber NetCDF trajectory to a PSF topology.
    pub fn from_psf_and_netcdf(
        psf_path: impl AsRef<Path>,
        netcdf_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf(psf_path)?;
        attach_netcdf(&mut universe, read_netcdf(netcdf_path)?)?;
        Ok(universe)
    }

    /// Attach NetCDF bytes to a PSF topology held as text.
    pub fn from_psf_and_netcdf_bytes(psf: &str, bytes: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_psf_str(psf)?;
        attach_netcdf(&mut universe, NetcdfFile::from_bytes(bytes)?)?;
        Ok(universe)
    }

    /// Attach an Amber NetCDF trajectory to a PRMTOP/PARM7/TOP topology.
    pub fn from_prmtop_and_netcdf(
        topology_path: impl AsRef<Path>,
        netcdf_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop(topology_path)?;
        attach_netcdf(&mut universe, read_netcdf(netcdf_path)?)?;
        Ok(universe)
    }

    /// Attach NetCDF bytes to an Amber topology held as text.
    pub fn from_prmtop_and_netcdf_bytes(topology: &str, bytes: &[u8]) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop_str(topology)?;
        attach_netcdf(&mut universe, NetcdfFile::from_bytes(bytes)?)?;
        Ok(universe)
    }
}

fn parse_reader(reader: &mut FileReader) -> Result<NetcdfFile, NetcdfError> {
    let (conventions, convention_version, title, coordinate_dims, shape) = {
        let data_set = reader.data_set();
        let conventions = data_set
            .get_global_attr_as_string("Conventions")
            .ok_or_else(|| {
                NetcdfError::InvalidStructure("missing global Conventions attribute".to_owned())
            })?;
        let convention_version = data_set
            .get_global_attr_as_string("ConventionVersion")
            .ok_or_else(|| {
                NetcdfError::InvalidStructure(
                    "missing global ConventionVersion attribute".to_owned(),
                )
            })?;
        let title = data_set
            .get_global_attr_as_string("title")
            .unwrap_or_default();
        let coordinate_dims = variable_dims(data_set, "coordinates")?;
        let shape = coordinate_shape(data_set, &coordinate_dims)?;
        (
            conventions,
            convention_version,
            title,
            coordinate_dims,
            shape,
        )
    };
    if !conventions.trim().eq_ignore_ascii_case("AMBER") {
        return Err(NetcdfError::InvalidStructure(format!(
            "unsupported Conventions value {conventions:?}; expected AMBER"
        )));
    }
    let (n_frames, n_atoms, has_frame_dim) = shape;
    validate_units(reader, "coordinates", &["angstrom"])?;
    for name in ["velocities", "forces", "cell_lengths", "cell_angles"] {
        let allowed = match name {
            "velocities" => &["angstrom/picosecond"][..],
            "forces" => &["kilocalorie/mole/angstrom"][..],
            "cell_lengths" => &["angstrom"][..],
            _ => &["degree"][..],
        };
        validate_units(reader, name, allowed)?;
    }
    validate_units(reader, "time", &["picosecond"])?;
    validate_scale_factor_targets(reader)?;
    let coordinate_values = read_values(reader, "coordinates")?;
    let expected_values = n_frames
        .checked_mul(n_atoms)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| {
            NetcdfError::InvalidStructure("coordinate size overflows usize".to_owned())
        })?;
    if coordinate_values.len() != expected_values {
        return Err(NetcdfError::InvalidStructure(format!(
            "coordinates contains {} values; expected {expected_values}",
            coordinate_values.len()
        )));
    }

    let velocity_values =
        optional_payload(reader, "velocities", &coordinate_dims, expected_values)?;
    let force_values = optional_payload(reader, "forces", &coordinate_dims, expected_values)?;
    let times = read_times(reader, n_frames, has_frame_dim)?;
    let steps = read_steps(reader, n_frames, has_frame_dim)?;
    let cell_lengths = read_cell_component(reader, "cell_lengths", n_frames, has_frame_dim)?;
    let cell_angles = read_cell_component(reader, "cell_angles", n_frames, has_frame_dim)?;

    let mut frames = Vec::with_capacity(n_frames);
    let mut forces = Vec::with_capacity(n_frames);
    for frame_index in 0..n_frames {
        let start = frame_index * n_atoms * 3;
        let positions = triples(&coordinate_values[start..start + n_atoms * 3]);
        let velocities = velocity_values
            .as_ref()
            .map(|values| triples(&values[start..start + n_atoms * 3]));
        let force = force_values
            .as_ref()
            .map(|values| triples(&values[start..start + n_atoms * 3]));
        let dimensions = match (&cell_lengths, &cell_angles) {
            (Some(lengths), Some(angles)) => dimensions_for_frame(lengths, angles, frame_index),
            _ => None,
        };
        let mut coordinate = CoordinateFrame::new(positions);
        coordinate.velocities = velocities;
        coordinate.dimensions = dimensions;
        coordinate.step = steps[frame_index];
        coordinate.time = times[frame_index];
        frames.push(coordinate);
        forces.push(force);
    }

    Ok(NetcdfFile {
        title,
        conventions,
        convention_version,
        n_atoms,
        coordinates: CoordinateFile::new(frames),
        forces,
        steps,
        times,
    })
}

fn variable_dims(data_set: &netcdf3::DataSet, name: &str) -> Result<Vec<String>, NetcdfError> {
    data_set
        .get_var(name)
        .map(|variable| variable.dim_names())
        .ok_or_else(|| NetcdfError::InvalidStructure(format!("missing required variable {name:?}")))
}

fn coordinate_shape(
    data_set: &netcdf3::DataSet,
    dims: &[String],
) -> Result<(usize, usize, bool), NetcdfError> {
    let (n_frames, n_atoms) = match dims {
        [frame, atom, spatial] if frame == "frame" && atom == "atom" && spatial == "spatial" => {
            let frames = data_set.dim_size("frame").unwrap_or(0);
            (frames, data_set.dim_size("atom").unwrap_or(0))
        }
        [atom, spatial] if atom == "atom" && spatial == "spatial" => {
            (1, data_set.dim_size("atom").unwrap_or(0))
        }
        _ => {
            return Err(NetcdfError::InvalidStructure(format!(
                "coordinates dimensions must be (frame, atom, spatial) or (atom, spatial), got {dims:?}"
            )));
        }
    };
    if n_frames == 0 || n_atoms == 0 {
        return Err(NetcdfError::InvalidStructure(
            "coordinates must contain at least one frame and atom".to_owned(),
        ));
    }
    if data_set.dim_size("spatial") != Some(3) {
        return Err(NetcdfError::InvalidStructure(
            "spatial dimension must have size 3".to_owned(),
        ));
    }
    Ok((
        n_frames,
        n_atoms,
        dims.first().is_some_and(|dim| dim == "frame"),
    ))
}

fn optional_payload(
    reader: &mut FileReader,
    name: &str,
    coordinate_dims: &[String],
    expected_values: usize,
) -> Result<Option<Vec<f64>>, NetcdfError> {
    {
        let data_set = reader.data_set();
        let Some(variable) = data_set.get_var(name) else {
            return Ok(None);
        };
        if variable.dim_names() != coordinate_dims {
            return Err(NetcdfError::InvalidStructure(format!(
                "{name} dimensions {:?} do not match coordinates dimensions {coordinate_dims:?}",
                variable.dim_names()
            )));
        }
    }
    let values = read_values(reader, name)?;
    if values.len() != expected_values {
        return Err(NetcdfError::InvalidStructure(format!(
            "{name} contains {} values; expected {expected_values}",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn validate_units(reader: &FileReader, name: &str, allowed: &[&str]) -> Result<(), NetcdfError> {
    let Some(units) = reader.data_set().get_var_attr_as_string(name, "units") else {
        return Ok(());
    };
    if allowed
        .iter()
        .any(|expected| units.trim().eq_ignore_ascii_case(expected))
    {
        Ok(())
    } else {
        Err(NetcdfError::InvalidStructure(format!(
            "variable {name:?} uses unsupported units {units:?}; expected one of {allowed:?}"
        )))
    }
}

fn validate_scale_factor_targets(reader: &FileReader) -> Result<(), NetcdfError> {
    let allowed = [
        "coordinates",
        "velocities",
        "forces",
        "cell_lengths",
        "cell_angles",
        "time",
        "step",
        "steps",
    ];
    for name in reader.data_set().get_var_names() {
        if reader
            .data_set()
            .get_var(&name)
            .is_some_and(|var| var.has_attr("scale_factor"))
            && !allowed.contains(&name.as_str())
        {
            return Err(NetcdfError::InvalidStructure(format!(
                "scale_factor on unsupported variable {name:?}"
            )));
        }
    }
    Ok(())
}

fn read_values(reader: &mut FileReader, name: &str) -> Result<Vec<f64>, NetcdfError> {
    let scale = {
        let data_set = reader.data_set();
        let f64_scale = data_set.get_var_attr_f64(name, "scale_factor");
        let f32_scale = data_set.get_var_attr_f32(name, "scale_factor");
        let has_scale = data_set
            .get_var(name)
            .is_some_and(|variable| variable.has_attr("scale_factor"));
        if f64_scale.is_some() && f32_scale.is_some() {
            return Err(NetcdfError::Parse(format!(
                "variable {name:?} has ambiguous scale_factor attribute"
            )));
        }
        if has_scale && f64_scale.is_none() && f32_scale.is_none() {
            return Err(NetcdfError::Parse(format!(
                "variable {name:?} has a non-numeric scale_factor attribute"
            )));
        }
        let value = f64_scale
            .and_then(|values| values.first().copied())
            .or_else(|| f32_scale.and_then(|values| values.first().copied().map(f64::from)))
            .unwrap_or(1.0);
        if !value.is_finite() {
            return Err(NetcdfError::Parse(format!(
                "variable {name:?} has a non-finite scale_factor"
            )));
        }
        value
    };
    let values = match reader.read_var(name)? {
        DataVector::F32(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        DataVector::F64(values) => values,
        DataVector::I8(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        DataVector::U8(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        DataVector::I16(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        DataVector::I32(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
    };
    Ok(values.into_iter().map(|value| value * scale).collect())
}

fn read_times(
    reader: &mut FileReader,
    n_frames: usize,
    has_frame_dim: bool,
) -> Result<Vec<f64>, NetcdfError> {
    let dims = {
        let data_set = reader.data_set();
        let Some(variable) = data_set.get_var("time") else {
            return Ok((0..n_frames).map(|index| index as f64).collect());
        };
        variable.dim_names()
    };
    if has_frame_dim {
        if dims != ["frame"] {
            return Err(NetcdfError::InvalidStructure(format!(
                "time dimensions must be (frame), got {dims:?}"
            )));
        }
    } else if dims != ["time"] && !dims.is_empty() {
        return Err(NetcdfError::InvalidStructure(format!(
            "time dimensions must be (time) or scalar, got {dims:?}"
        )));
    }
    let values = read_values(reader, "time")?;
    if values.len() == 1 && n_frames == 1 {
        return Ok(values);
    }
    if values.len() != n_frames {
        return Err(NetcdfError::InvalidStructure(format!(
            "time contains {} values; expected {n_frames}",
            values.len()
        )));
    }
    Ok(values)
}

fn read_steps(
    reader: &mut FileReader,
    n_frames: usize,
    has_frame_dim: bool,
) -> Result<Vec<usize>, NetcdfError> {
    let name = {
        let data_set = reader.data_set();
        ["step", "steps"]
            .into_iter()
            .find(|name| data_set.has_var(name))
    };
    let Some(name) = name else {
        return Ok((0..n_frames).collect());
    };
    let dims = reader
        .data_set()
        .get_var(name)
        .expect("variable name came from has_var")
        .dim_names();
    if has_frame_dim && dims != ["frame"] {
        return Err(NetcdfError::InvalidStructure(format!(
            "{name} dimensions must be (frame), got {:?}",
            dims
        )));
    }
    let values = read_values(reader, name)?;
    if values.len() != n_frames {
        return Err(NetcdfError::InvalidStructure(format!(
            "{name} contains {} values; expected {n_frames}",
            values.len()
        )));
    }
    values
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value < 0.0 || value > usize::MAX as f64 {
                return Err(NetcdfError::InvalidStructure(format!(
                    "{name} contains invalid step value {value}"
                )));
            }
            Ok(value as usize)
        })
        .collect()
}

fn read_cell_component(
    reader: &mut FileReader,
    name: &str,
    n_frames: usize,
    has_frame_dim: bool,
) -> Result<Option<Vec<f64>>, NetcdfError> {
    let dims = {
        let data_set = reader.data_set();
        let Some(variable) = data_set.get_var(name) else {
            return Ok(None);
        };
        variable.dim_names()
    };
    let expected_dims = if has_frame_dim {
        vec!["frame".to_owned(), "cell_spatial".to_owned()]
    } else {
        vec!["cell_spatial".to_owned()]
    };
    let angular = name == "cell_angles";
    let expected_dims = if angular {
        if has_frame_dim {
            vec!["frame".to_owned(), "cell_angular".to_owned()]
        } else {
            vec!["cell_angular".to_owned()]
        }
    } else {
        expected_dims
    };
    if dims != expected_dims {
        return Err(NetcdfError::InvalidStructure(format!(
            "{name} dimensions must be {expected_dims:?}, got {dims:?}"
        )));
    }
    let expected_len = if has_frame_dim { n_frames * 3 } else { 3 };
    let values = read_values(reader, name)?;
    if values.len() != expected_len {
        return Err(NetcdfError::InvalidStructure(format!(
            "{name} contains {} values; expected {expected_len}",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn dimensions_for_frame(lengths: &[f64], angles: &[f64], frame: usize) -> Option<[f64; 6]> {
    let length_start = if lengths.len() == 3 { 0 } else { frame * 3 };
    let angle_start = if angles.len() == 3 { 0 } else { frame * 3 };
    let values = [
        lengths[length_start],
        lengths[length_start + 1],
        lengths[length_start + 2],
        angles[angle_start],
        angles[angle_start + 1],
        angles[angle_start + 2],
    ];
    if values[..3]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        && values[3..]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0 && *value < 180.0)
    {
        Some(values)
    } else {
        None
    }
}

fn triples(values: &[f64]) -> Vec<[f64; 3]> {
    values.as_chunks::<3>().0.to_vec()
}

fn universe_from_netcdf(file: NetcdfFile) -> crate::Result<Universe> {
    let first = file.coordinates.frames.first().ok_or_else(|| {
        crate::Error::InvalidInput("NetCDF trajectory contains no frames".to_owned())
    })?;
    let atoms = first
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            let mut atom = Atom::new(index, "X", *position);
            atom.element = Some("X".to_owned());
            atom
        })
        .collect::<Vec<_>>();
    let mut universe = Universe {
        topology: Topology::new(atoms),
        trajectory: Trajectory::default(),
    };
    attach_netcdf(&mut universe, file)?;
    Ok(universe)
}

fn attach_netcdf(universe: &mut Universe, file: NetcdfFile) -> crate::Result<()> {
    if file.n_atoms != universe.n_atoms() {
        return Err(crate::Error::InvalidInput(format!(
            "NetCDF contains {} atoms, topology contains {}",
            file.n_atoms,
            universe.n_atoms()
        )));
    }
    if file.coordinates.frames.is_empty() {
        return Err(crate::Error::InvalidInput(
            "NetCDF trajectory contains no frames".to_owned(),
        ));
    }
    let mut frames = Vec::with_capacity(file.coordinates.frames.len());
    for (index, source) in file.coordinates.frames.into_iter().enumerate() {
        if source.n_atoms() != universe.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "NetCDF frame contains {} atoms, topology contains {}",
                source.n_atoms(),
                universe.n_atoms()
            )));
        }
        let mut frame = Frame::new(source.positions);
        frame.velocities = source.velocities;
        frame.dimensions = source.dimensions;
        frame.step = source.step;
        frame.time = source.time;
        if let Some(force) = file.forces.get(index).cloned().flatten() {
            frame.forces = Some(force);
        }
        frames.push(frame);
    }
    universe.trajectory = Trajectory::new(frames);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/Amber")
            .join(name)
    }

    #[test]
    fn reads_positions_forces_and_time() {
        let file = read_netcdf(fixture("posfor.ncdf")).unwrap();
        assert_eq!(file.n_atoms, 442);
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.coordinates.frames[0].step, 0);
        assert!((file.coordinates.frames[0].time - 35.02).abs() < 1e-5);
        assert_eq!(file.forces[0].as_ref().unwrap().len(), 442);
        assert!((file.coordinates.frames[0].positions[0][0] + 0.11980818).abs() < 1e-6);
    }

    #[test]
    fn reads_large_multiframe_fixture() {
        let file = read_netcdf(fixture("bala.ncdf")).unwrap();
        assert_eq!(file.n_atoms, 2661);
        assert_eq!(file.n_frames(), 30);
        assert_eq!(
            file.coordinates.frames[0].dimensions.unwrap()[3..],
            [90.0, 90.0, 90.0]
        );
    }

    #[test]
    fn reads_box_velocities_and_scale_factor() {
        let file = read_netcdf(fixture("ace_tip3p.nc")).unwrap();
        let frame = &file.coordinates.frames[0];
        assert_eq!(file.n_frames(), 10);
        assert_eq!(
            frame.dimensions.unwrap()[..3],
            [28.81876287443224, 28.278752611423382, 27.726163965035884]
        );
        assert!((frame.velocities.as_ref().unwrap()[0][0] + 0.5301689 * 20.455).abs() < 1e-4);
        assert_eq!(file.forces[0].as_ref().unwrap().len(), file.n_atoms);
    }

    #[test]
    fn missing_time_uses_frame_index() {
        let file = read_netcdf(fixture("cpptraj_traj.nc")).unwrap();
        assert_eq!(file.times, vec![0.0, 1.0, 2.0]);
        assert_eq!(file.steps, vec![0, 1, 2]);
    }

    #[test]
    fn universe_constructor_attaches_forces() {
        let file = read_netcdf(fixture("posfor.ncdf")).unwrap();
        let universe = Universe::from_netcdf_file(file).unwrap();
        assert_eq!(universe.n_atoms(), 442);
        assert_eq!(universe.n_frames(), 2);
        assert!(universe.trajectory.frames[0].forces.is_some());
    }

    #[test]
    fn prmtop_constructor_attaches_trajectory() {
        let universe =
            Universe::from_prmtop_and_netcdf(fixture("posfor.top"), fixture("posfor.ncdf"))
                .unwrap();
        assert_eq!(universe.n_atoms(), 442);
        assert_eq!(universe.n_frames(), 2);
        assert!(universe.trajectory.frames[0].forces.is_some());
    }
}
