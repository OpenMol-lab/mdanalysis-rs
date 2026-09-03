//! LAMMPS DATA and ASCII dump (text) topology and trajectory support.
//!
//! A LAMMPS DATA file combines a topology header, an `Atoms` section, and
//! optional sections such as `Masses`, `Velocities`, and `Bonds`.  This module
//! focuses on the fields needed to construct an MD trajectory while retaining
//! non-contiguous atom IDs and restricted-triclinic box bounds.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Restricted-triclinic LAMMPS box bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LammpsBox {
    pub xlo: f64,
    pub xhi: f64,
    pub ylo: f64,
    pub yhi: f64,
    pub zlo: f64,
    pub zhi: f64,
    /// XY, XZ, and YZ tilt factors.  They are zero for an orthogonal box.
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
}

impl LammpsBox {
    /// Construct an orthogonal box from lower and upper bounds.
    #[must_use]
    pub const fn orthogonal(xlo: f64, xhi: f64, ylo: f64, yhi: f64, zlo: f64, zhi: f64) -> Self {
        Self {
            xlo,
            xhi,
            ylo,
            yhi,
            zlo,
            zhi,
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
        }
    }

    /// Return whether any restricted-triclinic tilt is present.
    #[must_use]
    pub fn is_triclinic(self) -> bool {
        self.xy != 0.0 || self.xz != 0.0 || self.yz != 0.0
    }

    /// Convert bounds and tilts to `[a, b, c, alpha, beta, gamma]`.
    #[must_use]
    pub fn dimensions(self) -> [f64; 6] {
        crate::mdamath::triclinic_box([
            [self.xhi - self.xlo, 0.0, 0.0],
            [self.xy, self.yhi - self.ylo, 0.0],
            [self.xz, self.yz, self.zhi - self.zlo],
        ])
    }

    fn image_shift(self, image: [i32; 3]) -> [f64; 3] {
        let [ix, iy, iz] = image.map(f64::from);
        [
            ix * (self.xhi - self.xlo) + iy * self.xy + iz * self.xz,
            iy * (self.yhi - self.ylo) + iz * self.yz,
            iz * (self.zhi - self.zlo),
        ]
    }
}

/// An atom record from a LAMMPS DATA `Atoms` section.
#[derive(Clone, Debug, PartialEq)]
pub struct LammpsAtom {
    /// LAMMPS atom ID.  IDs need not be contiguous.
    pub id: usize,
    /// Molecule/residue ID, when supplied by the atom style.
    pub molecule_id: Option<i64>,
    /// LAMMPS atom type ID.
    pub atom_type: usize,
    /// Partial charge, when supplied by the atom style.
    pub charge: Option<f64>,
    pub position: [f64; 3],
    /// Optional image flags (`nx`, `ny`, `nz`) retained from full atom lines.
    pub image: Option<[i32; 3]>,
}

impl LammpsAtom {
    /// Residue alias used by the topology API.  LAMMPS atom styles without a
    /// molecule field use residue 1.
    #[must_use]
    pub fn resid(&self) -> i32 {
        self.molecule_id
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(1)
    }
}

/// A bond record from a LAMMPS DATA `Bonds` section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LammpsBond {
    pub id: usize,
    pub bond_type: usize,
    /// Atom IDs in the source LAMMPS numbering space.
    pub atom1: usize,
    pub atom2: usize,
}

/// A parsed LAMMPS DATA file.
#[derive(Clone, Debug, PartialEq)]
pub struct LammpsData {
    pub title: String,
    pub atoms: Vec<LammpsAtom>,
    pub bonds: Vec<LammpsBond>,
    /// Per-type masses from the `Masses` section.
    pub masses: BTreeMap<usize, f64>,
    /// Velocities in the same (sorted-by-ID) order as [`Self::atoms`].
    pub velocities: Option<Vec<[f64; 3]>>,
    pub bounds: Option<LammpsBox>,
    /// Atom-style field description used for parsing and writing.  Standard
    /// names (`atomic`, `molecular`, `charge`, and `full`) are accepted, as
    /// are custom strings such as `id resid type charge x y z`.
    pub atom_style: String,
}

impl Default for LammpsData {
    fn default() -> Self {
        Self {
            title: String::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            masses: BTreeMap::new(),
            velocities: None,
            bounds: None,
            atom_style: "full".to_owned(),
        }
    }
}

/// Descriptive alias for callers that prefer an explicit file name.
pub type LammpsDataFile = LammpsData;

impl LammpsData {
    /// Parse a LAMMPS DATA document from a string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, LammpsError> {
        parse_data(input, None)
    }

    /// Parse a DATA document with an explicit atom-style description.
    pub fn from_str_with_atom_style(input: &str, atom_style: &str) -> Result<Self, LammpsError> {
        parse_data(input, Some(atom_style))
    }

    /// Read a DATA document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, LammpsError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Read a DATA document from a filesystem path.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, LammpsError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    /// Serialize this file using the stored atom-style description.
    pub fn to_string(&self) -> Result<String, LammpsError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output).map_err(|error| {
            LammpsError::InvalidStructure(format!("DATA output is not UTF-8: {error}"))
        })
    }

    /// Write this DATA file to any writer.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), LammpsError> {
        validate_data(self)?;
        let style = if self.atom_style.trim().is_empty() {
            "full"
        } else {
            self.atom_style.trim()
        };
        let fields = style_fields(style)?;
        writeln!(writer, "{}", self.title)?;
        writeln!(writer)?;
        writeln!(writer, "{} atoms", self.atoms.len())?;
        writeln!(writer, "{} bonds", self.bonds.len())?;
        let atom_types = self
            .masses
            .keys()
            .copied()
            .chain(self.atoms.iter().map(|atom| atom.atom_type))
            .max()
            .unwrap_or(0);
        writeln!(writer, "{atom_types} atom types")?;
        let bond_types = self
            .bonds
            .iter()
            .map(|bond| bond.bond_type)
            .max()
            .unwrap_or(0);
        if bond_types > 0 {
            writeln!(writer, "{bond_types} bond types")?;
        }
        if let Some(bounds) = self.bounds {
            writeln!(writer)?;
            writeln!(writer, "{:.16e} {:.16e} xlo xhi", bounds.xlo, bounds.xhi)?;
            writeln!(writer, "{:.16e} {:.16e} ylo yhi", bounds.ylo, bounds.yhi)?;
            writeln!(writer, "{:.16e} {:.16e} zlo zhi", bounds.zlo, bounds.zhi)?;
            if bounds.is_triclinic() {
                writeln!(
                    writer,
                    "{:.16e} {:.16e} {:.16e} xy xz yz",
                    bounds.xy, bounds.xz, bounds.yz
                )?;
            }
        }
        if !self.masses.is_empty() {
            writeln!(writer, "\nMasses\n")?;
            for (atom_type, mass) in &self.masses {
                writeln!(writer, "{atom_type:>8} {mass:.16e}")?;
            }
        }
        writeln!(writer, "\nAtoms # {style}\n")?;
        for atom in &self.atoms {
            let mut values = fields
                .iter()
                .map(|field| atom_field(atom, field))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(image) = atom.image {
                values.extend(image.into_iter().map(|value| value.to_string()));
            }
            writeln!(writer, "{}", values.join(" "))?;
        }
        if let Some(velocities) = &self.velocities {
            writeln!(writer, "\nVelocities\n")?;
            for (atom, velocity) in self.atoms.iter().zip(velocities) {
                writeln!(
                    writer,
                    "{} {:.16e} {:.16e} {:.16e}",
                    atom.id, velocity[0], velocity[1], velocity[2]
                )?;
            }
        }
        if !self.bonds.is_empty() {
            writeln!(writer, "\nBonds\n")?;
            for bond in &self.bonds {
                writeln!(
                    writer,
                    "{} {} {} {}",
                    bond.id, bond.bond_type, bond.atom1, bond.atom2
                )?;
            }
        }
        Ok(())
    }

    /// Write this DATA file to a filesystem path.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), LammpsError> {
        self.write(File::create(path)?)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    #[must_use]
    pub fn dimensions(&self) -> Option<[f64; 6]> {
        self.bounds.map(LammpsBox::dimensions)
    }
}

impl std::str::FromStr for LammpsData {
    type Err = LammpsError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Read a LAMMPS DATA file from a path.
pub fn read_lammps_data<P: AsRef<Path>>(path: P) -> Result<LammpsData, LammpsError> {
    LammpsData::read_file(path)
}

/// Write a LAMMPS DATA file to a path.
pub fn write_lammps_data<P: AsRef<Path>>(path: P, data: &LammpsData) -> Result<(), LammpsError> {
    data.write_file(path)
}

/// Coordinate convention used by columns in a LAMMPS ASCII dump.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LammpsCoordinateConvention {
    /// Select the first available convention in LAMMPS' conventional order.
    #[default]
    Auto,
    /// Wrapped Cartesian coordinates (`x`, `y`, `z`).
    Unscaled,
    /// Wrapped fractional coordinates (`xs`, `ys`, `zs`).
    Scaled,
    /// Unwrapped Cartesian coordinates (`xu`, `yu`, `zu`).
    Unwrapped,
    /// Unwrapped fractional coordinates (`xsu`, `ysu`, `zsu`).
    ScaledUnwrapped,
}

impl LammpsCoordinateConvention {
    fn columns(self) -> [&'static str; 3] {
        match self {
            Self::Auto | Self::Unscaled => ["x", "y", "z"],
            Self::Scaled => ["xs", "ys", "zs"],
            Self::Unwrapped => ["xu", "yu", "zu"],
            Self::ScaledUnwrapped => ["xsu", "ysu", "zsu"],
        }
    }

    fn is_scaled(self) -> bool {
        matches!(self, Self::Scaled | Self::ScaledUnwrapped)
    }

    fn is_unwrapped(self) -> bool {
        matches!(self, Self::Unwrapped | Self::ScaledUnwrapped)
    }
}

impl fmt::Display for LammpsCoordinateConvention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Auto => "auto",
            Self::Unscaled => "unscaled",
            Self::Scaled => "scaled",
            Self::Unwrapped => "unwrapped",
            Self::ScaledUnwrapped => "scaled_unwrapped",
        };
        formatter.write_str(name)
    }
}

impl std::str::FromStr for LammpsCoordinateConvention {
    type Err = LammpsError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "unscaled" => Ok(Self::Unscaled),
            "scaled" => Ok(Self::Scaled),
            "unwrapped" => Ok(Self::Unwrapped),
            "scaled_unwrapped" => Ok(Self::ScaledUnwrapped),
            _ => Err(LammpsError::InvalidStructure(format!(
                "invalid LAMMPS coordinate convention {value:?}; expected auto, unscaled, scaled, unwrapped, or scaled_unwrapped"
            ))),
        }
    }
}

/// Options controlling parsing of a LAMMPS ASCII dump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LammpsDumpOptions {
    pub coordinate_convention: LammpsCoordinateConvention,
    /// Add `ix`, `iy`, and `iz` image flags to positions when present.
    pub unwrap_images: bool,
}

impl Default for LammpsDumpOptions {
    fn default() -> Self {
        Self {
            coordinate_convention: LammpsCoordinateConvention::Auto,
            unwrap_images: false,
        }
    }
}

/// One parsed frame from a LAMMPS ASCII dump.
#[derive(Clone, Debug, PartialEq)]
pub struct LammpsDumpFrame {
    pub step: usize,
    pub time: f64,
    pub atom_ids: Vec<usize>,
    pub positions: Vec<[f64; 3]>,
    pub velocities: Option<Vec<[f64; 3]>>,
    pub forces: Option<Vec<[f64; 3]>>,
    pub dimensions: Option<[f64; 6]>,
    /// True lower/upper bounds and restricted-triclinic tilts.
    pub bounds: LammpsBox,
    /// Numeric columns not consumed as coordinates, topology, velocities, or forces.
    pub additional_columns: BTreeMap<String, Vec<f64>>,
}

impl LammpsDumpFrame {
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

/// A parsed LAMMPS ASCII dump trajectory and first-frame topology metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct LammpsDumpFile {
    pub frames: Vec<LammpsDumpFrame>,
    pub atom_ids: Vec<usize>,
    pub molecule_ids: Option<Vec<i64>>,
    pub atom_types: Option<Vec<String>>,
    pub masses: Option<Vec<f64>>,
    /// Element symbols are normalized to the crate's uppercase convention;
    /// unknown source symbols are represented by an empty string.
    pub elements: Option<Vec<String>>,
    pub charges: Option<Vec<f64>>,
    pub coordinate_convention: LammpsCoordinateConvention,
}

pub type LammpsDumpData = LammpsDumpFile;
pub type LammpsDumpReader = LammpsDumpFile;

impl LammpsDumpFile {
    /// Parse a LAMMPS dump using automatic coordinate-convention detection.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, LammpsError> {
        Self::from_str_with_options(input, LammpsDumpOptions::default())
    }

    /// Parse a LAMMPS dump with explicit coordinate and image options.
    pub fn from_str_with_options(
        input: &str,
        options: LammpsDumpOptions,
    ) -> Result<Self, LammpsError> {
        parse_dump(input, options)
    }

    /// Parse a LAMMPS dump with one explicit coordinate convention.
    pub fn from_str_with_convention(
        input: &str,
        coordinate_convention: LammpsCoordinateConvention,
    ) -> Result<Self, LammpsError> {
        Self::from_str_with_options(
            input,
            LammpsDumpOptions {
                coordinate_convention,
                ..LammpsDumpOptions::default()
            },
        )
    }

    /// Read an uncompressed LAMMPS dump from any reader.
    pub fn read<R: Read>(reader: R) -> Result<Self, LammpsError> {
        Self::read_with_options(reader, LammpsDumpOptions::default())
    }

    /// Read a LAMMPS dump from any reader with explicit options.
    pub fn read_with_options<R: Read>(
        mut reader: R,
        options: LammpsDumpOptions,
    ) -> Result<Self, LammpsError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str_with_options(&input, options)
    }

    /// Read a LAMMPS dump from a path, transparently decoding gzip and bzip2
    /// streams based on their file signatures.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, LammpsError> {
        Self::read_file_with_options(path, LammpsDumpOptions::default())
    }

    /// Read a LAMMPS dump from a path with explicit options.
    pub fn read_file_with_options<P: AsRef<Path>>(
        path: P,
        options: LammpsDumpOptions,
    ) -> Result<Self, LammpsError> {
        let path = path.as_ref();
        let input = crate::io_utils::read_text_file(path)?;
        Self::from_str_with_options(&input, options)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atom_ids.len()
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&LammpsDumpFrame> {
        self.frames.get(index)
    }
}

impl std::str::FromStr for LammpsDumpFile {
    type Err = LammpsError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Read a LAMMPS ASCII dump from a filesystem path.
pub fn read_lammps_dump<P: AsRef<Path>>(path: P) -> Result<LammpsDumpFile, LammpsError> {
    LammpsDumpFile::read_file(path)
}

/// Read a LAMMPS ASCII dump from a path with explicit options.
pub fn read_lammps_dump_with_options<P: AsRef<Path>>(
    path: P,
    options: LammpsDumpOptions,
) -> Result<LammpsDumpFile, LammpsError> {
    LammpsDumpFile::read_file_with_options(path, options)
}

impl crate::core::Universe {
    /// Construct a universe from a LAMMPS DATA file on disk.
    pub fn from_lammps_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_lammps_data(read_lammps_data(path)?)
    }

    /// Construct a universe from a parsed LAMMPS DATA file.
    pub fn from_lammps_data(data: LammpsData) -> crate::Result<Self> {
        validate_data(&data)?;
        if data.atoms.is_empty() {
            return Err(crate::Error::InvalidInput(
                "LAMMPS DATA file contains no atoms".to_owned(),
            ));
        }
        let atoms = data
            .atoms
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let mut atom = crate::core::Atom::new(
                    index,
                    format!("TYPE{}", source.atom_type),
                    source.position,
                );
                atom.atom_type = Some(source.atom_type.to_string());
                atom.mass = data.masses.get(&source.atom_type).copied().unwrap_or(0.0);
                atom.charge = source.charge.unwrap_or(0.0);
                atom.resid = source.resid();
                atom.resname = "SYSTEM".to_owned();
                atom
            })
            .collect::<Vec<_>>();
        let mut universe = Self::from_atoms(atoms);
        let dimensions = data.dimensions();
        let velocities = data.velocities;
        let frame =
            universe.trajectory.frames.first_mut().ok_or_else(|| {
                crate::Error::InvalidInput("failed to create DATA frame".to_owned())
            })?;
        frame.velocities = velocities;
        frame.dimensions = dimensions;
        let id_to_index = data
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom.id, index))
            .collect::<HashMap<_, _>>();
        for bond in data.bonds {
            if let (Some(&atom1), Some(&atom2)) =
                (id_to_index.get(&bond.atom1), id_to_index.get(&bond.atom2))
            {
                let mut topology_bond = crate::core::Bond::new(atom1, atom2);
                topology_bond.order = u8::try_from(bond.bond_type).ok();
                universe.topology.add_bond(topology_bond);
            }
        }
        Ok(universe)
    }

    /// Construct a universe from a LAMMPS DATA document held in memory.
    pub fn from_lammps_data_str(input: &str) -> crate::Result<Self> {
        Self::from_lammps_data(LammpsData::from_str(input)?)
    }

    /// Alias for [`Self::from_lammps_data`].
    pub fn from_lammps(data: LammpsData) -> crate::Result<Self> {
        Self::from_lammps_data(data)
    }

    /// Alias for [`Self::from_lammps_data_str`].
    pub fn from_lammps_str(input: &str) -> crate::Result<Self> {
        Self::from_lammps_data_str(input)
    }

    /// Construct a universe from a LAMMPS ASCII dump on disk.
    pub fn from_lammps_dump(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_lammps_dump_file(LammpsDumpFile::read_file(path)?)
    }

    /// Construct a universe from a LAMMPS ASCII dump with explicit options.
    pub fn from_lammps_dump_with_options(
        path: impl AsRef<Path>,
        options: LammpsDumpOptions,
    ) -> crate::Result<Self> {
        Self::from_lammps_dump_file(LammpsDumpFile::read_file_with_options(path, options)?)
    }

    /// Construct a universe from a LAMMPS dump held in memory.
    pub fn from_lammps_dump_str(input: &str) -> crate::Result<Self> {
        Self::from_lammps_dump_file(LammpsDumpFile::from_str(input)?)
    }

    /// Construct a universe from a LAMMPS dump held in memory with options.
    pub fn from_lammps_dump_str_with_options(
        input: &str,
        options: LammpsDumpOptions,
    ) -> crate::Result<Self> {
        Self::from_lammps_dump_file(LammpsDumpFile::from_str_with_options(input, options)?)
    }

    /// Construct a universe from a parsed LAMMPS dump.
    pub fn from_lammps_dump_file(file: LammpsDumpFile) -> crate::Result<Self> {
        validate_dump_file(&file)?;
        let count = file.n_atoms();
        let first = &file.frames[0];
        let atoms = (0..count)
            .map(|index| {
                let atom_type = file
                    .atom_types
                    .as_ref()
                    .and_then(|values| values.get(index))
                    .cloned()
                    .unwrap_or_else(|| "1".to_owned());
                let element = file
                    .elements
                    .as_ref()
                    .and_then(|values| values.get(index))
                    .filter(|value| !value.is_empty())
                    .cloned();
                let name = element
                    .clone()
                    .unwrap_or_else(|| format!("TYPE{atom_type}"));
                let mut atom = crate::core::Atom::new(index, name, first.positions[index]);
                atom.atom_type = Some(atom_type.clone());
                atom.element = element;
                atom.mass = file
                    .masses
                    .as_ref()
                    .and_then(|values| values.get(index))
                    .copied()
                    .or_else(|| atom.element.as_deref().map(crate::guesser::guess_atom_mass))
                    .unwrap_or(1.0);
                atom.charge = file
                    .charges
                    .as_ref()
                    .and_then(|values| values.get(index))
                    .copied()
                    .unwrap_or(0.0);
                atom.resid = file
                    .molecule_ids
                    .as_ref()
                    .and_then(|values| values.get(index))
                    .and_then(|value| i32::try_from(*value).ok())
                    .unwrap_or(1);
                atom.resname = "SYSTEM".to_owned();
                atom
            })
            .collect();
        let topology = crate::core::Topology::new(atoms);
        Ok(Self {
            topology,
            trajectory: dump_trajectory(file.frames),
        })
    }

    /// Alias for [`Self::from_lammps_dump_file`].
    pub fn from_lammps_dump_data(file: LammpsDumpFile) -> crate::Result<Self> {
        Self::from_lammps_dump_file(file)
    }

    /// Construct a universe from LAMMPS DATA topology and dump trajectory.
    pub fn from_lammps_data_and_dump(
        data: LammpsData,
        dump: LammpsDumpFile,
    ) -> crate::Result<Self> {
        validate_dump_file(&dump)?;
        let data_ids = data.atoms.iter().map(|atom| atom.id).collect::<Vec<_>>();
        let mut universe = Self::from_lammps_data(data)?;
        if dump.n_atoms() != universe.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump contains {} atoms, DATA contains {}",
                dump.n_atoms(),
                universe.n_atoms()
            )));
        }
        let dump_ids = dump.atom_ids.iter().copied().collect::<HashSet<_>>();
        let data_id_set = data_ids.iter().copied().collect::<HashSet<_>>();
        if dump_ids != data_id_set {
            return Err(crate::Error::InvalidInput(
                "LAMMPS dump atom IDs do not match DATA atom IDs".to_owned(),
            ));
        }
        let frames = reorder_dump_frames(dump.frames, &dump.atom_ids, &data_ids);
        universe.trajectory = dump_trajectory(frames);
        if let Some(first_frame) = universe.trajectory.frames.first() {
            for (atom, position) in universe
                .topology
                .atoms
                .iter_mut()
                .zip(&first_frame.positions)
            {
                atom.position = *position;
            }
        }
        Ok(universe)
    }

    /// Construct a universe from LAMMPS DATA and dump files on disk.
    pub fn from_lammps_files(
        data_path: impl AsRef<Path>,
        dump_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        Self::from_lammps_data_and_dump(
            LammpsData::read_file(data_path)?,
            LammpsDumpFile::read_file(dump_path)?,
        )
    }
}

fn dump_trajectory(frames: Vec<LammpsDumpFrame>) -> crate::core::Trajectory {
    crate::core::Trajectory::new(
        frames
            .into_iter()
            .map(|source| {
                let mut frame = crate::core::Frame::new(source.positions);
                frame.velocities = source.velocities;
                frame.forces = source.forces;
                frame.dimensions = source.dimensions;
                frame.step = source.step;
                frame.time = source.time;
                frame
            })
            .collect(),
    )
}

fn reorder_dump_frames(
    frames: Vec<LammpsDumpFrame>,
    source_ids: &[usize],
    target_ids: &[usize],
) -> Vec<LammpsDumpFrame> {
    if source_ids == target_ids {
        return frames;
    }
    let source_indices = source_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<HashMap<_, _>>();
    let order = target_ids
        .iter()
        .map(|id| source_indices[id])
        .collect::<Vec<_>>();
    frames
        .into_iter()
        .map(|mut frame| {
            frame.atom_ids = target_ids.to_vec();
            frame.positions = order.iter().map(|&index| frame.positions[index]).collect();
            frame.velocities = frame
                .velocities
                .map(|values| order.iter().map(|&index| values[index]).collect());
            frame.forces = frame
                .forces
                .map(|values| order.iter().map(|&index| values[index]).collect());
            for values in frame.additional_columns.values_mut() {
                let reordered = order.iter().map(|&index| values[index]).collect();
                *values = reordered;
            }
            frame
        })
        .collect()
}

fn validate_dump_file(file: &LammpsDumpFile) -> crate::Result<()> {
    if file.frames.is_empty() {
        return Err(crate::Error::InvalidInput(
            "LAMMPS dump contains no coordinate frames".to_owned(),
        ));
    }
    let count = file.atom_ids.len();
    if count == 0 {
        return Err(crate::Error::InvalidInput(
            "LAMMPS dump contains no atoms".to_owned(),
        ));
    }
    let mut seen_ids = HashSet::with_capacity(count);
    if file
        .atom_ids
        .iter()
        .any(|id| *id == 0 || !seen_ids.insert(*id))
    {
        return Err(crate::Error::InvalidInput(
            "LAMMPS dump atom IDs must be positive and unique".to_owned(),
        ));
    }
    for (index, frame) in file.frames.iter().enumerate() {
        if frame.atom_ids != file.atom_ids {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} atom IDs do not match the first frame"
            )));
        }
        if frame
            .positions
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} positions must be finite"
            )));
        }
        if frame.n_atoms() != count
            || frame
                .velocities
                .as_ref()
                .is_some_and(|values| values.len() != count)
            || frame
                .forces
                .as_ref()
                .is_some_and(|values| values.len() != count)
            || frame.additional_columns.values().any(|values| {
                values.len() != count || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} has inconsistent atom metadata"
            )));
        }
        if !frame.time.is_finite() {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} time must be finite"
            )));
        }
        if frame
            .velocities
            .as_ref()
            .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
            || frame
                .forces
                .as_ref()
                .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
        {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} velocities and forces must be finite"
            )));
        }
        let bounds = frame.bounds;
        if ![
            bounds.xlo, bounds.xhi, bounds.ylo, bounds.yhi, bounds.zlo, bounds.zhi, bounds.xy,
            bounds.xz, bounds.yz,
        ]
        .iter()
        .all(|value| value.is_finite())
            || bounds.xhi <= bounds.xlo
            || bounds.yhi <= bounds.ylo
            || bounds.zhi <= bounds.zlo
        {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} has invalid box bounds"
            )));
        }
        if frame.dimensions.is_some_and(|dimensions| {
            dimensions.iter().any(|value| !value.is_finite())
                || dimensions[..3].iter().any(|value| *value <= 0.0)
                || dimensions[3..]
                    .iter()
                    .any(|value| !(*value > 0.0 && *value < 180.0))
        }) {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump frame {index} has invalid dimensions"
            )));
        }
        if let Some(dimensions) = frame.dimensions {
            let expected = bounds.dimensions();
            if dimensions.iter().zip(expected).any(|(actual, expected)| {
                (actual - expected).abs() > 1.0e-10 * actual.abs().max(expected.abs()).max(1.0)
            }) {
                return Err(crate::Error::InvalidInput(format!(
                    "LAMMPS dump frame {index} dimensions do not match box bounds"
                )));
            }
        }
    }
    for (name, values) in [
        ("molecule IDs", file.molecule_ids.as_ref().map(Vec::len)),
        ("atom types", file.atom_types.as_ref().map(Vec::len)),
        ("masses", file.masses.as_ref().map(Vec::len)),
        ("elements", file.elements.as_ref().map(Vec::len)),
        ("charges", file.charges.as_ref().map(Vec::len)),
    ] {
        if values.is_some_and(|length| length != count) {
            return Err(crate::Error::InvalidInput(format!(
                "LAMMPS dump {name} metadata does not match atom count"
            )));
        }
    }
    if file.masses.as_ref().is_some_and(|values| {
        values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    }) || file
        .charges
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(crate::Error::InvalidInput(
            "LAMMPS dump masses and charges must be finite; masses must be non-negative".to_owned(),
        ));
    }
    Ok(())
}

/// Errors produced while reading or writing LAMMPS DATA and dump files.
#[derive(Debug)]
pub enum LammpsError {
    Io(io::Error),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for LammpsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(
                    formatter,
                    "LAMMPS DATA parse error on line {line}: {message}"
                )
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid LAMMPS DATA structure: {message}")
            }
        }
    }
}

impl std::error::Error for LammpsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for LammpsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug)]
struct RawDumpAtom {
    line: usize,
    id: usize,
    molecule_id: Option<i64>,
    atom_type: Option<String>,
    mass: Option<f64>,
    element: Option<String>,
    charge: Option<f64>,
    coordinates: [f64; 3],
    velocity: Option<[f64; 3]>,
    force: Option<[f64; 3]>,
    image: Option<[i32; 3]>,
    additional_values: Vec<Option<f64>>,
}

#[derive(Clone, Debug)]
struct DumpMetadata {
    molecule_ids: Option<Vec<i64>>,
    atom_types: Option<Vec<String>>,
    masses: Option<Vec<f64>>,
    elements: Option<Vec<String>>,
    charges: Option<Vec<f64>>,
}

fn parse_dump(input: &str, options: LammpsDumpOptions) -> Result<LammpsDumpFile, LammpsError> {
    let lines = input.lines().collect::<Vec<_>>();
    let mut cursor = 0;
    let mut frames = Vec::new();
    let mut metadata = None;
    let mut atom_ids = None;
    let mut selected_convention = None;
    let mut saw_units = false;

    while let Some((mut line_number, mut marker)) = next_dump_line(&lines, &mut cursor) {
        if marker == "ITEM: UNITS" {
            if saw_units || !frames.is_empty() {
                return Err(parse_error(
                    line_number,
                    "ITEM: UNITS is only allowed once at the start of a dump file",
                ));
            }
            let (_, units) = required_dump_line(&lines, &mut cursor, "units style")?;
            if units.is_empty() {
                return Err(parse_error(
                    line_number,
                    "ITEM: UNITS requires a units style",
                ));
            }
            saw_units = true;
            (line_number, marker) = next_dump_line(&lines, &mut cursor).ok_or_else(|| {
                parse_error(
                    lines.len().max(1),
                    "missing ITEM: TIMESTEP after ITEM: UNITS",
                )
            })?;
        }
        let explicit_time = if marker == "ITEM: TIME" {
            let (time_line, time_text) = required_dump_line(&lines, &mut cursor, "elapsed time")?;
            let time = parse_f64(time_text, time_line, "elapsed time")?;
            if !time.is_finite() {
                return Err(parse_error(time_line, "elapsed time must be finite"));
            }
            (line_number, marker) = next_dump_line(&lines, &mut cursor).ok_or_else(|| {
                parse_error(
                    lines.len().max(1),
                    "missing ITEM: TIMESTEP after ITEM: TIME",
                )
            })?;
            Some(time)
        } else {
            None
        };
        if marker != "ITEM: TIMESTEP" {
            return Err(parse_error(
                line_number,
                format!("expected ITEM: TIMESTEP, found {marker:?}"),
            ));
        }
        let (step_line, step_text) = required_dump_line(&lines, &mut cursor, "timestep value")?;
        let step = parse_nonnegative_usize(step_text, step_line, "timestep")?;
        let mut time = explicit_time.unwrap_or(step as f64);

        let (mut number_line, mut number_marker) =
            required_dump_line(&lines, &mut cursor, "ITEM: NUMBER OF ATOMS")?;
        if number_marker == "ITEM: TIME" {
            if explicit_time.is_some() {
                return Err(parse_error(
                    number_line,
                    "ITEM: TIME appears more than once for a timestep",
                ));
            }
            let (time_line, time_text) = required_dump_line(&lines, &mut cursor, "elapsed time")?;
            time = parse_f64(time_text, time_line, "elapsed time")?;
            if !time.is_finite() {
                return Err(parse_error(time_line, "elapsed time must be finite"));
            }
            (number_line, number_marker) =
                required_dump_line(&lines, &mut cursor, "ITEM: NUMBER OF ATOMS")?;
        }
        if number_marker != "ITEM: NUMBER OF ATOMS" {
            return Err(parse_error(
                number_line,
                format!("expected ITEM: NUMBER OF ATOMS, found {number_marker:?}"),
            ));
        }
        let (count_line, count_text) = required_dump_line(&lines, &mut cursor, "atom count")?;
        let n_atoms = parse_usize(count_text, count_line, "atom count")?;
        if n_atoms == 0 {
            return Err(parse_error(count_line, "atom count must be positive"));
        }

        let (box_line, box_marker) = required_dump_line(&lines, &mut cursor, "ITEM: BOX BOUNDS")?;
        let box_tokens = box_marker.split_whitespace().collect::<Vec<_>>();
        if box_tokens.first() != Some(&"ITEM:")
            || box_tokens.get(1) != Some(&"BOX")
            || box_tokens.get(2) != Some(&"BOUNDS")
        {
            return Err(parse_error(
                box_line,
                format!("expected ITEM: BOX BOUNDS, found {box_marker:?}"),
            ));
        }
        if box_tokens
            .iter()
            .any(|token| token.eq_ignore_ascii_case("abc") || token.eq_ignore_ascii_case("origin"))
        {
            return Err(parse_error(
                box_line,
                "general-triclinic BOX BOUNDS is not supported",
            ));
        }
        let tilt_columns = ["xy", "xz", "yz"]
            .iter()
            .filter(|column| {
                box_marker
                    .split_whitespace()
                    .any(|token| token.eq_ignore_ascii_case(column))
            })
            .count();
        if tilt_columns != 0 && tilt_columns != 3 {
            return Err(parse_error(
                box_line,
                "BOX BOUNDS must include all of xy, xz, and yz for a triclinic box",
            ));
        }
        let triclinic = tilt_columns == 3;
        let mut box_values = [[0.0_f64; 3]; 3];
        for axis_values in &mut box_values {
            let (bounds_line, bounds_text) =
                required_dump_line(&lines, &mut cursor, "box-bound line")?;
            let values = bounds_text.split_whitespace().collect::<Vec<_>>();
            let expected_values = if triclinic { 3 } else { 2 };
            if values.len() != expected_values {
                return Err(parse_error(
                    bounds_line,
                    if triclinic {
                        "triclinic box-bound lines require lower, upper, and tilt"
                    } else {
                        "box-bound lines require lower and upper values"
                    },
                ));
            }
            for column in 0..expected_values {
                axis_values[column] = parse_f64(
                    values[column],
                    bounds_line,
                    if column == 0 {
                        "box lower bound"
                    } else if column == 1 {
                        "box upper bound"
                    } else {
                        "box tilt"
                    },
                )?;
            }
        }
        let bounds = if triclinic {
            let xlo_bound = box_values[0][0];
            let xhi_bound = box_values[0][1];
            let xy = box_values[0][2];
            let ylo_bound = box_values[1][0];
            let yhi_bound = box_values[1][1];
            let xz = box_values[1][2];
            let zlo = box_values[2][0];
            let zhi = box_values[2][1];
            let yz = box_values[2][2];
            LammpsBox {
                xlo: xlo_bound - 0.0_f64.min(xy).min(xz).min(xy + xz),
                xhi: xhi_bound - 0.0_f64.max(xy).max(xz).max(xy + xz),
                ylo: ylo_bound - 0.0_f64.min(yz),
                yhi: yhi_bound - 0.0_f64.max(yz),
                zlo,
                zhi,
                xy,
                xz,
                yz,
            }
        } else {
            LammpsBox::orthogonal(
                box_values[0][0],
                box_values[0][1],
                box_values[1][0],
                box_values[1][1],
                box_values[2][0],
                box_values[2][1],
            )
        };
        validate_dump_bounds(bounds, box_line)?;
        let dimensions = bounds.dimensions();
        if dimensions
            .iter()
            .enumerate()
            .any(|(index, value)| !value.is_finite() || (index < 3 && *value <= 0.0))
        {
            return Err(parse_error(
                box_line,
                "box dimensions must be finite and positive",
            ));
        }

        let (atoms_line, atoms_marker) = required_dump_line(&lines, &mut cursor, "ITEM: ATOMS")?;
        let atom_tokens = atoms_marker.split_whitespace().collect::<Vec<_>>();
        if atom_tokens.first() != Some(&"ITEM:") || atom_tokens.get(1) != Some(&"ATOMS") {
            return Err(parse_error(
                atoms_line,
                format!("expected ITEM: ATOMS, found {atoms_marker:?}"),
            ));
        }
        let attrs = atoms_marker
            .split_whitespace()
            .skip(2)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if attrs.is_empty() {
            return Err(parse_error(atoms_line, "ITEM: ATOMS has no column names"));
        }
        let mut attr_to_col = HashMap::with_capacity(attrs.len());
        for (index, attr) in attrs.iter().enumerate() {
            if attr_to_col.insert(attr.as_str(), index).is_some() {
                return Err(parse_error(
                    atoms_line,
                    format!("duplicate ITEM: ATOMS column {attr:?}"),
                ));
            }
        }
        let convention = if let Some(convention) = selected_convention {
            convention
        } else {
            let convention = match options.coordinate_convention {
                LammpsCoordinateConvention::Auto => [
                    LammpsCoordinateConvention::Unscaled,
                    LammpsCoordinateConvention::Scaled,
                    LammpsCoordinateConvention::Unwrapped,
                    LammpsCoordinateConvention::ScaledUnwrapped,
                ]
                .into_iter()
                .find(|candidate| {
                    candidate
                        .columns()
                        .iter()
                        .all(|column| attr_to_col.contains_key(column))
                })
                .ok_or_else(|| {
                    LammpsError::InvalidStructure(
                        "no coordinate information detected in ITEM: ATOMS".to_owned(),
                    )
                })?,
                explicit => explicit,
            };
            selected_convention = Some(convention);
            convention
        };
        let coordinate_columns = convention.columns();
        if !coordinate_columns
            .iter()
            .all(|column| attr_to_col.contains_key(column))
        {
            return Err(LammpsError::InvalidStructure(format!(
                "no coordinates following convention {convention} found in timestep"
            )));
        }
        let velocity_columns = ["vx", "vy", "vz"];
        let force_columns = ["fx", "fy", "fz"];
        let has_any_velocity = velocity_columns
            .iter()
            .any(|column| attr_to_col.contains_key(column));
        let has_all_velocity = velocity_columns
            .iter()
            .all(|column| attr_to_col.contains_key(column));
        if has_any_velocity && !has_all_velocity {
            return Err(parse_error(
                atoms_line,
                "velocity columns must include vx, vy, and vz",
            ));
        }
        let has_any_force = force_columns
            .iter()
            .any(|column| attr_to_col.contains_key(column));
        let has_all_force = force_columns
            .iter()
            .all(|column| attr_to_col.contains_key(column));
        if has_any_force && !has_all_force {
            return Err(parse_error(
                atoms_line,
                "force columns must include fx, fy, and fz",
            ));
        }
        let image_columns = ["ix", "iy", "iz"];
        let has_all_image = image_columns
            .iter()
            .all(|column| attr_to_col.contains_key(column));
        if options.unwrap_images && !convention.is_unwrapped() && !has_all_image {
            return Err(LammpsError::InvalidStructure(
                "trajectory must have image flags ix, iy, and iz to unwrap".to_owned(),
            ));
        }
        let has_any_image = image_columns
            .iter()
            .any(|column| attr_to_col.contains_key(column));
        if has_any_image && !has_all_image {
            return Err(parse_error(
                atoms_line,
                "image columns must include ix, iy, and iz",
            ));
        }
        let id_column = attr_to_col.get("id").copied().ok_or_else(|| {
            LammpsError::InvalidStructure("no id column found in dump file".to_owned())
        })?;
        let additional_names = attrs
            .iter()
            .filter(|name| !is_dump_known_column(name))
            .cloned()
            .collect::<Vec<_>>();

        let required_columns = attrs.len();
        if n_atoms > lines.len().saturating_sub(cursor) {
            return Err(parse_error(
                count_line,
                format!("atom count {n_atoms} exceeds remaining dump rows"),
            ));
        }
        let mut rows = Vec::with_capacity(n_atoms);
        for row_index in 0..n_atoms {
            let (row_line, row_text) = lines.get(cursor).copied().map_or_else(
                || {
                    Err(parse_error(
                        lines.len().max(1),
                        format!("expected {n_atoms} atom rows, found {row_index}"),
                    ))
                },
                |line| {
                    cursor += 1;
                    if line.trim().is_empty() {
                        Err(parse_error(cursor, "empty atom row"))
                    } else {
                        Ok((cursor, line.trim()))
                    }
                },
            )?;
            let fields = row_text.split_whitespace().collect::<Vec<_>>();
            if fields.len() != required_columns {
                return Err(parse_error(
                    row_line,
                    format!(
                        "atom row has {} fields but ITEM: ATOMS declares {required_columns}",
                        fields.len()
                    ),
                ));
            }
            let id = parse_usize(fields[id_column], row_line, "atom id")?;
            if id == 0 {
                return Err(parse_error(row_line, "atom id must be positive"));
            }
            let mut coordinates = [0.0; 3];
            for (axis, column) in coordinate_columns.iter().enumerate() {
                coordinates[axis] = parse_f64(fields[attr_to_col[column]], row_line, "coordinate")?;
                if !coordinates[axis].is_finite() {
                    return Err(parse_error(row_line, "coordinates must be finite"));
                }
            }
            let velocity = if has_all_velocity {
                Some([
                    parse_f64(fields[attr_to_col["vx"]], row_line, "velocity")?,
                    parse_f64(fields[attr_to_col["vy"]], row_line, "velocity")?,
                    parse_f64(fields[attr_to_col["vz"]], row_line, "velocity")?,
                ])
            } else {
                None
            };
            if velocity.is_some_and(|values| values.iter().any(|value| !value.is_finite())) {
                return Err(parse_error(row_line, "velocities must be finite"));
            }
            let force = if has_all_force {
                Some([
                    parse_f64(fields[attr_to_col["fx"]], row_line, "force")?,
                    parse_f64(fields[attr_to_col["fy"]], row_line, "force")?,
                    parse_f64(fields[attr_to_col["fz"]], row_line, "force")?,
                ])
            } else {
                None
            };
            if force.is_some_and(|values| values.iter().any(|value| !value.is_finite())) {
                return Err(parse_error(row_line, "forces must be finite"));
            }
            let image = if has_all_image {
                Some([
                    parse_i32(fields[attr_to_col["ix"]], row_line, "image")?,
                    parse_i32(fields[attr_to_col["iy"]], row_line, "image")?,
                    parse_i32(fields[attr_to_col["iz"]], row_line, "image")?,
                ])
            } else {
                None
            };
            let molecule_id = attr_to_col
                .get("mol")
                .map(|column| parse_i64(fields[*column], row_line, "molecule id"))
                .transpose()?;
            let atom_type = attr_to_col
                .get("type")
                .map(|column| fields[*column].to_owned());
            let mass = attr_to_col
                .get("mass")
                .map(|column| parse_f64(fields[*column], row_line, "mass"))
                .transpose()?;
            if mass.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(parse_error(
                    row_line,
                    "mass must be finite and non-negative",
                ));
            }
            let element = attr_to_col
                .get("element")
                .map(|column| normalize_dump_element(fields[*column]));
            let charge = attr_to_col
                .get("q")
                .map(|column| parse_f64(fields[*column], row_line, "charge"))
                .transpose()?;
            if charge.is_some_and(|value| !value.is_finite()) {
                return Err(parse_error(row_line, "charge must be finite"));
            }
            let additional_values = additional_names
                .iter()
                .map(|name| {
                    fields[attr_to_col[name.as_str()]]
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite())
                })
                .collect();
            rows.push(RawDumpAtom {
                line: row_line,
                id,
                molecule_id,
                atom_type,
                mass,
                element,
                charge,
                coordinates,
                velocity,
                force,
                image,
                additional_values,
            });
        }
        rows.sort_unstable_by_key(|row| row.id);
        let mut frame_ids = Vec::with_capacity(n_atoms);
        for row in &rows {
            if frame_ids.last().is_some_and(|last| *last == row.id) {
                return Err(LammpsError::InvalidStructure(format!(
                    "duplicate atom id {} in timestep {step}",
                    row.id
                )));
            }
            frame_ids.push(row.id);
        }
        if let Some(expected_ids) = &atom_ids {
            if expected_ids != &frame_ids {
                return Err(LammpsError::InvalidStructure(
                    "atom IDs changed between dump frames".to_owned(),
                ));
            }
        } else {
            atom_ids = Some(frame_ids.clone());
        }
        let scaled_positions = if convention.is_scaled() {
            let raw_positions = rows.iter().map(|row| row.coordinates).collect::<Vec<_>>();
            Some(
                crate::distances::transform_s_to_r(&raw_positions, dimensions).map_err(
                    |error| {
                        LammpsError::InvalidStructure(format!(
                            "cannot convert scaled coordinates: {error}"
                        ))
                    },
                )?,
            )
        } else {
            None
        };
        let mut positions = Vec::with_capacity(n_atoms);
        let mut velocities = has_all_velocity.then(|| Vec::with_capacity(n_atoms));
        let mut forces = has_all_force.then(|| Vec::with_capacity(n_atoms));
        let mut additional_columns = additional_names
            .iter()
            .map(|name| (name.clone(), Vec::with_capacity(n_atoms)))
            .collect::<BTreeMap<_, _>>();
        for (row_index, row) in rows.iter().enumerate() {
            let mut position = scaled_positions
                .as_ref()
                .map_or(row.coordinates, |positions| positions[row_index]);
            if options.unwrap_images && !convention.is_unwrapped() {
                let image = row.image.expect("image columns validated");
                let shift = bounds.image_shift(image);
                for axis in 0..3 {
                    position[axis] += shift[axis];
                }
            }
            position[0] -= bounds.xlo;
            position[1] -= bounds.ylo;
            position[2] -= bounds.zlo;
            if position.iter().any(|value| !value.is_finite()) {
                return Err(parse_error(row.line, "final positions must be finite"));
            }
            positions.push(position);
            if let Some(values) = velocities.as_mut() {
                values.push(row.velocity.expect("velocity columns validated"));
            }
            if let Some(values) = forces.as_mut() {
                values.push(row.force.expect("force columns validated"));
            }
            for (name, value) in additional_names.iter().zip(&row.additional_values) {
                if let Some(value) = value
                    && let Some(values) = additional_columns.get_mut(name)
                {
                    values.push(*value);
                }
            }
        }
        additional_columns.retain(|_, values| values.len() == n_atoms);
        let frame = LammpsDumpFrame {
            step,
            time,
            atom_ids: frame_ids,
            positions,
            velocities,
            forces,
            dimensions: Some(dimensions),
            bounds,
            additional_columns,
        };
        if metadata.is_none() {
            metadata = Some(DumpMetadata {
                molecule_ids: attr_to_col.contains_key("mol").then(|| {
                    rows.iter()
                        .map(|row| row.molecule_id.expect("molecule column validated"))
                        .collect()
                }),
                atom_types: attr_to_col.contains_key("type").then(|| {
                    rows.iter()
                        .map(|row| row.atom_type.clone().expect("type column validated"))
                        .collect()
                }),
                masses: attr_to_col.contains_key("mass").then(|| {
                    rows.iter()
                        .map(|row| row.mass.expect("mass column validated"))
                        .collect()
                }),
                elements: attr_to_col.contains_key("element").then(|| {
                    rows.iter()
                        .map(|row| row.element.clone().expect("element column validated"))
                        .collect()
                }),
                charges: attr_to_col.contains_key("q").then(|| {
                    rows.iter()
                        .map(|row| row.charge.expect("charge column validated"))
                        .collect()
                }),
            });
        }
        frames.push(frame);
    }
    let atom_ids = atom_ids.ok_or_else(|| {
        LammpsError::InvalidStructure("LAMMPS dump contains no coordinate frames".to_owned())
    })?;
    let metadata = metadata.expect("metadata exists with atom IDs");
    Ok(LammpsDumpFile {
        frames,
        atom_ids,
        molecule_ids: metadata.molecule_ids,
        atom_types: metadata.atom_types,
        masses: metadata.masses,
        elements: metadata.elements,
        charges: metadata.charges,
        coordinate_convention: selected_convention.expect("convention exists with frames"),
    })
}

fn next_dump_line<'a>(lines: &'a [&str], cursor: &mut usize) -> Option<(usize, &'a str)> {
    while let Some(line) = lines.get(*cursor) {
        let line_number = *cursor + 1;
        *cursor += 1;
        if !line.trim().is_empty() {
            return Some((line_number, line.trim()));
        }
    }
    None
}

fn required_dump_line<'a>(
    lines: &'a [&str],
    cursor: &mut usize,
    what: &str,
) -> Result<(usize, &'a str), LammpsError> {
    next_dump_line(lines, cursor)
        .ok_or_else(|| parse_error(lines.len().max(1), format!("missing {what}")))
}

fn parse_nonnegative_usize(value: &str, line: usize, field: &str) -> Result<usize, LammpsError> {
    let value = value
        .parse::<i64>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))?;
    usize::try_from(value).map_err(|_| parse_error(line, format!("{field} must be non-negative")))
}

fn validate_dump_bounds(bounds: LammpsBox, line: usize) -> Result<(), LammpsError> {
    let values = [
        bounds.xlo, bounds.xhi, bounds.ylo, bounds.yhi, bounds.zlo, bounds.zhi, bounds.xy,
        bounds.xz, bounds.yz,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(parse_error(line, "box bounds must be finite"));
    }
    if bounds.xhi <= bounds.xlo || bounds.yhi <= bounds.ylo || bounds.zhi <= bounds.zlo {
        return Err(parse_error(
            line,
            "box upper bounds must exceed lower bounds",
        ));
    }
    Ok(())
}

fn is_dump_known_column(name: &str) -> bool {
    matches!(
        name,
        "id" | "mol"
            | "type"
            | "mass"
            | "element"
            | "q"
            | "vx"
            | "vy"
            | "vz"
            | "fx"
            | "fy"
            | "fz"
            | "ix"
            | "iy"
            | "iz"
            | "x"
            | "y"
            | "z"
            | "xs"
            | "ys"
            | "zs"
            | "xu"
            | "yu"
            | "zu"
            | "xsu"
            | "ysu"
            | "zsu"
    )
}

fn normalize_dump_element(value: &str) -> String {
    let trimmed = value.trim();
    crate::guesser::ELEMENT_MASSES
        .iter()
        .find_map(|(symbol, _)| {
            symbol
                .eq_ignore_ascii_case(trimmed)
                .then_some((*symbol).to_owned())
        })
        .unwrap_or_default()
}

fn parse_data(input: &str, explicit_style: Option<&str>) -> Result<LammpsData, LammpsError> {
    let mut lines = input.lines().enumerate();
    let (_, title_line) = lines
        .next()
        .ok_or_else(|| parse_error(1, "missing title line"))?;
    let title = title_line.trim_end().to_owned();
    let mut header = Vec::new();
    let mut sections: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    let mut section_name: Option<String> = None;
    let mut section_style = explicit_style
        .map(str::trim)
        .filter(|style| !style.is_empty());

    for (line_index, raw_line) in lines {
        let line_number = line_index + 1;
        let (without_comment, comment) = raw_line
            .split_once('#')
            .map_or((raw_line, None), |(a, b)| (a, Some(b.trim())));
        let clean = without_comment.trim();
        if clean.is_empty() {
            continue;
        }
        if let Some(name) = section_title(clean) {
            section_name = Some(name.to_owned());
            if name == "Atoms" && section_style.is_none() {
                section_style = comment
                    .and_then(|text| text.split_whitespace().next())
                    .filter(|style| is_atom_style_descriptor(style));
            }
            sections.entry(name.to_owned()).or_default();
            continue;
        }
        if section_name.is_some() {
            // DATA section entries start with a numeric ID.  Unknown section
            // titles are still recognized by this guard and safely skipped.
            if clean
                .split_whitespace()
                .next()
                .is_some_and(|token| token.parse::<i64>().is_err())
            {
                section_name = None;
            }
        }
        if let Some(name) = &section_name {
            sections
                .entry(name.clone())
                .or_default()
                .push((line_number, clean.to_owned()));
        } else {
            header.push((line_number, clean.to_owned()));
        }
    }

    let counts = parse_header(&header)?;
    let bounds = parse_bounds(&header)?;
    let atom_lines = sections
        .get("Atoms")
        .ok_or_else(|| parse_error(1, "data file is missing an Atoms section"))?;
    let atom_style = section_style.unwrap_or_else(|| infer_style_from_atom_line(atom_lines));
    let style_fields = style_fields(atom_style)?;
    let mut atoms = atom_lines
        .iter()
        .map(|(line, text)| parse_atom_line(text, *line, &style_fields))
        .collect::<Result<Vec<_>, _>>()?;
    atoms.sort_by_key(|atom| atom.id);
    let mut atom_ids = HashSet::with_capacity(atoms.len());
    for atom in &atoms {
        if !atom_ids.insert(atom.id) {
            return Err(parse_error(0, format!("duplicate atom ID {}", atom.id)));
        }
    }

    if let Some(expected) = counts.get("atom types")
        && atoms.iter().any(|atom| atom.atom_type > *expected)
    {
        return Err(parse_error(
            0,
            format!("Atoms contains a type ID greater than declared {expected} atom types"),
        ));
    }
    if let Some(expected) = counts.get("atoms")
        && *expected != atoms.len()
    {
        return Err(parse_error(
            0,
            format!(
                "header declares {expected} atoms but Atoms contains {}",
                atoms.len()
            ),
        ));
    }

    let mut masses = BTreeMap::new();
    if let Some(mass_lines) = sections.get("Masses") {
        for (line, text) in mass_lines {
            let tokens = text.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 2 {
                return Err(parse_error(*line, "Masses entry requires type and mass"));
            }
            let atom_type = parse_usize(tokens[0], *line, "atom type")?;
            let mass = parse_f64(tokens[1], *line, "mass")?;
            if !mass.is_finite() || mass < 0.0 {
                return Err(parse_error(*line, "mass must be finite and non-negative"));
            }
            if let Some(expected) = counts.get("atom types") {
                if atom_type == 0 || atom_type > *expected {
                    return Err(parse_error(
                        *line,
                        format!(
                            "Masses contains type {atom_type} outside declared range 1..={expected}"
                        ),
                    ));
                }
            } else if atom_type == 0 {
                return Err(parse_error(*line, "Masses atom type IDs must be positive"));
            }
            if masses.insert(atom_type, mass).is_some() {
                return Err(parse_error(*line, "duplicate Masses atom type"));
            }
        }
    }

    let bonds = parse_bonds(sections.get("Bonds"), &atom_ids, counts.get("bonds"))?;
    if let Some(expected) = counts.get("bond types")
        && bonds.iter().any(|bond| bond.bond_type > *expected)
    {
        return Err(parse_error(
            0,
            format!("Bonds contains a type ID greater than declared {expected} bond types"),
        ));
    }
    let velocities = parse_velocities(sections.get("Velocities"), &atoms)?;
    Ok(LammpsData {
        title,
        atoms,
        bonds,
        masses,
        velocities,
        bounds,
        atom_style: atom_style.to_owned(),
    })
}

fn section_title(line: &str) -> Option<&str> {
    let first = line.split_whitespace().next()?;
    // Coefficient and topology sections that are not modeled still need to
    // delimit the preceding section.
    matches!(
        first,
        "Atoms"
            | "Masses"
            | "Velocities"
            | "Bonds"
            | "Angles"
            | "Dihedrals"
            | "Impropers"
            | "Ellipsoids"
            | "Lines"
            | "Triangles"
            | "Bodies"
            | "Pair"
            | "PairIJ"
            | "Bond"
            | "Angle"
            | "Dihedral"
            | "Improper"
    )
    .then_some(first)
}

fn parse_header(lines: &[(usize, String)]) -> Result<HashMap<String, usize>, LammpsError> {
    let mut counts = HashMap::new();
    for (line, text) in lines {
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        let key = match tokens.as_slice() {
            [_, key] => Some((*key).to_owned()),
            [_, first, second] => Some(format!("{first} {second}")),
            _ => None,
        };
        let Some(key) = key else {
            continue;
        };
        let recognized = matches!(
            key.as_str(),
            "atoms"
                | "bonds"
                | "angles"
                | "dihedrals"
                | "impropers"
                | "ellipsoids"
                | "lines"
                | "triangles"
                | "bodies"
                | "atom types"
                | "bond types"
                | "angle types"
                | "dihedral types"
                | "improper types"
        );
        if !recognized {
            continue;
        }
        let value = parse_usize(tokens[0], *line, &format!("{key} count"))?;
        if counts.insert(key.clone(), value).is_some() {
            return Err(parse_error(
                *line,
                format!("duplicate header count for {key}"),
            ));
        }
    }
    Ok(counts)
}

// Header values are parsed from raw lines so floating-point bounds are not
// lossy-converted through integer map keys.
fn parse_bounds(header: &[(usize, String)]) -> Result<Option<LammpsBox>, LammpsError> {
    let mut values: HashMap<&str, f64> = HashMap::new();
    let mut saw_bound = false;
    for (line, text) in header {
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        if tokens.len() >= 6 && tokens[3..6] == ["xy", "xz", "yz"] {
            saw_bound = true;
            values.insert("xy", parse_f64(tokens[0], *line, "xy tilt")?);
            values.insert("xz", parse_f64(tokens[1], *line, "xz tilt")?);
            values.insert("yz", parse_f64(tokens[2], *line, "yz tilt")?);
        } else if tokens.len() >= 4 {
            match (tokens[2], tokens[3]) {
                ("xlo", "xhi") => {
                    saw_bound = true;
                    values.insert("xlo", parse_f64(tokens[0], *line, "xlo bound")?);
                    values.insert("xhi", parse_f64(tokens[1], *line, "xhi bound")?);
                }
                ("ylo", "yhi") => {
                    saw_bound = true;
                    values.insert("ylo", parse_f64(tokens[0], *line, "ylo bound")?);
                    values.insert("yhi", parse_f64(tokens[1], *line, "yhi bound")?);
                }
                ("zlo", "zhi") => {
                    saw_bound = true;
                    values.insert("zlo", parse_f64(tokens[0], *line, "zlo bound")?);
                    values.insert("zhi", parse_f64(tokens[1], *line, "zhi bound")?);
                }
                _ => {}
            }
        }
    }
    let Some((&_, _)) = values.get_key_value("xlo") else {
        if saw_bound {
            return Err(parse_error(0, "box bounds are incomplete"));
        }
        return Ok(None);
    };
    let required = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"];
    if required.iter().any(|key| !values.contains_key(key)) {
        return Err(parse_error(0, "box bounds are incomplete"));
    }
    let bounds = LammpsBox {
        xlo: values["xlo"],
        xhi: values["xhi"],
        ylo: values["ylo"],
        yhi: values["yhi"],
        zlo: values["zlo"],
        zhi: values["zhi"],
        xy: values.get("xy").copied().unwrap_or(0.0),
        xz: values.get("xz").copied().unwrap_or(0.0),
        yz: values.get("yz").copied().unwrap_or(0.0),
    };
    if [
        bounds.xlo, bounds.xhi, bounds.ylo, bounds.yhi, bounds.zlo, bounds.zhi, bounds.xy,
        bounds.xz, bounds.yz,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || bounds.xhi <= bounds.xlo
        || bounds.yhi <= bounds.ylo
        || bounds.zhi <= bounds.zlo
    {
        return Err(parse_error(0, "box bounds must be finite with hi > lo"));
    }
    Ok(Some(bounds))
}

fn style_fields(style: &str) -> Result<Vec<String>, LammpsError> {
    let normalized = style.trim().to_ascii_lowercase();
    let fields: Vec<String> = match normalized.as_str() {
        "atomic" => vec!["id", "type", "x", "y", "z"]
            .into_iter()
            .map(String::from)
            .collect(),
        "molecular" => vec!["id", "mol", "type", "x", "y", "z"]
            .into_iter()
            .map(String::from)
            .collect(),
        "charge" => vec!["id", "type", "q", "x", "y", "z"]
            .into_iter()
            .map(String::from)
            .collect(),
        "full" => vec!["id", "mol", "type", "q", "x", "y", "z"]
            .into_iter()
            .map(String::from)
            .collect(),
        _ => style
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect::<Vec<String>>(),
    };
    let required = ["id", "type", "x", "y", "z"];
    if fields.is_empty()
        || required
            .iter()
            .any(|field| !fields.iter().any(|candidate| candidate == field))
    {
        return Err(LammpsError::InvalidStructure(format!(
            "atom style {style:?} must contain id, type, x, y, and z fields"
        )));
    }
    Ok(fields)
}

fn is_atom_style_descriptor(style: &str) -> bool {
    let normalized = style.trim().to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "atomic" | "molecular" | "charge" | "full"
    ) || normalized.split_whitespace().count() >= 5
        && ["id", "type", "x", "y", "z"].iter().all(|required| {
            normalized
                .split_whitespace()
                .any(|field| field == *required)
        })
}

fn infer_style_from_atom_line(lines: &[(usize, String)]) -> &'static str {
    match lines
        .first()
        .map(|(_, line)| line.split_whitespace().count())
        .unwrap_or(0)
    {
        5 | 8 => "atomic",
        6 | 9 => "molecular",
        7 | 10 => "full",
        _ => "full",
    }
}

fn parse_atom_line(text: &str, line: usize, fields: &[String]) -> Result<LammpsAtom, LammpsError> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < fields.len() {
        return Err(parse_error(
            line,
            format!(
                "Atoms entry has {} fields; expected at least {}",
                tokens.len(),
                fields.len()
            ),
        ));
    }
    let index = |name: &str| fields.iter().position(|field| field == name);
    let id = parse_usize(
        tokens[index("id").expect("validated style")],
        line,
        "atom ID",
    )?;
    let atom_type = parse_usize(
        tokens[index("type").expect("validated style")],
        line,
        "atom type",
    )?;
    if id == 0 {
        return Err(parse_error(line, "atom IDs must be positive"));
    }
    if atom_type == 0 {
        return Err(parse_error(line, "atom type IDs must be positive"));
    }
    let position = [
        parse_f64(
            tokens[index("x").expect("validated style")],
            line,
            "x coordinate",
        )?,
        parse_f64(
            tokens[index("y").expect("validated style")],
            line,
            "y coordinate",
        )?,
        parse_f64(
            tokens[index("z").expect("validated style")],
            line,
            "z coordinate",
        )?,
    ];
    if position.iter().any(|value| !value.is_finite()) {
        return Err(parse_error(line, "coordinates must be finite"));
    }
    let molecule_id = index("mol")
        .or_else(|| index("molecule"))
        .or_else(|| index("resid"))
        .map(|at| parse_i64(tokens[at], line, "molecule/residue ID"))
        .transpose()?;
    let charge = index("q")
        .or_else(|| index("charge"))
        .map(|at| parse_f64(tokens[at], line, "charge"))
        .transpose()?;
    if charge.is_some_and(|value| !value.is_finite()) {
        return Err(parse_error(line, "charge must be finite"));
    }
    let image = if tokens.len() >= fields.len() + 3 {
        let base = fields.len();
        Some([
            parse_i32(tokens[base], line, "x image flag")?,
            parse_i32(tokens[base + 1], line, "y image flag")?,
            parse_i32(tokens[base + 2], line, "z image flag")?,
        ])
    } else {
        None
    };
    Ok(LammpsAtom {
        id,
        molecule_id,
        atom_type,
        charge,
        position,
        image,
    })
}

fn parse_bonds(
    lines: Option<&Vec<(usize, String)>>,
    atom_ids: &HashSet<usize>,
    expected: Option<&usize>,
) -> Result<Vec<LammpsBond>, LammpsError> {
    let mut bonds = Vec::new();
    let mut bond_ids = HashSet::new();
    if let Some(lines) = lines {
        for (line, text) in lines {
            let tokens = text.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 4 {
                return Err(parse_error(
                    *line,
                    "Bonds entry requires id, type, and two atoms",
                ));
            }
            let bond = LammpsBond {
                id: parse_usize(tokens[0], *line, "bond ID")?,
                bond_type: parse_usize(tokens[1], *line, "bond type")?,
                atom1: parse_usize(tokens[2], *line, "first atom ID")?,
                atom2: parse_usize(tokens[3], *line, "second atom ID")?,
            };
            if bond.id == 0 || !bond_ids.insert(bond.id) {
                return Err(parse_error(*line, "bond IDs must be positive and unique"));
            }
            if bond.bond_type == 0 || bond.atom1 == bond.atom2 {
                return Err(parse_error(
                    *line,
                    "bond type must be positive and atoms distinct",
                ));
            }
            if !atom_ids.contains(&bond.atom1) || !atom_ids.contains(&bond.atom2) {
                return Err(parse_error(*line, "bond references an unknown atom ID"));
            }
            bonds.push(bond);
        }
    }
    if let Some(expected) = expected
        && *expected != bonds.len()
    {
        return Err(parse_error(
            0,
            format!(
                "header declares {expected} bonds but Bonds contains {}",
                bonds.len()
            ),
        ));
    }
    bonds.sort_by_key(|bond| bond.id);
    Ok(bonds)
}

fn parse_velocities(
    lines: Option<&Vec<(usize, String)>>,
    atoms: &[LammpsAtom],
) -> Result<Option<Vec<[f64; 3]>>, LammpsError> {
    let Some(lines) = lines else {
        return Ok(None);
    };
    let mapping = atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| (atom.id, index))
        .collect::<HashMap<_, _>>();
    let mut values = vec![[0.0; 3]; atoms.len()];
    let mut seen = HashSet::new();
    for (line, text) in lines {
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 4 {
            return Err(parse_error(
                *line,
                "Velocities entry requires atom ID and three values",
            ));
        }
        let id = parse_usize(tokens[0], *line, "velocity atom ID")?;
        let index = mapping
            .get(&id)
            .copied()
            .ok_or_else(|| parse_error(*line, "velocity references an unknown atom ID"))?;
        if !seen.insert(id) {
            return Err(parse_error(*line, "duplicate velocity atom ID"));
        }
        values[index] = [
            parse_f64(tokens[1], *line, "x velocity")?,
            parse_f64(tokens[2], *line, "y velocity")?,
            parse_f64(tokens[3], *line, "z velocity")?,
        ];
        if values[index].iter().any(|value| !value.is_finite()) {
            return Err(parse_error(*line, "velocities must be finite"));
        }
    }
    if seen.len() != atoms.len() {
        return Err(parse_error(
            0,
            format!(
                "Velocities contains {} entries; expected {}",
                seen.len(),
                atoms.len()
            ),
        ));
    }
    Ok(Some(values))
}

fn atom_field(atom: &LammpsAtom, field: &str) -> Result<String, LammpsError> {
    let value = match field {
        "id" => atom.id.to_string(),
        "mol" | "resid" => atom.molecule_id.unwrap_or(1).to_string(),
        "type" => atom.atom_type.to_string(),
        "q" | "charge" => format!("{:.16e}", atom.charge.unwrap_or(0.0)),
        "x" => format!("{:.16e}", atom.position[0]),
        "y" => format!("{:.16e}", atom.position[1]),
        "z" => format!("{:.16e}", atom.position[2]),
        other => {
            return Err(LammpsError::InvalidStructure(format!(
                "cannot write unsupported atom-style field {other:?}"
            )));
        }
    };
    Ok(value)
}

fn validate_data(data: &LammpsData) -> Result<(), LammpsError> {
    if data.title.contains(['\n', '\r']) {
        return Err(LammpsError::InvalidStructure(
            "title must be a single line".to_owned(),
        ));
    }
    style_fields(&data.atom_style)?;
    let mut atom_ids = HashSet::new();
    for atom in &data.atoms {
        if atom.id == 0 || !atom_ids.insert(atom.id) || atom.atom_type == 0 {
            return Err(LammpsError::InvalidStructure(
                "atom IDs and types must be positive and IDs unique".to_owned(),
            ));
        }
        if atom
            .position
            .iter()
            .chain(atom.charge.iter())
            .any(|value| !value.is_finite())
        {
            return Err(LammpsError::InvalidStructure(
                "atom coordinates and charges must be finite".to_owned(),
            ));
        }
    }
    let mut bond_ids = HashSet::new();
    for bond in &data.bonds {
        if bond.id == 0
            || !bond_ids.insert(bond.id)
            || bond.bond_type == 0
            || bond.atom1 == bond.atom2
            || !atom_ids.contains(&bond.atom1)
            || !atom_ids.contains(&bond.atom2)
        {
            return Err(LammpsError::InvalidStructure(
                "bond IDs/types must be positive and reference distinct known atoms".to_owned(),
            ));
        }
    }
    for (&atom_type, &mass) in &data.masses {
        if atom_type == 0 || !mass.is_finite() || mass < 0.0 {
            return Err(LammpsError::InvalidStructure(
                "mass type IDs must be positive and masses finite and non-negative".to_owned(),
            ));
        }
    }
    if let Some(velocities) = &data.velocities
        && (velocities.len() != data.atoms.len()
            || velocities
                .iter()
                .flat_map(|velocity| velocity.iter())
                .any(|value| !value.is_finite()))
    {
        return Err(LammpsError::InvalidStructure(
            "velocities must match atom count and be finite".to_owned(),
        ));
    }
    if let Some(bounds) = data.bounds
        && ([
            bounds.xlo, bounds.xhi, bounds.ylo, bounds.yhi, bounds.zlo, bounds.zhi, bounds.xy,
            bounds.xz, bounds.yz,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || bounds.xhi <= bounds.xlo
            || bounds.yhi <= bounds.ylo
            || bounds.zhi <= bounds.zlo)
    {
        return Err(LammpsError::InvalidStructure(
            "box bounds must be finite with hi > lo".to_owned(),
        ));
    }
    Ok(())
}

fn parse_usize(value: &str, line: usize, field: &str) -> Result<usize, LammpsError> {
    value
        .parse::<usize>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))
}

fn parse_i64(value: &str, line: usize, field: &str) -> Result<i64, LammpsError> {
    value
        .parse::<i64>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))
}

fn parse_i32(value: &str, line: usize, field: &str) -> Result<i32, LammpsError> {
    value
        .parse::<i32>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))
}

fn parse_f64(value: &str, line: usize, field: &str) -> Result<f64, LammpsError> {
    value
        .parse::<f64>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))
}

fn parse_error(line: usize, message: impl Into<String>) -> LammpsError {
    LammpsError::Parse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::{Cursor, Write};

    const MINI: &str = concat!(
        "LAMMPS minimal data file\n\n",
        "2 atoms\n",
        "1 bonds\n",
        "1 atom types\n",
        "1 bond types\n\n",
        "0.0 10.0 xlo xhi\n",
        "-2.0 8.0 ylo yhi\n",
        "1.0 11.0 zlo zhi\n\n",
        "Masses\n\n",
        "1 12.011\n\n",
        "Atoms # full\n\n",
        "10 2 1 -0.2 1.0 2.0 3.0\n",
        "3 2 1 0.2 4.0 5.0 6.0\n\n",
        "Velocities\n\n",
        "10 0.1 0.2 0.3\n",
        "3 -0.1 -0.2 -0.3\n\n",
        "Bonds\n\n",
        "1 1 10 3\n",
    );

    #[test]
    fn parses_sections_noncontiguous_ids_and_box() {
        let data = LammpsData::from_str(MINI).expect("valid DATA");
        assert_eq!(data.title, "LAMMPS minimal data file");
        assert_eq!(
            data.atoms.iter().map(|atom| atom.id).collect::<Vec<_>>(),
            [3, 10]
        );
        assert_eq!(data.atoms[0].molecule_id, Some(2));
        assert_eq!(data.atoms[1].charge, Some(-0.2));
        assert_eq!(data.bonds[0].atom1, 10);
        assert_eq!(data.velocities.as_ref().unwrap()[0], [-0.1, -0.2, -0.3]);
        assert_eq!(
            data.dimensions(),
            Some([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        );
    }

    #[test]
    fn parses_custom_atomic_style_and_triclinic_tilts() {
        let input = concat!(
            "custom\n\n1 atoms\n1 atom types\n\n",
            "0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n",
            "1.0 -0.5 0.25 xy xz yz\n\n",
            "Atoms\n\n7 1 1.25 2.5 3.75\n",
        );
        let data = LammpsData::from_str_with_atom_style(input, "id type x y z").unwrap();
        assert_eq!(data.atoms[0].molecule_id, None);
        assert_eq!(data.atoms[0].position, [1.25, 2.5, 3.75]);
        let bounds = data.bounds.unwrap();
        assert_eq!([bounds.xy, bounds.xz, bounds.yz], [1.0, -0.5, 0.25]);
        assert!((data.dimensions().unwrap()[5] - 78.6900675).abs() < 1.0e-6);

        let with_image = data.to_string().unwrap();
        let reparsed = LammpsData::from_str_with_atom_style(&with_image, "id type x y z").unwrap();
        assert_eq!(reparsed.atoms[0].image, data.atoms[0].image);
    }

    #[test]
    fn round_trip_preserves_data_and_reader_api() {
        let data = LammpsData::read(Cursor::new(MINI.as_bytes())).unwrap();
        let output = data.to_string().unwrap();
        let reparsed = LammpsData::from_str(&output).unwrap();
        assert_eq!(reparsed.atoms, data.atoms);
        assert_eq!(reparsed.bonds, data.bonds);
        assert_eq!(reparsed.velocities, data.velocities);
        assert_eq!(reparsed.masses, data.masses);
    }

    #[test]
    fn unknown_atoms_comment_uses_field_count_inference() {
        let input = MINI.replace("Atoms # full", "Atoms # written by a simulator");
        let data = LammpsData::from_str(&input).unwrap();
        assert_eq!(data.atom_style, "full");
        assert_eq!(data.atoms[0].molecule_id, Some(2));
        assert_eq!(data.atoms[0].charge, Some(0.2));
    }

    #[test]
    fn constructs_universe_with_metadata_velocities_box_and_bonds() {
        let data = LammpsData::from_str(MINI).unwrap();
        let universe = crate::core::Universe::from_lammps_data(data).unwrap();

        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.topology.atoms[0].name, "TYPE1");
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("1"));
        assert_eq!(universe.topology.atoms[0].mass, 12.011);
        assert_eq!(universe.topology.atoms[0].resid, 2);
        assert_eq!(universe.topology.atoms[0].charge, 0.2);
        assert_eq!(
            universe.trajectory.frames[0].velocities.as_ref().unwrap()[0],
            [-0.1, -0.2, -0.3]
        );
        assert_eq!(
            universe.trajectory.frames[0].dimensions,
            Some([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        );
        assert_eq!(universe.topology.bonds.len(), 1);
        assert_eq!(
            (
                universe.topology.bonds[0].atom1,
                universe.topology.bonds[0].atom2
            ),
            (1, 0)
        );
        assert_eq!(universe.topology.bonds[0].order, Some(1));
    }

    #[test]
    fn malformed_counts_and_references_are_rejected() {
        let bad_count = MINI.replace("2 atoms", "3 atoms");
        assert!(matches!(
            LammpsData::from_str(&bad_count),
            Err(LammpsError::Parse { .. })
        ));
        let bad_bond = MINI.replace("1 1 10 3", "1 1 10 99");
        assert!(matches!(
            LammpsData::from_str(&bad_bond),
            Err(LammpsError::Parse { .. })
        ));
        let missing_atoms = "title\n\n1 atoms\n";
        assert!(LammpsData::from_str(missing_atoms).is_err());

        let bad_bounds = MINI.replace("0.0 10.0 xlo xhi", "bad 10.0 xlo xhi");
        assert!(matches!(
            LammpsData::from_str(&bad_bounds),
            Err(LammpsError::Parse { .. })
        ));
    }

    #[test]
    fn writer_rejects_nonfinite_data() {
        let mut data = LammpsData::from_str(MINI).unwrap();
        data.atoms[0].position[0] = f64::NAN;
        assert!(matches!(
            data.to_string(),
            Err(LammpsError::InvalidStructure(_))
        ));
    }

    const DUMP: &str = concat!(
        "ITEM: TIMESTEP\n",
        "10\n",
        "ITEM: NUMBER OF ATOMS\n",
        "2\n",
        "ITEM: BOX BOUNDS pp pp pp\n",
        "1 11\n",
        "-2 8\n",
        "3 13\n",
        "ITEM: ATOMS id mol type q x y z vx vy vz fx fy fz foo\n",
        "2 7 2 -0.2 4 5 6 0.1 0.2 0.3 1 2 3 8\n",
        "1 7 1 0.2 2 3 4 -0.1 -0.2 -0.3 -1 -2 -3 9\n",
        "ITEM: TIMESTEP\n",
        "20\n",
        "ITEM: NUMBER OF ATOMS\n",
        "2\n",
        "ITEM: BOX BOUNDS pp pp pp\n",
        "1 11\n",
        "-2 8\n",
        "3 13\n",
        "ITEM: ATOMS id mol type q x y z vx vy vz fx fy fz foo\n",
        "1 7 1 0.2 3 4 5 -0.4 -0.5 -0.6 -4 -5 -6 10\n",
        "2 7 2 -0.2 5 6 7 0.4 0.5 0.6 4 5 6 11\n",
    );

    #[test]
    fn parses_dump_frames_and_sorts_atoms() {
        let dump = LammpsDumpFile::from_str(DUMP).unwrap();
        assert_eq!(dump.n_atoms(), 2);
        assert_eq!(dump.n_frames(), 2);
        assert_eq!(dump.atom_ids, vec![1, 2]);
        assert_eq!(dump.molecule_ids, Some(vec![7, 7]));
        assert_eq!(dump.atom_types, Some(vec!["1".into(), "2".into()]));
        assert_eq!(dump.charges, Some(vec![0.2, -0.2]));
        assert_eq!(
            dump.frames[0].positions,
            vec![[1.0, 5.0, 1.0], [3.0, 7.0, 3.0]]
        );
        assert_eq!(
            dump.frames[0].velocities.as_ref().unwrap()[0],
            [-0.1, -0.2, -0.3]
        );
        assert_eq!(dump.frames[0].forces.as_ref().unwrap()[1], [1.0, 2.0, 3.0]);
        assert_eq!(dump.frames[0].additional_columns["foo"], vec![9.0, 8.0]);
        assert_eq!(dump.frames[0].step, 10);
        assert_eq!(dump.frames[1].step, 20);
        assert_eq!(
            dump.frames[0].dimensions,
            Some([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        );
        assert_eq!(dump.frames[0].time, 10.0);
        assert_eq!(dump.frames[1].time, 20.0);
    }

    #[test]
    fn parses_optional_units_and_elapsed_time_headers() {
        let input = format!(
            "ITEM: UNITS\nmetal\nITEM: TIME\n1.25\n{}",
            DUMP.replace("ITEM: TIMESTEP\n20", "ITEM: TIMESTEP\n20\nITEM: TIME\n3.75")
        );
        let dump = LammpsDumpFile::from_str(&input).unwrap();
        assert_eq!(dump.frames[0].time, 1.25);
        assert_eq!(dump.frames[1].time, 3.75);
    }

    #[test]
    fn reads_gzip_dump_files() {
        let path = std::env::temp_dir().join(format!(
            "mdanalysis-rs-lammps-{}-{}.dump.gz",
            std::process::id(),
            "gzip"
        ));
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(DUMP.as_bytes()).unwrap();
        std::fs::write(&path, encoder.finish().unwrap()).unwrap();
        let dump = LammpsDumpFile::read_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(dump.n_frames(), 2);
        assert_eq!(dump.atom_ids, vec![1, 2]);
    }

    #[test]
    fn parses_scaled_and_triclinic_dump_coordinates() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n1\n",
            "ITEM: BOX BOUNDS xy xz yz pp pp pp\n",
            "-1 5 1\n-2 6 -0.5\n3 9 0.25\n",
            "ITEM: ATOMS id type xs ys zs\n1 1 0.5 0.5 0.5\n",
        );
        let dump = LammpsDumpFile::from_str_with_options(
            input,
            LammpsDumpOptions {
                coordinate_convention: LammpsCoordinateConvention::Scaled,
                ..LammpsDumpOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            dump.coordinate_convention,
            LammpsCoordinateConvention::Scaled
        );
        let bounds = dump.frames[0].bounds;
        assert_eq!(
            [bounds.xlo, bounds.xhi, bounds.ylo, bounds.yhi],
            [-0.5, 4.0, -2.0, 5.75]
        );
        let position = dump.frames[0].positions[0];
        assert!(
            position
                .iter()
                .zip([3.0, 6.0, 0.0])
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-12)
        );
        assert!(
            dump.frames[0].positions[0]
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn unwraps_images_with_triclinic_lattice_vectors() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n1\n",
            "ITEM: BOX BOUNDS xy xz yz pp pp pp\n",
            "-1 5 1\n-2 6 -0.5\n3 9 0.25\n",
            "ITEM: ATOMS id x y z ix iy iz\n",
            "1 0.5 1 4 1 -1 2\n",
        );
        let dump = LammpsDumpFile::from_str_with_options(
            input,
            LammpsDumpOptions {
                unwrap_images: true,
                ..LammpsDumpOptions::default()
            },
        )
        .unwrap();
        assert_eq!(dump.frames[0].positions[0], [3.5, -4.25, 13.0]);
    }

    #[test]
    fn does_not_double_unwrap_unwrapped_coordinates() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n1\n",
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n",
            "ITEM: ATOMS id xu yu zu ix iy iz\n",
            "1 12 3 4 1 0 0\n",
        );
        let dump = LammpsDumpFile::from_str_with_options(
            input,
            LammpsDumpOptions {
                coordinate_convention: LammpsCoordinateConvention::Unwrapped,
                unwrap_images: true,
            },
        )
        .unwrap();
        assert_eq!(dump.frames[0].positions[0], [12.0, 3.0, 4.0]);
    }

    #[test]
    fn unwraps_images_and_reads_universe_metadata() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n2\n",
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n",
            "ITEM: ATOMS id mol type mass element q x y z ix iy iz\n",
            "2 4 2 1.008 H -0.1 2 3 4 0 0 1\n",
            "1 4 1 12.011 C 0.1 1 2 3 1 -1 0\n",
        );
        let dump = LammpsDumpFile::from_str_with_options(
            input,
            LammpsDumpOptions {
                unwrap_images: true,
                ..LammpsDumpOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            dump.frames[0].positions,
            vec![[11.0, -8.0, 3.0], [2.0, 3.0, 14.0]]
        );
        let universe = crate::core::Universe::from_lammps_dump_file(dump).unwrap();
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("C"));
        assert_eq!(universe.topology.atoms[0].mass, 12.011);
        assert_eq!(universe.topology.atoms[0].charge, 0.1);
        assert_eq!(universe.topology.atoms[0].resid, 4);
        assert_eq!(universe.trajectory.frames[0].positions[1], [2.0, 3.0, 14.0]);
    }

    #[test]
    fn absent_optional_topology_columns_remain_absent() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n1\n",
            "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n",
            "ITEM: ATOMS id x y z\n1 0.1 0.2 0.3\n",
        );
        let dump = LammpsDumpFile::from_str(input).unwrap();
        assert_eq!(dump.molecule_ids, None);
        assert_eq!(dump.atom_types, None);
        assert_eq!(dump.masses, None);
        assert_eq!(dump.elements, None);
        assert_eq!(dump.charges, None);
        let universe = crate::core::Universe::from_lammps_dump_file(dump).unwrap();
        assert_eq!(universe.topology.atoms[0].atom_type.as_deref(), Some("1"));
        assert_eq!(universe.topology.atoms[0].mass, 1.0);
    }

    #[test]
    fn rejects_invalid_dump_structure_and_convention() {
        let no_coords = DUMP.replace("q x y z vx", "q badx bady badz vx");
        assert!(matches!(
            LammpsDumpFile::from_str(&no_coords),
            Err(LammpsError::InvalidStructure(_))
        ));
        assert!(matches!(
            LammpsDumpFile::from_str_with_options(
                DUMP,
                LammpsDumpOptions {
                    coordinate_convention: LammpsCoordinateConvention::Scaled,
                    ..LammpsDumpOptions::default()
                }
            ),
            Err(LammpsError::InvalidStructure(_))
        ));
        let changed_count = DUMP.replace("ITEM: NUMBER OF ATOMS\n2", "ITEM: NUMBER OF ATOMS\n3");
        assert!(matches!(
            LammpsDumpFile::from_str(&changed_count),
            Err(LammpsError::Parse { .. })
        ));
        let partial_tilt = DUMP.replace(
            "ITEM: BOX BOUNDS pp pp pp",
            "ITEM: BOX BOUNDS xy xz pp pp pp",
        );
        assert!(matches!(
            LammpsDumpFile::from_str(&partial_tilt),
            Err(LammpsError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_final_positions() {
        let input = concat!(
            "ITEM: TIMESTEP\n0\n",
            "ITEM: NUMBER OF ATOMS\n1\n",
            "ITEM: BOX BOUNDS pp pp pp\n0 1e308\n0 1e308\n0 1e308\n",
            "ITEM: ATOMS id x y z ix iy iz\n",
            "1 1e308 0 0 1 0 0\n",
        );
        assert!(matches!(
            LammpsDumpFile::from_str_with_options(
                input,
                LammpsDumpOptions {
                    unwrap_images: true,
                    ..LammpsDumpOptions::default()
                }
            ),
            Err(LammpsError::Parse { .. })
        ));
    }

    #[test]
    fn validates_public_dump_frames_when_attaching_to_data() {
        let mut data = LammpsData::from_str(MINI).unwrap();
        data.atoms[0].id = 1;
        data.atoms[1].id = 2;
        data.bonds.clear();
        let mut dump = LammpsDumpFile::from_str(DUMP).unwrap();
        dump.frames[1].velocities = Some(Vec::new());
        assert!(matches!(
            crate::core::Universe::from_lammps_data_and_dump(data, dump),
            Err(crate::Error::InvalidInput(_))
        ));
    }

    #[test]
    fn combines_data_and_dump_by_atom_id_order() {
        let mut data = LammpsData::from_str(MINI).unwrap();
        data.atoms[0].id = 2;
        data.atoms[1].id = 1;
        data.bonds.clear();
        let dump = LammpsDumpFile::from_str(DUMP).unwrap();
        let universe = crate::core::Universe::from_lammps_data_and_dump(data, dump).unwrap();
        assert_eq!(
            universe.trajectory.frames[0].positions,
            vec![[3.0, 7.0, 3.0], [1.0, 5.0, 1.0]]
        );
        assert_eq!(universe.topology.atoms[0].position, [3.0, 7.0, 3.0]);
    }
}
