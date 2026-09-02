//! LAMMPS DATA (text) topology and coordinate support.
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
        Self::read(File::open(path)?)
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
}

/// Errors produced while reading or writing LAMMPS DATA files.
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
    use std::io::Cursor;

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
}
