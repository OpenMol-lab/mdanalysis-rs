//! HOOMD XML topology and coordinate support.
//!
//! HOOMD's XML format stores a complete configuration in one XML document.
//! Atom data are unitless in the format; this module therefore preserves the
//! supplied masses, charges, and diameters without applying a unit
//! conversion.  Coordinates are exposed through [`HoomdXmlFile::coordinates`]
//! and can be attached to a [`crate::core::Universe`] with
//! [`crate::core::Universe::from_hoomdxml`].

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use bzip2::read::BzDecoder;
use quick_xml::escape::unescape;
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// One atom record from a HOOMD XML configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct HoomdXmlAtom {
    pub index: usize,
    pub name: String,
    pub atom_type: String,
    pub position: [f64; 3],
    pub velocity: Option<[f64; 3]>,
    pub mass: f64,
    pub charge: f64,
    /// Particle diameter.  HOOMD stores diameter, while most analysis APIs
    /// expose radius; [`Self::radius`] performs the conversion.
    pub diameter: Option<f64>,
    pub body: Option<i64>,
}

impl HoomdXmlAtom {
    #[must_use]
    pub fn radius(&self) -> Option<f64> {
        self.diameter.map(|diameter| diameter / 2.0)
    }
}

/// A HOOMD bond expressed in zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HoomdXmlBond {
    pub atom1: usize,
    pub atom2: usize,
}

impl HoomdXmlBond {
    #[must_use]
    pub const fn new(atom1: usize, atom2: usize) -> Self {
        Self { atom1, atom2 }
    }
}

/// A HOOMD angle expressed in zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HoomdXmlAngle {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
}

impl HoomdXmlAngle {
    #[must_use]
    pub const fn new(atom1: usize, atom2: usize, atom3: usize) -> Self {
        Self {
            atom1,
            atom2,
            atom3,
        }
    }
}

/// A HOOMD dihedral expressed in zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HoomdXmlDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

impl HoomdXmlDihedral {
    #[must_use]
    pub const fn new(atom1: usize, atom2: usize, atom3: usize, atom4: usize) -> Self {
        Self {
            atom1,
            atom2,
            atom3,
            atom4,
        }
    }
}

/// A HOOMD improper expressed in zero-based atom indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HoomdXmlImproper {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
}

impl HoomdXmlImproper {
    #[must_use]
    pub const fn new(atom1: usize, atom2: usize, atom3: usize, atom4: usize) -> Self {
        Self {
            atom1,
            atom2,
            atom3,
            atom4,
        }
    }
}

/// HOOMD's triclinic box parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoomdXmlBox {
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
}

impl HoomdXmlBox {
    #[must_use]
    pub const fn new(lx: f64, ly: f64, lz: f64, xy: f64, xz: f64, yz: f64) -> Self {
        Self {
            lx,
            ly,
            lz,
            xy,
            xz,
            yz,
        }
    }

    /// Convert HOOMD's tilt-factor representation to MDAnalysis dimensions.
    #[must_use]
    pub fn dimensions(self) -> [f64; 6] {
        triclinic_box([
            [self.lx, 0.0, 0.0],
            [self.xy, self.ly, 0.0],
            [self.xz, self.yz, self.lz],
        ])
    }
}

/// Parsed HOOMD XML topology and its single coordinate frame.
#[derive(Clone, Debug, PartialEq)]
pub struct HoomdXmlFile {
    pub atoms: Vec<HoomdXmlAtom>,
    pub bonds: Vec<HoomdXmlBond>,
    pub angles: Vec<HoomdXmlAngle>,
    pub dihedrals: Vec<HoomdXmlDihedral>,
    pub impropers: Vec<HoomdXmlImproper>,
    pub cell: Option<HoomdXmlBox>,
    pub dimensions: Option<[f64; 6]>,
    pub time_step: usize,
    pub coordinates: CoordinateFile,
}

/// Naming aliases matching the terminology used by other format modules.
pub type HoomdXmlData = HoomdXmlFile;
pub type HoomdXmlStructure = HoomdXmlFile;

impl HoomdXmlFile {
    /// Parse an uncompressed HOOMD XML document from UTF-8 text.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, HoomdXmlError> {
        parse_xml(input.as_bytes())
    }

    /// Parse an uncompressed HOOMD XML document from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HoomdXmlError> {
        Self::read(Cursor::new(bytes))
    }

    /// Read an uncompressed HOOMD XML document from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, HoomdXmlError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        if bytes.starts_with(b"BZh") {
            let mut decoder = BzDecoder::new(Cursor::new(bytes));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            parse_xml(&decompressed)
        } else {
            parse_xml(&bytes)
        }
    }

    /// Read a HOOMD XML document, transparently decompressing `.bz2` files.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, HoomdXmlError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        Self::read(file)
    }

    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    #[must_use]
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|atom| atom.position).collect()
    }

    #[must_use]
    pub fn atom_types(&self) -> Vec<String> {
        self.atoms
            .iter()
            .map(|atom| atom.atom_type.clone())
            .collect()
    }

    #[must_use]
    pub fn masses(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.mass).collect()
    }

    #[must_use]
    pub fn charges(&self) -> Vec<f64> {
        self.atoms.iter().map(|atom| atom.charge).collect()
    }

    #[must_use]
    pub fn radii(&self) -> Vec<f64> {
        self.atoms
            .iter()
            .map(|atom| atom.radius().unwrap_or(0.0))
            .collect()
    }

    #[must_use]
    pub fn optional_radii(&self) -> Vec<Option<f64>> {
        self.atoms.iter().map(HoomdXmlAtom::radius).collect()
    }
}

/// Read a HOOMD XML document from a filesystem path.
pub fn read_hoomdxml(path: impl AsRef<Path>) -> Result<HoomdXmlFile, HoomdXmlError> {
    HoomdXmlFile::read_file(path)
}

/// Errors produced by the HOOMD XML parser.
#[derive(Debug)]
pub enum HoomdXmlError {
    Io(io::Error),
    Xml(String),
    Utf8(std::str::Utf8Error),
    Parse { section: String, message: String },
    InvalidStructure(String),
}

impl fmt::Display for HoomdXmlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "HOOMD XML I/O error: {error}"),
            Self::Xml(error) => write!(formatter, "HOOMD XML syntax error: {error}"),
            Self::Utf8(error) => write!(formatter, "HOOMD XML is not UTF-8: {error}"),
            Self::Parse { section, message } => {
                write!(formatter, "HOOMD XML {section} parse error: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid HOOMD XML structure: {message}")
            }
        }
    }
}

impl std::error::Error for HoomdXmlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Utf8(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for HoomdXmlError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<std::str::Utf8Error> for HoomdXmlError {
    fn from(error: std::str::Utf8Error) -> Self {
        Self::Utf8(error)
    }
}

#[derive(Clone, Debug)]
struct Section {
    num: Option<usize>,
    text: String,
}

fn parse_xml(bytes: &[u8]) -> Result<HoomdXmlFile, HoomdXmlError> {
    let mut reader = Reader::from_reader(bytes);
    let mut buffer = Vec::new();
    let mut root_seen = false;
    let mut configuration_seen = false;
    let mut natoms = None;
    let mut time_step = 0usize;
    let mut sections: HashMap<String, Section> = HashMap::new();
    let mut cell = None;
    let mut active: Option<(String, Section)> = None;

    loop {
        let event = reader
            .read_event_into(&mut buffer)
            .map_err(|error| HoomdXmlError::Xml(error.to_string()))?;
        match event {
            Event::Eof => break,
            Event::Decl(_) | Event::Comment(_) | Event::PI(_) | Event::DocType(_) => {}
            Event::GeneralRef(_) => {}
            Event::Start(start) => {
                let name = local_name(start.name().as_ref());
                match name.as_str() {
                    "hoomd_xml" => {
                        if root_seen {
                            return Err(invalid("multiple hoomd_xml roots"));
                        }
                        root_seen = true;
                    }
                    "configuration" => {
                        if configuration_seen {
                            return Err(invalid("multiple configuration elements"));
                        }
                        configuration_seen = true;
                        let attributes = attributes(&start)?;
                        natoms = Some(required_usize(&attributes, "natoms", "configuration")?);
                        if let Some(value) = attributes.get("time_step") {
                            time_step = parse_usize(value, "configuration", "time_step")?;
                        }
                    }
                    "box" if configuration_seen => {
                        if cell.is_some() {
                            return Err(invalid("multiple box elements"));
                        }
                        cell = Some(parse_box(&attributes(&start)?)?);
                    }
                    _ if configuration_seen && active.is_none() => {
                        let attributes = attributes(&start)?;
                        active = Some((
                            name,
                            Section {
                                num: attributes
                                    .get("num")
                                    .map(|value| parse_usize(value, "section", "num"))
                                    .transpose()?,
                                text: String::new(),
                            },
                        ));
                    }
                    _ => {}
                }
            }
            Event::Empty(start) => {
                let name = local_name(start.name().as_ref());
                if name == "box" && configuration_seen {
                    if cell.is_some() {
                        return Err(invalid("multiple box elements"));
                    }
                    cell = Some(parse_box(&attributes(&start)?)?);
                }
            }
            Event::Text(text) => {
                if let Some((_, section)) = active.as_mut() {
                    let value = unescape(text.as_ref())
                        .map_err(|error| HoomdXmlError::Xml(error.to_string()))?;
                    section.text.push_str(&value);
                }
            }
            Event::CData(text) => {
                if let Some((_, section)) = active.as_mut() {
                    section.text.push_str(text.as_ref());
                }
            }
            Event::End(end) => {
                let name = local_name(end.name().as_ref());
                if let Some((active_name, section)) = active.take() {
                    if active_name == name {
                        if sections.insert(active_name.clone(), section).is_some() {
                            return Err(invalid(format!("duplicate section {active_name}")));
                        }
                    } else {
                        active = Some((active_name, section));
                    }
                }
            }
        }
        buffer.clear();
    }

    if !root_seen {
        return Err(invalid("missing hoomd_xml root"));
    }
    let natoms = natoms.ok_or_else(|| invalid("missing configuration element"))?;
    if natoms == 0 {
        return Err(invalid("natoms must be greater than zero"));
    }
    let position_section = sections
        .get("position")
        .ok_or_else(|| invalid("missing position section"))?;
    let positions = parse_vector_section(position_section, natoms, "position")?;
    let types = parse_labels(sections.get("type"), natoms, "type", "none")?;
    let names = parse_labels(sections.get("name"), natoms, "name", "")?;
    let masses = parse_optional_scalars(sections.get("mass"), natoms, "mass")?
        .unwrap_or_else(|| vec![0.0; natoms]);
    let charges = parse_optional_scalars(sections.get("charge"), natoms, "charge")?
        .unwrap_or_else(|| vec![0.0; natoms]);
    let diameters = parse_optional_scalars(sections.get("diameter"), natoms, "diameter")?;
    let bodies = parse_optional_integers(sections.get("body"), natoms, "body")?;
    let velocities = sections
        .get("velocity")
        .map(|section| parse_vector_section(section, natoms, "velocity"))
        .transpose()?;

    let atoms = (0..natoms)
        .map(|index| HoomdXmlAtom {
            index,
            name: if names[index].is_empty() {
                types[index].clone()
            } else {
                names[index].clone()
            },
            atom_type: types[index].clone(),
            position: positions[index],
            velocity: velocities.as_ref().map(|values| values[index]),
            mass: masses[index],
            charge: charges[index],
            diameter: diameters.as_ref().map(|values| values[index]),
            body: bodies.as_ref().map(|values| values[index]),
        })
        .collect::<Vec<_>>();

    let bonds = parse_interactions::<2>(sections.get("bond"), natoms, "bond")?
        .into_iter()
        .map(|value| HoomdXmlBond::new(value[0], value[1]))
        .collect::<Vec<_>>();
    let angles = parse_interactions::<3>(sections.get("angle"), natoms, "angle")?
        .into_iter()
        .map(|value| HoomdXmlAngle::new(value[0], value[1], value[2]))
        .collect::<Vec<_>>();
    let dihedrals = parse_interactions::<4>(sections.get("dihedral"), natoms, "dihedral")?
        .into_iter()
        .map(|value| HoomdXmlDihedral::new(value[0], value[1], value[2], value[3]))
        .collect::<Vec<_>>();
    let impropers = parse_interactions::<4>(sections.get("improper"), natoms, "improper")?
        .into_iter()
        .map(|value| HoomdXmlImproper::new(value[0], value[1], value[2], value[3]))
        .collect::<Vec<_>>();

    let dimensions = cell.map(HoomdXmlBox::dimensions);
    if dimensions.is_some_and(|values| values[..3].iter().any(|value| *value <= 0.0)) {
        return Err(invalid("box lengths must be positive"));
    }
    let mut coordinate = CoordinateFrame::new(positions.clone());
    coordinate.names = atoms.iter().map(|atom| atom.name.clone()).collect();
    coordinate.residue_names = vec!["SYSTEM".to_owned(); natoms];
    coordinate.residue_ids = vec![1; natoms];
    coordinate.atom_ids = (1..=natoms).collect();
    coordinate.velocities = velocities;
    coordinate.dimensions = dimensions;
    coordinate.step = time_step;
    coordinate.time = time_step as f64;
    let coordinates = CoordinateFile::new(vec![coordinate]);
    let file = HoomdXmlFile {
        atoms,
        bonds,
        angles,
        dihedrals,
        impropers,
        cell,
        dimensions,
        time_step,
        coordinates,
    };
    validate_file(&file)?;
    Ok(file)
}

fn parse_box(attributes: &HashMap<String, String>) -> Result<HoomdXmlBox, HoomdXmlError> {
    let lx = required_float(attributes, "lx", "box")?;
    let ly = required_float(attributes, "ly", "box")?;
    let lz = required_float(attributes, "lz", "box")?;
    let xy = optional_float(attributes, "xy", "box")?.unwrap_or(0.0);
    let xz = optional_float(attributes, "xz", "box")?.unwrap_or(0.0);
    let yz = optional_float(attributes, "yz", "box")?.unwrap_or(0.0);
    let values = [lx, ly, lz, xy, xz, yz];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("box contains non-finite values"));
    }
    if [lx, ly, lz].iter().any(|value| *value <= 0.0) {
        return Err(invalid("box lengths must be positive"));
    }
    Ok(HoomdXmlBox::new(lx, ly, lz, xy, xz, yz))
}

fn parse_labels(
    section: Option<&Section>,
    expected: usize,
    name: &str,
    default: &str,
) -> Result<Vec<String>, HoomdXmlError> {
    let Some(section) = section else {
        return Ok(vec![default.to_owned(); expected]);
    };
    let values = section
        .text
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    check_count(section, values.len(), expected, name)?;
    Ok(values)
}

fn parse_optional_scalars(
    section: Option<&Section>,
    expected: usize,
    name: &str,
) -> Result<Option<Vec<f64>>, HoomdXmlError> {
    let Some(section) = section else {
        return Ok(None);
    };
    let values = section
        .text
        .split_whitespace()
        .map(|value| parse_float(value, name))
        .collect::<Result<Vec<_>, _>>()?;
    check_count(section, values.len(), expected, name)?;
    Ok(Some(values))
}

fn parse_optional_integers(
    section: Option<&Section>,
    expected: usize,
    name: &str,
) -> Result<Option<Vec<i64>>, HoomdXmlError> {
    let Some(section) = section else {
        return Ok(None);
    };
    let values = section
        .text
        .split_whitespace()
        .map(|value| {
            value
                .parse::<i64>()
                .map_err(|error| parse_error(name, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    check_count(section, values.len(), expected, name)?;
    Ok(Some(values))
}

fn parse_vectors(text: &str, expected: usize, name: &str) -> Result<Vec<[f64; 3]>, HoomdXmlError> {
    let values = text
        .split_whitespace()
        .map(|value| parse_float(value, name))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected.saturating_mul(3) {
        return Err(parse_error(
            name,
            format!("expected {} values, found {}", expected * 3, values.len()),
        ));
    }
    Ok(values
        .chunks(3)
        .map(|value| [value[0], value[1], value[2]])
        .collect())
}

fn parse_vector_section(
    section: &Section,
    expected: usize,
    name: &str,
) -> Result<Vec<[f64; 3]>, HoomdXmlError> {
    let values = parse_vectors(&section.text, expected, name)?;
    check_count(section, values.len(), expected, name)?;
    Ok(values)
}

fn parse_interactions<const N: usize>(
    section: Option<&Section>,
    natoms: usize,
    name: &str,
) -> Result<Vec<[usize; N]>, HoomdXmlError> {
    let Some(section) = section else {
        return Ok(Vec::new());
    };
    let lines = section
        .text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let records = if lines.iter().all(|tokens| tokens.len() == N + 1) {
        lines
    } else {
        let tokens = section.text.split_whitespace().collect::<Vec<_>>();
        if tokens.len() % (N + 1) != 0 {
            return Err(parse_error(
                name,
                format!("expected a label and {N} atom indices per record"),
            ));
        }
        tokens.chunks(N + 1).map(|chunk| chunk.to_vec()).collect()
    };
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for tokens in records {
        if tokens.len() != N + 1 {
            return Err(parse_error(
                name,
                format!("expected a label and {N} atom indices per line"),
            ));
        }
        let mut record = [0usize; N];
        for (index, token) in tokens[1..].iter().enumerate() {
            record[index] = token
                .parse::<usize>()
                .map_err(|error| parse_error(name, error))?;
            if record[index] >= natoms {
                return Err(parse_error(
                    name,
                    format!("atom index {} is outside 0..{}", record[index], natoms),
                ));
            }
        }
        if (N == 2 && record[0] == record[1]) || !seen.insert(record) {
            return Err(invalid(format!("duplicate or self-referential {name}")));
        }
        values.push(record);
    }
    if section.num.is_some_and(|declared| declared != values.len()) {
        return Err(parse_error(
            name,
            format!(
                "declares {} records, found {}",
                section.num.unwrap_or_default(),
                values.len()
            ),
        ));
    }
    Ok(values)
}

fn validate_file(file: &HoomdXmlFile) -> Result<(), HoomdXmlError> {
    if file.atoms.is_empty() || file.coordinates.n_frames() != 1 {
        return Err(invalid("configuration must contain one non-empty frame"));
    }
    for atom in &file.atoms {
        if !atom.position.iter().all(|value| value.is_finite())
            || !atom.mass.is_finite()
            || !atom.charge.is_finite()
            || atom.mass < 0.0
            || atom
                .diameter
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(invalid(format!(
                "atom {} has non-finite or negative values",
                atom.index
            )));
        }
    }
    Ok(())
}

fn check_count(
    section: &Section,
    actual: usize,
    expected: usize,
    name: &str,
) -> Result<(), HoomdXmlError> {
    if let Some(declared) = section.num
        && declared != actual
    {
        return Err(parse_error(
            name,
            format!("declares {declared} records, found {actual}"),
        ));
    }
    if actual != expected {
        return Err(parse_error(
            name,
            format!("expected {expected}, found {actual}"),
        ));
    }
    Ok(())
}

fn attributes(start: &BytesStart<'_>) -> Result<HashMap<String, String>, HoomdXmlError> {
    let mut result = HashMap::new();
    for attribute in start.attributes() {
        let attribute = attribute.map_err(|error| HoomdXmlError::Xml(error.to_string()))?;
        let key = attribute.key.as_ref().to_owned();
        let value = attribute
            .normalized_value(XmlVersion::Implicit1_0)
            .map_err(|error| HoomdXmlError::Xml(error.to_string()))?
            .into_owned();
        result.insert(key, value);
    }
    Ok(result)
}

fn local_name(name: &str) -> String {
    name.rsplit(':').next().unwrap_or(name).to_owned()
}

fn required_usize(
    values: &HashMap<String, String>,
    key: &str,
    section: &str,
) -> Result<usize, HoomdXmlError> {
    let value = values
        .get(key)
        .ok_or_else(|| parse_error(section, format!("missing {key} attribute")))?;
    parse_usize(value, section, key)
}

fn parse_usize(value: &str, section: &str, key: &str) -> Result<usize, HoomdXmlError> {
    value
        .parse::<usize>()
        .map_err(|error| parse_error(section, format!("invalid {key} {value:?}: {error}")))
}

fn required_float(
    values: &HashMap<String, String>,
    key: &str,
    section: &str,
) -> Result<f64, HoomdXmlError> {
    values
        .get(key)
        .ok_or_else(|| parse_error(section, format!("missing {key} attribute")))
        .and_then(|value| parse_float(value, section))
}

fn optional_float(
    values: &HashMap<String, String>,
    key: &str,
    section: &str,
) -> Result<Option<f64>, HoomdXmlError> {
    values
        .get(key)
        .map(|value| parse_float(value, section))
        .transpose()
}

fn parse_float(value: &str, section: &str) -> Result<f64, HoomdXmlError> {
    value
        .parse::<f64>()
        .map_err(|error| parse_error(section, format!("invalid number {value:?}: {error}")))
}

fn parse_error(section: &str, error: impl fmt::Display) -> HoomdXmlError {
    HoomdXmlError::Parse {
        section: section.to_owned(),
        message: error.to_string(),
    }
}

fn invalid(error: impl Into<String>) -> HoomdXmlError {
    HoomdXmlError::InvalidStructure(error.into())
}

impl Universe {
    /// Construct a universe from a HOOMD XML file, including its coordinates.
    pub fn from_hoomdxml(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_hoomdxml_file(read_hoomdxml(path)?)
    }

    /// Construct a universe from HOOMD XML text.
    pub fn from_hoomdxml_str(input: &str) -> crate::Result<Self> {
        Self::from_hoomdxml_file(HoomdXmlFile::from_str(input)?)
    }

    /// Construct a universe from HOOMD XML bytes.  Bzip2-compressed bytes are
    /// accepted as well as plain UTF-8 XML.
    pub fn from_hoomdxml_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_hoomdxml_file(HoomdXmlFile::read(Cursor::new(bytes))?)
    }

    /// Construct a universe from parsed HOOMD XML data.
    pub fn from_hoomdxml_file(file: HoomdXmlFile) -> crate::Result<Self> {
        let atoms = file
            .atoms
            .iter()
            .map(|source| {
                let element = crate::guesser::guess_element(
                    &source.name,
                    Some(&source.atom_type),
                    Some("SYSTEM"),
                )
                .ok();
                let mut atom = Atom::new(source.index, source.name.clone(), source.position);
                atom.atom_type = Some(source.atom_type.clone());
                atom.element = element;
                atom.mass = source.mass;
                atom.charge = source.charge;
                atom.resid = 1;
                atom.resname = "SYSTEM".to_owned();
                atom.segid = "SYSTEM".to_owned();
                atom.velocity = source.velocity;
                atom
            })
            .collect::<Vec<_>>();
        let mut topology = Topology::new(atoms);
        for source in &file.bonds {
            topology.add_bond(Bond::new(source.atom1, source.atom2));
        }
        let source_frame = file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("HOOMD XML has no coordinate frame".to_owned())
        })?;
        let mut frame = Frame::new(source_frame.positions.clone());
        frame.velocities = source_frame.velocities.clone();
        frame.dimensions = source_frame.dimensions;
        frame.time = source_frame.time;
        frame.step = source_frame.step;
        Ok(Self {
            topology,
            trajectory: Trajectory::new(vec![frame]),
        })
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
    fn reads_fixture_topology_and_coordinates() {
        let file = read_hoomdxml(fixture("C12x64.xml.bz2")).expect("HOOMD fixture should parse");
        assert_eq!(file.n_atoms(), 769);
        assert_eq!(file.bonds.len(), 704);
        assert_eq!(file.angles.len(), 640);
        assert_eq!(file.dihedrals.len(), 576);
        assert!(file.impropers.is_empty());
        assert_eq!(file.atoms[0].atom_type, "CH3");
        assert_eq!(file.atoms[0].mass, 1.0);
        assert_eq!(file.atoms[0].charge, 0.0);
        assert_eq!(file.atoms[0].radius(), Some(0.5));
        assert_eq!(file.positions()[0], [-100.0, -100.0, -100.0]);
        assert_eq!(
            file.dimensions,
            Some([300.0, 300.0, 300.0, 90.0, 90.0, 90.0])
        );
        assert_eq!(file.coordinates.frames[0].time, 0.0);
    }

    #[test]
    fn universe_constructor_preserves_bond_graph() {
        let universe = Universe::from_hoomdxml(fixture("C12x64.xml.bz2")).unwrap();
        assert_eq!(universe.n_atoms(), 769);
        assert_eq!(universe.n_residues(), 1);
        assert_eq!(universe.n_segments(), 1);
        assert_eq!(universe.topology.bonds.len(), 704);
        assert_eq!(universe.current_frame().unwrap().step, 0);
        assert_eq!(
            universe.current_frame().unwrap().dimensions,
            Some([300.0, 300.0, 300.0, 90.0, 90.0, 90.0])
        );
    }

    #[test]
    fn rejects_wrong_vector_count() {
        let input = r#"<hoomd_xml><configuration natoms="1"><position num="1">0 0</position></configuration></hoomd_xml>"#;
        assert!(matches!(
            HoomdXmlFile::from_str(input),
            Err(HoomdXmlError::Parse { .. })
        ));
    }
}
