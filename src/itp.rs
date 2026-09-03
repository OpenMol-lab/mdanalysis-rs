//! GROMACS ITP/TOP topology support.
//!
//! ITP files are text topology files made up of bracketed sections.  This
//! module parses the topology information that is useful to the core
//! [`crate::core::Universe`]: atoms, atom types, bonds, angles, dihedrals,
//! impropers, and molecule composition.  GROMACS preprocessor directives
//! (`#define`, `#if`, `#ifdef`, `#ifndef`, `#else`, `#endif`, and `#include`)
//! are handled while reading.
//!
//! Coordinates are not part of an ITP file.  [`Universe::from_itp`] therefore
//! creates a zero-coordinate frame; use [`Universe::from_itp_and_pdb`] or
//! [`Universe::from_itp_and_gro`] when coordinates are available separately.

use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::guesser::{guess_element, guess_mass};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// An atom from a GROMACS `[ atoms ]` section.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpAtom {
    /// One-based atom number within its molecule type (or the expanded
    /// system, for [`ItpData::atoms`]).
    pub id: usize,
    /// Force-field atom type.
    pub atom_type: String,
    /// Residue number within the molecule type.
    pub residue_id: i32,
    /// Residue name.
    pub residue_name: String,
    /// Atom name.
    pub name: String,
    /// Charge-group number, when supplied by the file.
    pub charge_group: Option<i64>,
    /// Partial charge.
    pub charge: Option<f64>,
    /// Mass, if present in the record or resolved from `[ atomtypes ]`.
    pub mass: Option<f64>,
    /// Molecule type name after system expansion.
    pub molecule_type: String,
    /// One-based molecule number after system expansion.
    pub molecule_number: usize,
}

impl ItpAtom {
    /// Return the conventional `resid` spelling.
    #[must_use]
    pub const fn resid(&self) -> i32 {
        self.residue_id
    }

    /// Return the conventional `resname` spelling.
    #[must_use]
    pub fn resname(&self) -> &str {
        &self.residue_name
    }
}

/// A GROMACS bond or bond-like constraint.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpBond {
    pub atom1: usize,
    pub atom2: usize,
    pub function: usize,
    pub parameters: Vec<String>,
}

/// A GROMACS angle.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpAngle {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub function: usize,
    pub parameters: Vec<String>,
}

/// A GROMACS proper dihedral.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
    pub function: usize,
    pub parameters: Vec<String>,
}

/// A GROMACS improper dihedral.
pub type ItpImproper = ItpDihedral;

/// A `[ settles ]` water constraint.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpSettle {
    pub atom: usize,
    pub function: usize,
    pub oxygen_hydrogen: String,
    pub hydrogen_hydrogen: String,
}

/// A force-field atom type from `[ atomtypes ]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ItpAtomType {
    pub name: String,
    pub bonded_type: Option<String>,
    pub atomic_number: Option<u16>,
    pub mass: Option<f64>,
    pub charge: Option<f64>,
    pub particle_type: Option<String>,
    pub parameters: Vec<String>,
}

/// A molecule type declared by `[ moleculetype ]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ItpMoleculeType {
    pub name: String,
    pub exclusions: Option<usize>,
    pub atoms: Vec<ItpAtom>,
    pub bonds: Vec<ItpBond>,
    pub angles: Vec<ItpAngle>,
    pub dihedrals: Vec<ItpDihedral>,
    pub impropers: Vec<ItpImproper>,
    pub settles: Vec<ItpSettle>,
}

/// A row from the system-level `[ molecules ]` section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItpMoleculeCount {
    pub name: String,
    pub count: usize,
}

/// Parsed GROMACS ITP/TOP topology.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ItpData {
    /// Expanded atom records in system order.
    pub atoms: Vec<ItpAtom>,
    /// Explicit bonds and bonds generated from `[ constraints ]`/`[ settles ]`.
    pub bonds: Vec<ItpBond>,
    pub angles: Vec<ItpAngle>,
    pub dihedrals: Vec<ItpDihedral>,
    pub impropers: Vec<ItpImproper>,
    pub settles: Vec<ItpSettle>,
    /// Molecule types before system expansion.
    pub molecule_types: Vec<ItpMoleculeType>,
    pub molecules: Vec<ItpMoleculeCount>,
    pub atomtypes: HashMap<String, ItpAtomType>,
    pub system: Option<String>,
}

/// Alias matching the naming used by other topology readers in this crate.
pub type ItpStructure = ItpData;

/// Options controlling ITP preprocessing and system expansion.
#[derive(Clone, Debug, Default)]
pub struct ItpOptions {
    /// Include directories searched after the including file's directory.
    pub include_dirs: Vec<PathBuf>,
    /// Preprocessor definitions.  These override definitions in files.
    pub defines: HashMap<String, String>,
    /// If no `[ molecules ]` section exists, include one copy of each
    /// molecule type when true.  This is the MDAnalysis-compatible default.
    pub infer_system: bool,
}

impl ItpOptions {
    /// Add a preprocessor definition with value `1`.
    pub fn define(&mut self, name: impl Into<String>) {
        self.defines.insert(name.into(), "1".to_string());
    }

    /// Add a preprocessor definition with an explicit value.
    pub fn define_value(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), value.into());
    }

    /// Add a directory searched by `#include`.
    pub fn include_dir(&mut self, path: impl Into<PathBuf>) {
        self.include_dirs.push(path.into());
    }
}

/// Errors produced while parsing an ITP/TOP document.
#[derive(Debug)]
pub enum ItpError {
    Io(io::Error),
    Parse { line: usize, message: String },
    Include { path: PathBuf, message: String },
    InvalidStructure(String),
}

impl fmt::Display for ItpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "ITP parse error on line {line}: {message}")
            }
            Self::Include { path, message } => {
                write!(
                    formatter,
                    "ITP include error for {}: {message}",
                    path.display()
                )
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid ITP structure: {message}")
            }
        }
    }
}

impl std::error::Error for ItpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ItpError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl ItpData {
    /// Parse an ITP/TOP document held in memory.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, ItpError> {
        Self::from_str_with_options(input, ItpOptions::default())
    }

    /// Parse an in-memory document with preprocessor definitions.
    pub fn from_str_with_options(input: &str, options: ItpOptions) -> Result<Self, ItpError> {
        let mut state = Preprocessor::new(options);
        let lines = state.process_text(input, None)?;
        parse_lines(&lines)
    }

    /// Read and parse a filesystem ITP/TOP file.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, ItpError> {
        Self::read_file_with_options(path, ItpOptions::default())
    }

    /// Read and parse a filesystem file with include and define options.
    pub fn read_file_with_options(
        path: impl AsRef<Path>,
        options: ItpOptions,
    ) -> Result<Self, ItpError> {
        let path = path.as_ref();
        let mut state = Preprocessor::new(options);
        let lines = state.process_file(path)?;
        parse_lines(&lines)
    }

    /// Read and parse an arbitrary text reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, ItpError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Number of expanded atoms.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Look up an expanded atom by its one-based ID.
    #[must_use]
    pub fn atom(&self, id: usize) -> Option<&ItpAtom> {
        self.atoms.iter().find(|atom| atom.id == id)
    }
}

impl std::str::FromStr for ItpData {
    type Err = ItpError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Read an ITP/TOP topology from a path.
pub fn read_itp(path: impl AsRef<Path>) -> Result<ItpData, ItpError> {
    ItpData::read_file(path)
}

/// Read an ITP/TOP topology using explicit options.
pub fn read_itp_with_options(
    path: impl AsRef<Path>,
    options: ItpOptions,
) -> Result<ItpData, ItpError> {
    ItpData::read_file_with_options(path, options)
}

#[derive(Clone, Debug)]
struct SourceLine {
    line: usize,
    text: String,
}

#[derive(Clone, Debug)]
struct Conditional {
    parent_active: bool,
    condition_active: bool,
    seen_else: bool,
}

struct Preprocessor {
    options: ItpOptions,
    defines: HashMap<String, String>,
    protected: HashSet<String>,
    include_stack: Vec<PathBuf>,
}

impl Preprocessor {
    fn new(options: ItpOptions) -> Self {
        let protected = options.defines.keys().cloned().collect();
        let defines = options.defines.clone();
        Self {
            options,
            defines,
            protected,
            include_stack: Vec::new(),
        }
    }

    fn process_file(&mut self, path: &Path) -> Result<Vec<SourceLine>, ItpError> {
        let resolved = self.resolve_include(path, None)?;
        if self.include_stack.iter().any(|item| item == &resolved) {
            return Err(ItpError::Include {
                path: resolved,
                message: "recursive include detected".to_string(),
            });
        }
        let input = std::fs::read_to_string(&resolved).map_err(|error| ItpError::Include {
            path: resolved.clone(),
            message: error.to_string(),
        })?;
        self.include_stack.push(resolved);
        let origin = self.include_stack.last().cloned();
        let result = self.process_text(&input, origin.as_deref());
        self.include_stack.pop();
        result
    }

    fn process_text(
        &mut self,
        input: &str,
        origin: Option<&Path>,
    ) -> Result<Vec<SourceLine>, ItpError> {
        let mut output = Vec::new();
        let mut conditions: Vec<Conditional> = Vec::new();
        let mut logical_line = String::new();
        let mut logical_start = 0usize;

        for (line_index, raw) in input.lines().enumerate() {
            let line_number = line_index + 1;
            let mut line = raw.trim_end_matches(['\r', '\n']).to_string();
            if logical_line.is_empty() {
                logical_start = line_number;
            }
            if line.trim_end().ends_with('\\') {
                line = line
                    .trim_end()
                    .trim_end_matches('\\')
                    .trim_end()
                    .to_string();
                logical_line.push_str(&line);
                logical_line.push(' ');
                continue;
            }
            logical_line.push_str(&line);
            let line = std::mem::take(&mut logical_line);
            let cleaned = line.split(';').next().unwrap_or_default().trim();
            if cleaned.is_empty() {
                continue;
            }
            let active = conditions
                .iter()
                .all(|condition| condition.parent_active && condition.condition_active);
            if let Some(directive) = cleaned.strip_prefix('#') {
                let mut parts = directive.split_whitespace();
                let command = parts.next().unwrap_or_default().to_ascii_lowercase();
                match command.as_str() {
                    "ifdef" | "ifndef" | "if" => {
                        let parent_active = active;
                        let condition_active = if !parent_active {
                            false
                        } else {
                            evaluate_condition(&command, parts.collect::<Vec<_>>(), &self.defines)
                        };
                        conditions.push(Conditional {
                            parent_active,
                            condition_active,
                            seen_else: false,
                        });
                    }
                    "else" => {
                        let Some(condition) = conditions.last_mut() else {
                            return Err(parse_error(logical_start, "#else without matching #if"));
                        };
                        if condition.seen_else {
                            return Err(parse_error(logical_start, "duplicate #else"));
                        }
                        condition.seen_else = true;
                        condition.condition_active = !condition.condition_active;
                    }
                    "endif" => {
                        if conditions.pop().is_none() {
                            return Err(parse_error(logical_start, "#endif without matching #if"));
                        }
                    }
                    "define" if active => {
                        let name = parts
                            .next()
                            .ok_or_else(|| parse_error(logical_start, "#define requires a name"))?;
                        if !self.protected.contains(name) {
                            let value = parts.collect::<Vec<_>>().join(" ");
                            self.defines.insert(
                                name.to_string(),
                                if value.is_empty() { "1" } else { &value }.to_string(),
                            );
                        }
                    }
                    "undef" if active => {
                        if let Some(name) = parts.next()
                            && !self.protected.contains(name)
                        {
                            self.defines.remove(name);
                        }
                    }
                    "include" if active => {
                        let token = parts.next().ok_or_else(|| {
                            parse_error(logical_start, "#include requires a path")
                        })?;
                        let include = token
                            .strip_prefix('<')
                            .and_then(|value| value.strip_suffix('>'))
                            .or_else(|| {
                                token
                                    .strip_prefix('"')
                                    .and_then(|value| value.strip_suffix('"'))
                            })
                            .ok_or_else(|| {
                                parse_error(logical_start, "#include path must be quoted")
                            })?;
                        let resolved = self.resolve_include(Path::new(include), origin)?;
                        let nested = self.process_file(&resolved).map_err(|error| match error {
                            ItpError::Include { path, message } => {
                                ItpError::Include { path, message }
                            }
                            other => other,
                        })?;
                        output.extend(nested);
                    }
                    _ => {}
                }
            } else if active {
                output.push(SourceLine {
                    line: logical_start,
                    text: substitute_tokens(cleaned, &self.defines),
                });
            }
        }
        if !logical_line.is_empty() {
            let active = conditions
                .iter()
                .all(|condition| condition.parent_active && condition.condition_active);
            if active {
                output.push(SourceLine {
                    line: logical_start,
                    text: substitute_tokens(logical_line.trim(), &self.defines),
                });
            }
        }
        if !conditions.is_empty() {
            return Err(parse_error(input.lines().count().max(1), "missing #endif"));
        }
        Ok(output)
    }

    fn resolve_include(&self, path: &Path, origin: Option<&Path>) -> Result<PathBuf, ItpError> {
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(origin) = origin {
            origin.parent().unwrap_or_else(|| Path::new(".")).join(path)
        } else {
            path.to_path_buf()
        };
        if candidate.is_file() {
            return candidate.canonicalize().map_err(|error| ItpError::Include {
                path: candidate,
                message: error.to_string(),
            });
        }
        for directory in &self.options.include_dirs {
            let candidate = directory.join(path);
            if candidate.is_file() {
                return candidate.canonicalize().map_err(|error| ItpError::Include {
                    path: candidate,
                    message: error.to_string(),
                });
            }
        }
        Err(ItpError::Include {
            path: path.to_path_buf(),
            message: "file not found".to_string(),
        })
    }
}

fn evaluate_condition(command: &str, tokens: Vec<&str>, defines: &HashMap<String, String>) -> bool {
    match command {
        "ifdef" => tokens
            .first()
            .is_some_and(|name| defines.contains_key(*name)),
        "ifndef" => tokens
            .first()
            .is_none_or(|name| !defines.contains_key(*name)),
        "if" => {
            let expression = tokens.join(" ");
            let expression = expression.trim();
            if let Some(name) = expression
                .strip_prefix("defined(")
                .and_then(|value| value.strip_suffix(')'))
            {
                return defines.contains_key(name.trim());
            }
            if let Some(name) = expression
                .strip_prefix("!defined(")
                .and_then(|value| value.strip_suffix(')'))
            {
                return !defines.contains_key(name.trim());
            }
            defines
                .get(expression)
                .is_some_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        }
        _ => false,
    }
}

fn substitute_tokens(line: &str, defines: &HashMap<String, String>) -> String {
    line.split_whitespace()
        .map(|token| defines.get(token).map_or(token, String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_lines(lines: &[SourceLine]) -> Result<ItpData, ItpError> {
    let mut molecule_types = Vec::new();
    let mut current_molecule: Option<usize> = None;
    let mut atomtypes = HashMap::new();
    let mut molecules = Vec::new();
    let mut system = None;
    let mut section = String::new();

    for source in lines {
        let trimmed = source.text.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed[1..trimmed.len() - 1].trim().to_ascii_lowercase();
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        match section.as_str() {
            "atomtypes" => {
                let atomtype = parse_atomtype(trimmed, source.line)?;
                atomtypes.insert(atomtype.name.clone(), atomtype);
            }
            "moleculetype" => {
                let fields: Vec<&str> = trimmed.split_whitespace().collect();
                if fields.is_empty() {
                    return Err(parse_error(source.line, "moleculetype requires a name"));
                }
                let exclusions = fields.get(1).map(|value| {
                    parse_value::<usize>(value, source.line, "moleculetype exclusions")
                });
                let exclusions = match exclusions {
                    Some(Ok(value)) => Some(value),
                    Some(Err(error)) => return Err(error),
                    None => None,
                };
                molecule_types.push(ItpMoleculeType {
                    name: fields[0].to_string(),
                    exclusions,
                    ..ItpMoleculeType::default()
                });
                current_molecule = Some(molecule_types.len() - 1);
            }
            "molecules" => {
                let fields: Vec<&str> = trimmed.split_whitespace().collect();
                if fields.len() < 2 {
                    return Err(parse_error(
                        source.line,
                        "molecules row requires name and count",
                    ));
                }
                let count = parse_value(fields[1], source.line, "molecule count")?;
                molecules.push(ItpMoleculeCount {
                    name: fields[0].to_string(),
                    count,
                });
            }
            "system" => {
                if system.is_none() {
                    system = Some(trimmed.to_string());
                }
            }
            "atoms" => {
                let index = ensure_molecule(&mut molecule_types, &mut current_molecule);
                molecule_types[index]
                    .atoms
                    .push(parse_atom(trimmed, source.line)?);
            }
            "bonds" | "constraints" => {
                let index = ensure_molecule(&mut molecule_types, &mut current_molecule);
                let bond = parse_bond(trimmed, source.line)?;
                molecule_types[index].bonds.push(bond);
            }
            "angles" => {
                let index = ensure_molecule(&mut molecule_types, &mut current_molecule);
                molecule_types[index]
                    .angles
                    .push(parse_angle(trimmed, source.line)?);
            }
            "dihedrals" => {
                let index = ensure_molecule(&mut molecule_types, &mut current_molecule);
                let dihedral = parse_dihedral(trimmed, source.line)?;
                if matches!(dihedral.function, 2 | 4) {
                    molecule_types[index].impropers.push(dihedral);
                } else {
                    molecule_types[index].dihedrals.push(dihedral);
                }
            }
            "settles" => {
                let index = ensure_molecule(&mut molecule_types, &mut current_molecule);
                molecule_types[index]
                    .settles
                    .push(parse_settle(trimmed, source.line)?);
            }
            _ => {}
        }
    }
    if molecule_types.is_empty() {
        return Err(parse_error(
            1,
            "no [ moleculetype ] or [ atoms ] section found",
        ));
    }
    let mut data = ItpData {
        molecule_types,
        molecules,
        atomtypes,
        system,
        ..ItpData::default()
    };
    for molecule in &mut data.molecule_types {
        dedupe_molecule_interactions(molecule);
    }
    resolve_missing_atom_attributes(&mut data);
    expand_system(&mut data)?;
    validate_data(&data)?;
    Ok(data)
}

fn dedupe_molecule_interactions(molecule: &mut ItpMoleculeType) {
    let mut bonds = HashSet::new();
    molecule
        .bonds
        .retain(|bond| bonds.insert((bond.atom1, bond.atom2)));
    let mut angles = HashSet::new();
    molecule
        .angles
        .retain(|angle| angles.insert((angle.atom1, angle.atom2, angle.atom3)));
    let mut dihedrals = HashSet::new();
    molecule.dihedrals.retain(|dihedral| {
        dihedrals.insert((
            dihedral.atom1,
            dihedral.atom2,
            dihedral.atom3,
            dihedral.atom4,
        ))
    });
    let mut impropers = HashSet::new();
    molecule.impropers.retain(|improper| {
        impropers.insert((
            improper.atom1,
            improper.atom2,
            improper.atom3,
            improper.atom4,
        ))
    });
}

fn ensure_molecule(molecules: &mut Vec<ItpMoleculeType>, current: &mut Option<usize>) -> usize {
    if let Some(index) = *current {
        return index;
    }
    molecules.push(ItpMoleculeType {
        name: "SYSTEM".to_string(),
        exclusions: None,
        ..ItpMoleculeType::default()
    });
    let index = molecules.len() - 1;
    *current = Some(index);
    index
}

fn parse_atom(text: &str, line: usize) -> Result<ItpAtom, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 6 {
        return Err(parse_error(
            line,
            "[ atoms ] row requires at least six fields",
        ));
    }
    let id = parse_value(fields[0], line, "atom id")?;
    if id == 0 {
        return Err(parse_error(line, "atom ids must be positive"));
    }
    let residue_id = parse_value(fields[2], line, "residue id")?;
    let charge_group = parse_value(fields[5], line, "charge group").ok();
    let charge = parse_optional_float(fields.get(6).copied(), line, "charge")?;
    let mass = parse_optional_float(fields.get(7).copied(), line, "mass")?;
    Ok(ItpAtom {
        id,
        atom_type: fields[1].to_string(),
        residue_id,
        residue_name: fields[3].to_string(),
        name: fields[4].to_string(),
        charge_group,
        charge,
        mass,
        molecule_type: String::new(),
        molecule_number: 0,
    })
}

fn parse_atomtype(text: &str, line: usize) -> Result<ItpAtomType, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(parse_error(line, "[ atomtypes ] row is too short"));
    }
    let has_bonded = fields
        .get(1)
        .is_some_and(|field| field.parse::<f64>().is_err());
    let offset = 1 + usize::from(has_bonded);
    if fields.len() < offset + 3 {
        return Err(parse_error(
            line,
            "[ atomtypes ] row is missing mass or charge",
        ));
    }
    // GROMACS accepts both `type [bonded_type] at.num mass charge ptype ...`
    // and the shorter `type [bonded_type] mass charge ptype ...` form.  An
    // atomic number is only considered present when it is an integer and the
    // following fields have the standard numeric/nonnumeric shape.
    let has_atomic_number = fields
        .get(offset)
        .and_then(|value| value.parse::<u16>().ok())
        .is_some_and(|_| {
            fields
                .get(offset + 3)
                .is_some_and(|value| value.chars().all(char::is_alphabetic))
        });
    let atomic_number = has_atomic_number.then(|| fields[offset].parse::<u16>().unwrap());
    let value_offset = offset + usize::from(has_atomic_number);
    if fields.len() < value_offset + 3 {
        return Err(parse_error(
            line,
            "[ atomtypes ] row is missing mass or charge",
        ));
    }
    let mass = parse_optional_float(fields.get(value_offset).copied(), line, "atomtype mass")?;
    let charge = parse_optional_float(
        fields.get(value_offset + 1).copied(),
        line,
        "atomtype charge",
    )?;
    let particle_type = fields
        .get(value_offset + 2)
        .map(|value| (*value).to_string());
    Ok(ItpAtomType {
        name: fields[0].to_string(),
        bonded_type: has_bonded.then(|| fields[1].to_string()),
        atomic_number,
        mass,
        charge,
        particle_type,
        parameters: fields[(value_offset + 3)..]
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    })
}

fn parse_bond(text: &str, line: usize) -> Result<ItpBond, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 3 {
        return Err(parse_error(
            line,
            "bond row requires two atom ids and a function",
        ));
    }
    let atom1 = parse_value(fields[0], line, "bond atom id")?;
    let atom2 = parse_value(fields[1], line, "bond atom id")?;
    let function = parse_value(fields[2], line, "bond function")?;
    Ok(ItpBond {
        atom1,
        atom2,
        function,
        parameters: fields[3..]
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    })
}

fn parse_angle(text: &str, line: usize) -> Result<ItpAngle, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(parse_error(
            line,
            "angle row requires three atom ids and a function",
        ));
    }
    Ok(ItpAngle {
        atom1: parse_value(fields[0], line, "angle atom id")?,
        atom2: parse_value(fields[1], line, "angle atom id")?,
        atom3: parse_value(fields[2], line, "angle atom id")?,
        function: parse_value(fields[3], line, "angle function")?,
        parameters: fields[4..]
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    })
}

fn parse_dihedral(text: &str, line: usize) -> Result<ItpDihedral, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 5 {
        return Err(parse_error(
            line,
            "dihedral row requires four atom ids and a function",
        ));
    }
    Ok(ItpDihedral {
        atom1: parse_value(fields[0], line, "dihedral atom id")?,
        atom2: parse_value(fields[1], line, "dihedral atom id")?,
        atom3: parse_value(fields[2], line, "dihedral atom id")?,
        atom4: parse_value(fields[3], line, "dihedral atom id")?,
        function: parse_value(fields[4], line, "dihedral function")?,
        parameters: fields[5..]
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    })
}

fn parse_settle(text: &str, line: usize) -> Result<ItpSettle, ItpError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(parse_error(line, "settles row requires four fields"));
    }
    Ok(ItpSettle {
        atom: parse_value(fields[0], line, "settle atom id")?,
        function: parse_value(fields[1], line, "settle function")?,
        oxygen_hydrogen: fields[2].to_string(),
        hydrogen_hydrogen: fields[3].to_string(),
    })
}

fn parse_optional_float(
    value: Option<&str>,
    line: usize,
    field: &str,
) -> Result<Option<f64>, ItpError> {
    let Some(value) = value else { return Ok(None) };
    if value == "%%" || value.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let parsed = value.replace(['d', 'D'], "e");
    let number = parsed
        .parse::<f64>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))?;
    if !number.is_finite() {
        return Err(parse_error(line, format!("{field} must be finite")));
    }
    Ok(Some(number))
}

fn parse_value<T>(value: &str, line: usize, field: &str) -> Result<T, ItpError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| parse_error(line, format!("invalid {field} {value:?}: {error}")))
}

fn resolve_missing_atom_attributes(data: &mut ItpData) {
    for molecule in &mut data.molecule_types {
        for atom in &mut molecule.atoms {
            let atomtype = data.atomtypes.get(&atom.atom_type);
            if atom.charge.is_none() {
                atom.charge = atomtype.and_then(|value| value.charge);
            }
            if atom.mass.is_none() {
                atom.mass = atomtype.and_then(|value| value.mass).or_else(|| {
                    guess_element(&atom.name, Some(&atom.atom_type), Some(&atom.residue_name))
                        .ok()
                        .and_then(|element| guess_mass(&element).ok())
                });
            }
        }
    }
}

fn expand_system(data: &mut ItpData) -> Result<(), ItpError> {
    let definitions: HashMap<String, ItpMoleculeType> = data
        .molecule_types
        .iter()
        .cloned()
        .map(|molecule| (molecule.name.clone(), molecule))
        .collect();
    let composition = if data.molecules.is_empty() {
        data.molecule_types
            .iter()
            .map(|molecule| ItpMoleculeCount {
                name: molecule.name.clone(),
                count: 1,
            })
            .collect::<Vec<_>>()
    } else {
        data.molecules.clone()
    };
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut angles = Vec::new();
    let mut dihedrals = Vec::new();
    let mut impropers = Vec::new();
    let mut settles = Vec::new();
    let mut atom_offset = 0usize;
    let mut residue_offset = 0i32;
    let mut molecule_number = 0usize;
    for molecule_count in composition {
        let molecule = definitions
            .get(&molecule_count.name)
            .cloned()
            .ok_or_else(|| {
                ItpError::InvalidStructure(format!(
                    "[ molecules ] references unknown molecule type {}",
                    molecule_count.name
                ))
            })?;
        if molecule_count.count == 0 {
            continue;
        }
        let max_resid = molecule
            .atoms
            .iter()
            .map(|atom| atom.residue_id)
            .max()
            .unwrap_or(0);
        for _copy in 0..molecule_count.count {
            molecule_number += 1;
            let current_offset = atom_offset;
            let current_residue_offset = residue_offset;
            let mut id_map = HashMap::with_capacity(molecule.atoms.len());
            for (index, source) in molecule.atoms.iter().enumerate() {
                let id = current_offset + index + 1;
                id_map.insert(source.id, id);
                let mut atom = source.clone();
                atom.id = id;
                atom.residue_id = atom.residue_id.saturating_add(current_residue_offset);
                atom.molecule_type = molecule.name.clone();
                atom.molecule_number = molecule_number;
                atoms.push(atom);
            }
            for source in &molecule.bonds {
                if let (Some(&atom1), Some(&atom2)) =
                    (id_map.get(&source.atom1), id_map.get(&source.atom2))
                {
                    let mut bond = source.clone();
                    bond.atom1 = atom1;
                    bond.atom2 = atom2;
                    bonds.push(bond);
                }
            }
            for source in &molecule.angles {
                if let (Some(&atom1), Some(&atom2), Some(&atom3)) = (
                    id_map.get(&source.atom1),
                    id_map.get(&source.atom2),
                    id_map.get(&source.atom3),
                ) {
                    let mut angle = source.clone();
                    angle.atom1 = atom1;
                    angle.atom2 = atom2;
                    angle.atom3 = atom3;
                    angles.push(angle);
                }
            }
            for source in &molecule.dihedrals {
                if let (Some(&atom1), Some(&atom2), Some(&atom3), Some(&atom4)) = (
                    id_map.get(&source.atom1),
                    id_map.get(&source.atom2),
                    id_map.get(&source.atom3),
                    id_map.get(&source.atom4),
                ) {
                    let mut dihedral = source.clone();
                    dihedral.atom1 = atom1;
                    dihedral.atom2 = atom2;
                    dihedral.atom3 = atom3;
                    dihedral.atom4 = atom4;
                    dihedrals.push(dihedral);
                }
            }
            for source in &molecule.impropers {
                if let (Some(&atom1), Some(&atom2), Some(&atom3), Some(&atom4)) = (
                    id_map.get(&source.atom1),
                    id_map.get(&source.atom2),
                    id_map.get(&source.atom3),
                    id_map.get(&source.atom4),
                ) {
                    let mut improper = source.clone();
                    improper.atom1 = atom1;
                    improper.atom2 = atom2;
                    improper.atom3 = atom3;
                    improper.atom4 = atom4;
                    impropers.push(improper);
                }
            }
            for source in &molecule.settles {
                if let Some(&atom) = id_map.get(&source.atom) {
                    let mut settle = source.clone();
                    settle.atom = atom;
                    settles.push(settle.clone());
                    if let (Some(&hydrogen1), Some(&hydrogen2)) = (
                        id_map.get(&(source.atom + 1)),
                        id_map.get(&(source.atom + 2)),
                    ) {
                        bonds.push(ItpBond {
                            atom1: atom,
                            atom2: hydrogen1,
                            function: source.function,
                            parameters: vec!["settles".to_string()],
                        });
                        bonds.push(ItpBond {
                            atom1: atom,
                            atom2: hydrogen2,
                            function: source.function,
                            parameters: vec!["settles".to_string()],
                        });
                    }
                }
            }
            atom_offset = atoms.len();
            residue_offset = residue_offset.saturating_add(max_resid.max(1));
        }
    }
    data.atoms = atoms;
    data.bonds = bonds;
    data.angles = angles;
    data.dihedrals = dihedrals;
    data.impropers = impropers;
    data.settles = settles;
    Ok(())
}

fn validate_data(data: &ItpData) -> Result<(), ItpError> {
    if data.atoms.is_empty() {
        return Err(ItpError::InvalidStructure(
            "topology contains no atoms".to_string(),
        ));
    }
    let ids: HashSet<usize> = data.atoms.iter().map(|atom| atom.id).collect();
    if ids.len() != data.atoms.len() {
        return Err(ItpError::InvalidStructure(
            "expanded atom ids are not unique".to_string(),
        ));
    }
    for atom in &data.atoms {
        if atom.atom_type.trim().is_empty()
            || atom.name.trim().is_empty()
            || atom.residue_name.trim().is_empty()
        {
            return Err(ItpError::InvalidStructure(format!(
                "atom {} has an empty topology field",
                atom.id
            )));
        }
        if atom.charge.is_some_and(|value| !value.is_finite())
            || atom
                .mass
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(ItpError::InvalidStructure(format!(
                "atom {} has invalid charge or mass",
                atom.id
            )));
        }
    }
    for bond in &data.bonds {
        if !ids.contains(&bond.atom1) || !ids.contains(&bond.atom2) || bond.atom1 == bond.atom2 {
            return Err(ItpError::InvalidStructure(format!(
                "bond references invalid atoms ({}, {})",
                bond.atom1, bond.atom2
            )));
        }
    }
    Ok(())
}

fn parse_error(line: usize, message: impl Into<String>) -> ItpError {
    ItpError::Parse {
        line,
        message: message.into(),
    }
}

impl Universe {
    /// Construct a zero-coordinate universe from an ITP/TOP file.
    pub fn from_itp(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_itp_data(ItpData::read_file(path)?)
    }

    /// Construct a zero-coordinate universe from an in-memory ITP/TOP file.
    pub fn from_itp_str(input: &str) -> crate::Result<Self> {
        Self::from_itp_data(ItpData::from_str(input)?)
    }

    /// Construct a universe from an ITP/TOP file and a PDB trajectory.
    pub fn from_itp_and_pdb(
        itp_path: impl AsRef<Path>,
        pdb_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let topology = ItpData::read_file(itp_path)?;
        let coordinates = Self::from_pdb(pdb_path)?;
        Self::from_itp_data_and_coordinates(topology, coordinates)
    }

    /// Construct a universe from in-memory ITP/TOP and PDB documents.
    pub fn from_itp_and_pdb_str(itp: &str, pdb: &str) -> crate::Result<Self> {
        let topology = ItpData::from_str(itp)?;
        let coordinates = Self::from_pdb_str(pdb)?;
        Self::from_itp_data_and_coordinates(topology, coordinates)
    }

    /// Construct a universe from an ITP/TOP file and a GRO trajectory.
    pub fn from_itp_and_gro(
        itp_path: impl AsRef<Path>,
        gro_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let topology = ItpData::read_file(itp_path)?;
        let coordinates = Self::from_gro(gro_path)?;
        Self::from_itp_data_and_coordinates(topology, coordinates)
    }

    /// Construct a universe from in-memory ITP/TOP and GRO documents.
    pub fn from_itp_and_gro_str(itp: &str, gro: &str) -> crate::Result<Self> {
        let topology = ItpData::from_str(itp)?;
        let coordinates = Self::from_gro_str(gro)?;
        Self::from_itp_data_and_coordinates(topology, coordinates)
    }

    fn from_itp_data(data: ItpData) -> crate::Result<Self> {
        let mut atoms = Vec::with_capacity(data.atoms.len());
        for (index, source) in data.atoms.iter().enumerate() {
            let element = guess_element(
                &source.name,
                Some(&source.atom_type),
                Some(&source.residue_name),
            )
            .ok();
            let mass = source
                .mass
                .or_else(|| element.as_deref().and_then(|value| guess_mass(value).ok()))
                .unwrap_or(0.0);
            let mut atom = Atom::new(index, source.name.clone(), [0.0, 0.0, 0.0]).with_mass(mass);
            atom.atom_type = Some(source.atom_type.clone());
            atom.element = element;
            atom.charge = source.charge.unwrap_or(0.0);
            atom.resid = source.residue_id;
            atom.resname = source.residue_name.clone();
            atom.segid = if source.molecule_type.is_empty() {
                "SYSTEM".to_string()
            } else {
                source.molecule_type.clone()
            };
            atoms.push(atom);
        }
        let mut topology = Topology::new(atoms);
        for source in &data.bonds {
            let Some(atom1) = source.atom1.checked_sub(1) else {
                continue;
            };
            let Some(atom2) = source.atom2.checked_sub(1) else {
                continue;
            };
            if atom1 < topology.atoms.len() && atom2 < topology.atoms.len() {
                let mut bond = Bond::new(atom1, atom2);
                bond.order = u8::try_from(source.function).ok();
                topology.add_bond(bond);
            }
        }
        Ok(Self {
            topology,
            trajectory: Trajectory::new(vec![Frame::new(
                data.atoms.iter().map(|_| [0.0, 0.0, 0.0]).collect(),
            )]),
        })
    }

    fn from_itp_data_and_coordinates(data: ItpData, coordinates: Self) -> crate::Result<Self> {
        let mut universe = Self::from_itp_data(data)?;
        if coordinates.n_atoms() != universe.n_atoms() {
            return Err(crate::Error::InvalidInput(format!(
                "coordinate file contains {} atoms, ITP contains {}",
                coordinates.n_atoms(),
                universe.n_atoms()
            )));
        }
        if coordinates.trajectory.frames.is_empty() {
            return Err(crate::Error::InvalidInput(
                "coordinate file has no frames".to_string(),
            ));
        }
        universe.trajectory = coordinates.trajectory;
        Ok(universe)
    }
}

/// Read an ITP/TOP file into an owned topology.
pub fn read_itp_file(path: impl AsRef<Path>) -> Result<ItpData, ItpError> {
    read_itp(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC: &str = concat!(
        "[ atomtypes ]\n",
        "C 6 12.011 0.0 A 0.3 0.2\n",
        "H 1 1.008 0.0 A 0.2 0.1\n",
        "[ moleculetype ]\n",
        "METH 3\n",
        "[ atoms ]\n",
        "1 C 1 MET C1 1 -0.2\n",
        "2 H 1 MET H1 1 0.2\n",
        "[ bonds ]\n",
        "1 2 1 0.109\n",
        "[ angles ]\n",
        "1 2 1 1 109.5\n",
        "[ dihedrals ]\n",
        "1 2 1 2 1 0.0\n",
        "1 2 1 2 2 0.0\n",
    );

    #[test]
    fn parses_atoms_types_and_connectivity() {
        let data = ItpData::from_str(BASIC).expect("valid ITP");
        assert_eq!(data.n_atoms(), 2);
        assert_eq!(data.atoms[0].mass, Some(12.011));
        assert_eq!(data.atoms[0].charge, Some(-0.2));
        assert_eq!(data.bonds.len(), 1);
        assert_eq!(data.angles.len(), 1);
        assert_eq!(data.dihedrals.len(), 1);
        assert_eq!(data.impropers.len(), 1);
    }

    #[test]
    fn preprocessor_defines_and_overrides_are_applied() {
        let input = concat!(
            "#define CHARGE 0.1\n",
            "#ifndef EXTRA\n",
            "[ moleculetype ]\nM 1\n[ atoms ]\n1 H 1 W H 1 CHARGE 1.0\n",
            "#else\n",
            "[ moleculetype ]\nM 1\n[ atoms ]\n1 H 1 W H 1 0.2 1.0\n",
            "#endif\n",
        );
        let mut options = ItpOptions::default();
        options.define_value("EXTRA", "1");
        let data = ItpData::from_str_with_options(input, options).expect("valid ITP");
        assert_eq!(data.atoms[0].charge, Some(0.2));
    }

    #[test]
    fn settles_generate_water_bonds() {
        let input = concat!(
            "[ moleculetype ]\nW 2\n[ atoms ]\n",
            "1 OW 1 SOL OW 1 -0.8 16\n",
            "2 H 1 SOL H1 1 0.4 1\n",
            "3 H 1 SOL H2 1 0.4 1\n",
            "[ settles ]\n1 1 0.1 0.15\n",
        );
        let data = ItpData::from_str(input).expect("valid ITP");
        assert_eq!(data.bonds.len(), 2);
        assert_eq!(data.bonds[0].parameters, vec!["settles"]);
    }

    #[test]
    fn universe_constructor_maps_metadata_and_bonds() {
        let universe = Universe::from_itp_str(BASIC).expect("valid ITP");
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.n_residues(), 1);
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("C"));
        let bond = &universe.topology.bonds[0];
        assert_eq!((bond.atom1, bond.atom2, bond.order), (0, 1, Some(1)));
    }

    #[test]
    fn composition_expands_molecules_and_remaps_bonds() {
        let input = concat!(
            "[ moleculetype ]\nW 1\n[ atoms ]\n",
            "1 OW 1 SOL OW 1 -0.8 16\n",
            "2 H 1 SOL H 1 0.4 1\n",
            "[ bonds ]\n1 2 1\n",
            "[ molecules ]\nW 2\n",
        );
        let data = ItpData::from_str(input).expect("valid ITP");
        assert_eq!(data.n_atoms(), 4);
        assert_eq!((data.bonds[1].atom1, data.bonds[1].atom2), (3, 4));
        assert_ne!(data.atoms[0].residue_id, data.atoms[2].residue_id);
    }

    #[test]
    fn malformed_atom_rows_are_rejected() {
        let input = "[ moleculetype ]\nM 1\n[ atoms ]\n1 H 1 W\n";
        assert!(matches!(
            ItpData::from_str(input),
            Err(ItpError::Parse { .. })
        ));
    }

    #[test]
    fn parses_migrated_gromacs_fixture() {
        let path =
            std::path::Path::new("../mdanalysis/testsuite/MDAnalysisTests/data/gromacs_ala10.itp");
        let data = ItpData::read_file(path).expect("GROMACS ITP fixture should parse");
        assert_eq!(data.n_atoms(), 63);
        assert_eq!(data.bonds.len(), 62);
        assert_eq!(data.angles.len(), 91);
        assert_eq!(data.dihedrals.len(), 30);
        assert_eq!(data.impropers.len(), 29);
        assert_eq!(data.atoms[0].residue_id, 2);
        assert_eq!(data.atoms[62].name, "HO");
    }

    #[test]
    fn resolves_atomtype_masses_and_missing_mass_fixture() {
        let path =
            std::path::Path::new("../mdanalysis/testsuite/MDAnalysisTests/data/itp_nomass.itp");
        let data = ItpData::read_file(path).expect("ATB ITP fixture should parse");
        assert_eq!(data.n_atoms(), 60);
        assert_eq!(data.atoms[0].mass, Some(1.008));
        assert_eq!(data.atoms[1].mass, Some(12.011));
    }

    #[test]
    fn parses_migrated_top_with_relative_includes() {
        let path =
            std::path::Path::new("../mdanalysis/testsuite/MDAnalysisTests/data/gromacs_ala10.top");
        let mut options = ItpOptions::default();
        options.include_dir(path.parent().expect("fixture has parent").join("gromacs"));
        let data =
            ItpData::read_file_with_options(path, options).expect("TOP fixture should parse");
        assert_eq!(data.n_atoms(), 135);
        assert_eq!(data.molecules.len(), 2);
    }

    #[test]
    fn parses_atomtypes_and_atom_charge_fixture() {
        let atomtypes =
            std::path::Path::new("../mdanalysis/testsuite/MDAnalysisTests/data/atomtypes.itp");
        let data = ItpData::read_file(atomtypes).expect("atomtypes fixture should parse");
        assert_eq!(data.n_atoms(), 4);
        assert_eq!(data.atoms[0].charge, Some(4.0));
        assert_eq!(data.atoms[0].mass, Some(8.0));
        assert_eq!(data.atoms[1].mass, Some(20.989));
        assert_eq!(data.atoms[2].mass, Some(20.989));
        assert_eq!(data.atoms[3].mass, Some(1.008));
    }
}
