//! Atom selection expressions.
//!
//! The selection parser operates on the small [`AtomLike`] trait instead of a
//! concrete topology type, so it can be used with any atom representation.

use std::fmt;

/// The atom attributes needed by [`Selection`].
pub trait AtomLike {
    /// Atom name (for example, `CA`).
    fn name(&self) -> &str;
    /// Residue name (for example, `ALA`).
    fn resname(&self) -> &str;
    /// Residue identifier. Negative residue identifiers are valid.
    fn resid(&self) -> i32;
    /// Zero-based atom index.
    fn index(&self) -> usize;
    /// Chemical element (for example, `C`), when one is available.
    fn element(&self) -> Option<&str>;
    /// Chain identifier.
    fn chain_id(&self) -> &str;
    /// Segment identifier.
    fn segid(&self) -> &str;

    /// Force-field atom type, when available.
    fn atom_type(&self) -> Option<&str> {
        None
    }

    /// Cartesian coordinates used by spatial selectors.
    fn position(&self) -> [f64; 3] {
        [0.0; 3]
    }

    /// Atomic mass, when available.
    fn mass(&self) -> Option<f64> {
        None
    }

    /// Partial charge, when available.
    fn charge(&self) -> Option<f64> {
        None
    }
}

/// Errors produced while lexing or parsing a selection expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// The input contained no expression.
    EmptyExpression,
    /// An unexpected character was found at a byte offset.
    UnexpectedCharacter { position: usize, character: char },
    /// The parser reached the end of input while expecting another token.
    UnexpectedEnd { context: &'static str },
    /// A token was not valid in the current position.
    UnexpectedToken { position: usize, token: String },
    /// A predicate name was not supported.
    UnknownPredicate(String),
    /// A predicate value had the wrong form.
    InvalidValue { predicate: String, value: String },
    /// A range was written backwards.
    InvalidRange { start: i64, end: i64 },
    /// A named atom group was requested but was not supplied to the selection
    /// context.
    UnknownGroup(String),
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExpression => f.write_str("selection expression is empty"),
            Self::UnexpectedCharacter {
                position,
                character,
            } => write!(f, "unexpected character {character:?} at byte {position}"),
            Self::UnexpectedEnd { context } => {
                write!(f, "unexpected end of selection while parsing {context}")
            }
            Self::UnexpectedToken { position, token } => {
                write!(f, "unexpected token {token:?} at byte {position}")
            }
            Self::UnknownPredicate(predicate) => {
                write!(f, "unknown selection predicate {predicate:?}")
            }
            Self::InvalidValue { predicate, value } => {
                write!(f, "invalid value {value:?} for predicate {predicate:?}")
            }
            Self::InvalidRange { start, end } => {
                write!(f, "selection range cannot descend from {start} to {end}")
            }
            Self::UnknownGroup(group) => write!(f, "unknown selection group {group:?}"),
        }
    }
}

impl std::error::Error for SelectionError {}

/// A parsed selection expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    expression: Expr,
}

impl Selection {
    /// Parse a selection expression.
    pub fn parse(input: &str) -> Result<Self, SelectionError> {
        let tokens = Lexer::new(input).lex()?;
        if tokens.is_empty() {
            return Err(SelectionError::EmptyExpression);
        }

        let mut parser = Parser::new(tokens);
        let expression = parser.parse_expression()?;
        if let Some(token) = parser.peek() {
            return Err(SelectionError::UnexpectedToken {
                position: token.position(),
                token: token.display(),
            });
        }
        Ok(Self { expression })
    }

    /// Test one atom against this selection.
    pub fn matches<A: AtomLike>(&self, atom: &A) -> bool {
        self.expression.matches(atom, std::slice::from_ref(atom))
    }

    /// Apply this selection to a slice, preserving the input order.
    pub fn apply<'a, A: AtomLike>(&self, atoms: &'a [A]) -> Vec<&'a A> {
        self.apply_with_bonds(atoms, &[])
    }

    /// Apply this selection with an optional zero-based bond table.
    pub fn apply_with_bonds<'a, A: AtomLike>(
        &self,
        atoms: &'a [A],
        bonds: &[(usize, usize)],
    ) -> Vec<&'a A> {
        atoms
            .iter()
            .filter(|atom| self.expression.matches_with_bonds(*atom, atoms, bonds))
            .collect()
    }

    /// Apply this selection against named groups and preserve input order.
    ///
    /// Each tuple contains a group name and the zero-based atom indices in
    /// that group.  The group names are resolved before matching so a typo is
    /// reported as [`SelectionError::UnknownGroup`] rather than silently
    /// producing an empty selection.
    pub fn apply_with_groups<'a, A: AtomLike>(
        &self,
        atoms: &'a [A],
        groups: &[(&str, &[usize])],
    ) -> Result<Vec<&'a A>, SelectionError> {
        self.apply_with_bonds_and_groups(atoms, &[], groups)
    }

    /// Apply this selection with both a bond table and named atom groups.
    pub fn apply_with_bonds_and_groups<'a, A: AtomLike>(
        &self,
        atoms: &'a [A],
        bonds: &[(usize, usize)],
        groups: &[(&str, &[usize])],
    ) -> Result<Vec<&'a A>, SelectionError> {
        self.apply_with_bonds_and_groups_and_global_atoms(atoms, bonds, groups, None)
    }

    /// Apply a selection with topology, named groups, and an optional global
    /// atom scope used by modifiers such as `around ... global ...`.
    pub fn apply_with_bonds_and_groups_and_global_atoms<'a, A: AtomLike>(
        &self,
        atoms: &'a [A],
        bonds: &[(usize, usize)],
        groups: &[(&str, &[usize])],
        global_atoms: Option<&[A]>,
    ) -> Result<Vec<&'a A>, SelectionError> {
        self.validate_groups(groups)?;
        Ok(atoms
            .iter()
            .filter(|atom| {
                self.expression
                    .matches_with_context(*atom, atoms, bonds, groups, global_atoms)
            })
            .collect())
    }

    /// Apply a selection where a root `global` modifier changes the output
    /// scope to `global_atoms`. Nested `global` modifiers, such as those in an
    /// `around` expression, still return atoms from the local scope.
    pub fn apply_with_global_scope<'a, A: AtomLike>(
        &self,
        atoms: &'a [A],
        bonds: &[(usize, usize)],
        groups: &[(&str, &[usize])],
        global_atoms: &'a [A],
    ) -> Result<Vec<&'a A>, SelectionError> {
        self.validate_groups(groups)?;
        let output_atoms = if self.expression_is_global_root() {
            global_atoms
        } else {
            atoms
        };
        Ok(output_atoms
            .iter()
            .filter(|atom| {
                self.expression.matches_with_context(
                    *atom,
                    output_atoms,
                    bonds,
                    groups,
                    Some(global_atoms),
                )
            })
            .collect())
    }

    fn validate_groups(&self, groups: &[(&str, &[usize])]) -> Result<(), SelectionError> {
        let mut names = Vec::new();
        self.expression.group_names(&mut names);
        for name in names {
            if !groups.iter().any(|(candidate, _)| *candidate == name) {
                return Err(SelectionError::UnknownGroup(name));
            }
        }
        Ok(())
    }

    /// Alias for [`Selection::apply`].
    pub fn select<'a, A: AtomLike>(&self, atoms: &'a [A]) -> Vec<&'a A> {
        self.apply(atoms)
    }

    pub(crate) fn expression_is_global_root(&self) -> bool {
        matches!(self.expression, Expr::Global(_))
    }
}

/// Parse `expression` and return matching atoms in input order.
pub fn select<'a, A: AtomLike>(
    atoms: &'a [A],
    expression: &str,
) -> Result<Vec<&'a A>, SelectionError> {
    Selection::parse(expression).map(|selection| selection.apply(atoms))
}

/// Parse `expression` and return matching atoms using a zero-based bond table.
pub fn select_with_bonds<'a, A: AtomLike>(
    atoms: &'a [A],
    expression: &str,
    bonds: &[(usize, usize)],
) -> Result<Vec<&'a A>, SelectionError> {
    Selection::parse(expression).map(|selection| selection.apply_with_bonds(atoms, bonds))
}

/// Parse `expression` and return matching atoms using named groups.
pub fn select_with_groups<'a, A: AtomLike>(
    atoms: &'a [A],
    expression: &str,
    groups: &[(&str, &[usize])],
) -> Result<Vec<&'a A>, SelectionError> {
    Selection::parse(expression)?.apply_with_groups(atoms, groups)
}

/// Parse and evaluate a selection with topology, named groups, and a global
/// atom scope used by `global` modifiers.
pub fn select_with_bonds_and_groups<'a, A: AtomLike>(
    atoms: &'a [A],
    expression: &str,
    bonds: &[(usize, usize)],
    groups: &[(&str, &[usize])],
    global_atoms: Option<&'a [A]>,
) -> Result<Vec<&'a A>, SelectionError> {
    Selection::parse(expression)?.apply_with_global_scope(
        atoms,
        bonds,
        groups,
        global_atoms.unwrap_or(atoms),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Expr {
    All,
    None,
    Predicate(Predicate),
    Protein,
    Backbone,
    Water,
    Nucleic,
    NucleicBackbone,
    NucleicBase,
    NucleicSugar,
    Around {
        cutoff: FloatValue,
        selection: Box<Self>,
    },
    Point {
        point: [FloatValue; 3],
        cutoff: FloatValue,
    },
    Same {
        property: SameProperty,
        selection: Box<Self>,
    },
    Atom {
        segid: String,
        resid: i64,
        name: String,
    },
    ByRes(Box<Self>),
    Global(Box<Self>),
    Bonded(Box<Self>),
    Group(String),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
}

impl Expr {
    fn matches<A: AtomLike>(&self, atom: &A, atoms: &[A]) -> bool {
        self.matches_with_bonds(atom, atoms, &[])
    }

    fn matches_with_bonds<A: AtomLike>(
        &self,
        atom: &A,
        atoms: &[A],
        bonds: &[(usize, usize)],
    ) -> bool {
        self.matches_with_context(atom, atoms, bonds, &[], None)
    }

    fn matches_with_context<A: AtomLike>(
        &self,
        atom: &A,
        atoms: &[A],
        bonds: &[(usize, usize)],
        groups: &[(&str, &[usize])],
        global_atoms: Option<&[A]>,
    ) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Predicate(predicate) => predicate.matches(atom),
            Self::Protein => protein_resnames()
                .iter()
                .any(|name| atom.resname() == *name),
            Self::Backbone => {
                protein_resnames()
                    .iter()
                    .any(|name| atom.resname() == *name)
                    && matches!(atom.name(), "N" | "CA" | "C" | "O")
            }
            Self::Water => water_resnames().iter().any(|name| atom.resname() == *name),
            Self::Nucleic => nucleic_resnames()
                .iter()
                .any(|name| atom.resname() == *name),
            Self::NucleicBackbone => {
                nucleic_resnames()
                    .iter()
                    .any(|name| atom.resname() == *name)
                    && matches!(atom.name(), "P" | "C5'" | "C3'" | "O3'" | "O5'")
            }
            Self::NucleicBase => {
                nucleic_resnames()
                    .iter()
                    .any(|name| atom.resname() == *name)
                    && matches!(
                        atom.name(),
                        "N9" | "N7"
                            | "C8"
                            | "C5"
                            | "C4"
                            | "N3"
                            | "C2"
                            | "N1"
                            | "C6"
                            | "O6"
                            | "N2"
                            | "N6"
                            | "O2"
                            | "N4"
                            | "O4"
                            | "C5M"
                    )
            }
            Self::NucleicSugar => {
                nucleic_resnames()
                    .iter()
                    .any(|name| atom.resname() == *name)
                    && matches!(atom.name(), "C1'" | "C2'" | "C3'" | "C4'" | "O4'")
            }
            Self::Around { cutoff, selection } => {
                if selection.matches_with_context(atom, atoms, bonds, groups, global_atoms) {
                    return false;
                }
                let cutoff = cutoff.0;
                let cutoff_squared = cutoff * cutoff;
                let reference_atoms = if selection.contains_global() {
                    global_atoms.unwrap_or(atoms)
                } else {
                    atoms
                };
                reference_atoms.iter().any(|reference| {
                    selection.matches_with_context(reference, atoms, bonds, groups, global_atoms)
                        && squared_distance(atom.position(), reference.position()) < cutoff_squared
                })
            }
            Self::Point { point, cutoff } => {
                let point = [point[0].0, point[1].0, point[2].0];
                squared_distance(atom.position(), point) < cutoff.0 * cutoff.0
            }
            Self::Same {
                property,
                selection,
            } => atoms.iter().any(|reference| {
                selection.matches_with_context(reference, atoms, bonds, groups, global_atoms)
                    && property.matches(atom, reference)
            }),
            Self::Atom { segid, resid, name } => {
                atom.segid() == segid && i64::from(atom.resid()) == *resid && atom.name() == name
            }
            Self::ByRes(selection) => atoms.iter().any(|reference| {
                selection.matches_with_context(reference, atoms, bonds, groups, global_atoms)
                    && reference.resid() == atom.resid()
                    && reference.segid() == atom.segid()
            }),
            Self::Global(selection) => {
                selection.matches_with_context(atom, atoms, bonds, groups, global_atoms)
            }
            Self::Bonded(selection) => bonds.iter().any(|(left, right)| {
                let neighbor_index = if *left == atom.index() {
                    Some(*right)
                } else if *right == atom.index() {
                    Some(*left)
                } else {
                    None
                };
                neighbor_index.is_some_and(|index| {
                    atoms
                        .iter()
                        .find(|candidate| candidate.index() == index)
                        .is_some_and(|candidate| {
                            selection.matches_with_context(
                                candidate,
                                atoms,
                                bonds,
                                groups,
                                global_atoms,
                            )
                        })
                })
            }),
            Self::Group(name) => groups
                .iter()
                .find(|(candidate, _)| *candidate == name)
                .is_some_and(|(_, indices)| indices.contains(&atom.index())),
            Self::And(left, right) => {
                left.matches_with_context(atom, atoms, bonds, groups, global_atoms)
                    && right.matches_with_context(atom, atoms, bonds, groups, global_atoms)
            }
            Self::Or(left, right) => {
                left.matches_with_context(atom, atoms, bonds, groups, global_atoms)
                    || right.matches_with_context(atom, atoms, bonds, groups, global_atoms)
            }
            Self::Not(expression) => {
                !expression.matches_with_context(atom, atoms, bonds, groups, global_atoms)
            }
        }
    }

    fn group_names(&self, names: &mut Vec<String>) {
        match self {
            Self::Group(name) => names.push(name.clone()),
            Self::Around { selection, .. }
            | Self::Same { selection, .. }
            | Self::ByRes(selection)
            | Self::Global(selection)
            | Self::Bonded(selection)
            | Self::Not(selection) => selection.group_names(names),
            Self::And(left, right) | Self::Or(left, right) => {
                left.group_names(names);
                right.group_names(names);
            }
            _ => {}
        }
    }

    fn contains_global(&self) -> bool {
        match self {
            Self::Global(_) => true,
            Self::Around { selection, .. }
            | Self::Same { selection, .. }
            | Self::ByRes(selection)
            | Self::Bonded(selection)
            | Self::Not(selection) => selection.contains_global(),
            Self::And(left, right) | Self::Or(left, right) => {
                left.contains_global() || right.contains_global()
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Predicate {
    Name(Vec<String>),
    Resname(Vec<String>),
    Resid(Vec<IntRange>),
    Index(Vec<IntRange>),
    ByNum(Vec<IntRange>),
    Element(Vec<String>),
    ChainId(Vec<String>),
    Segid(Vec<String>),
    Type(Vec<String>),
    Prop {
        axis: Axis,
        operator: Comparison,
        value: FloatValue,
        absolute: bool,
    },
}

impl Predicate {
    fn matches<A: AtomLike>(&self, atom: &A) -> bool {
        match self {
            Self::Name(values) => values
                .iter()
                .any(|value| matches_pattern(atom.name(), value)),
            Self::Resname(values) => values
                .iter()
                .any(|value| matches_pattern(atom.resname(), value)),
            Self::Resid(ranges) => ranges
                .iter()
                .any(|range| range.contains(i64::from(atom.resid()))),
            Self::Index(ranges) => i64::try_from(atom.index())
                .map(|index| ranges.iter().any(|range| range.contains(index)))
                .unwrap_or(false),
            Self::ByNum(ranges) => i64::try_from(atom.index())
                .ok()
                .and_then(|index| index.checked_add(1))
                .map(|index| ranges.iter().any(|range| range.contains(index)))
                .unwrap_or(false),
            Self::Element(values) => atom
                .element()
                .is_some_and(|element| values.iter().any(|value| matches_pattern(element, value))),
            Self::ChainId(values) => values
                .iter()
                .any(|value| matches_pattern(atom.chain_id(), value)),
            Self::Segid(values) => values
                .iter()
                .any(|value| matches_pattern(atom.segid(), value)),
            Self::Type(values) => atom.atom_type().is_some_and(|atom_type| {
                values.iter().any(|value| matches_pattern(atom_type, value))
            }),
            Self::Prop {
                axis,
                operator,
                value,
                absolute,
            } => {
                let mut actual = atom.position()[axis.index()];
                if *absolute {
                    actual = actual.abs();
                }
                operator.matches(actual, value.0)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FloatValue(f64);

impl PartialEq for FloatValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatValue {}

impl From<f64> for FloatValue {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Comparison {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
}

impl Comparison {
    fn matches(self, left: f64, right: f64) -> bool {
        match self {
            Self::Less => left < right,
            Self::LessEqual => left <= right,
            Self::Equal => (left - right).abs() <= 1.0e-6,
            Self::NotEqual => (left - right).abs() > 1.0e-6,
            Self::Greater => left > right,
            Self::GreaterEqual => left >= right,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SameProperty {
    X,
    Y,
    Z,
    Resid,
    Residue,
    Resname,
    Name,
    Element,
    Type,
    Segid,
    Mass,
    Charge,
}

impl SameProperty {
    fn matches<A: AtomLike>(self, atom: &A, reference: &A) -> bool {
        match self {
            Self::X | Self::Y | Self::Z => {
                let axis = match self {
                    Self::X => 0,
                    Self::Y => 1,
                    Self::Z => 2,
                    _ => unreachable!(),
                };
                (atom.position()[axis] - reference.position()[axis]).abs() <= 1.0e-6
            }
            Self::Resid => atom.resid() == reference.resid(),
            Self::Residue => atom.resid() == reference.resid() && atom.segid() == reference.segid(),
            Self::Resname => atom.resname() == reference.resname(),
            Self::Name => atom.name() == reference.name(),
            Self::Element => atom.element() == reference.element(),
            Self::Type => atom.atom_type() == reference.atom_type(),
            Self::Segid => atom.segid() == reference.segid(),
            Self::Mass => match (atom.mass(), reference.mass()) {
                (Some(left), Some(right)) => (left - right).abs() <= 1.0e-6,
                _ => false,
            },
            Self::Charge => match (atom.charge(), reference.charge()) {
                (Some(left), Some(right)) => (left - right).abs() <= 1.0e-6,
                _ => false,
            },
        }
    }
}

fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

fn protein_resnames() -> &'static [&'static str] {
    &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HSD", "HSE", "HSP", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "CYS2",
        "CYSH", "HID", "HIE", "HIP", "ASH", "GLH", "ACE", "NME", "ARGN", "ASPH", "QLN", "PGLU",
        "GLUH", "HIS1", "HISD", "HISE", "HISH", "LYSH", "ASN1", "CYS1", "HISA", "HISB", "HIS2",
        "ORN", "DAB", "LYN", "HYP", "CYM", "CYX", "NALA", "NGLY", "NSER", "NTHR", "NLEU", "NILE",
        "NVAL", "NASN", "NGLN", "NARG", "NHID", "NHIE", "NHIP", "NTRP", "NPHE", "NTYR", "NGLU",
        "NASP", "NLYS", "NPRO", "NCYS", "NCYX", "NMET", "CALA", "CGLY", "CSER", "CTHR", "CLEU",
        "CILE", "CVAL", "CASF", "CASN", "CGLN", "CARG", "CHID", "CHIE", "CHIP", "CTRP", "CPHE",
        "CTYR", "CGLU", "CASP", "CLYS", "CPRO", "CCYS", "CCYX", "CMET", "CME", "ASF",
    ]
}

fn water_resnames() -> &'static [&'static str] {
    &[
        "H2O", "HOH", "OH2", "HHO", "OHH", "T3P", "T4P", "T5P", "SOL", "WAT", "TIP", "TIP2",
        "TIP3", "TIP4",
    ]
}

fn nucleic_resnames() -> &'static [&'static str] {
    &[
        "ADE", "URA", "CYT", "GUA", "THY", "DA", "DC", "DG", "DT", "RA", "RU", "RG", "RC", "A",
        "T", "U", "C", "G", "DA5", "DC5", "DG5", "DT5", "DA3", "DC3", "DG3", "DT3", "RA5", "RU5",
        "RG5", "RC5", "RA3", "RU3", "RG3", "RC3",
    ]
}

/// Match the shell-style wildcards accepted by MDAnalysis selectors.
/// `*` matches any sequence, `?` matches one character, and bracket classes
/// such as `[NY]` or `[!NY]` match one selected character.
fn matches_pattern(value: &str, pattern: &str) -> bool {
    let value: Vec<char> = value.chars().collect();
    let pattern: Vec<char> = pattern.chars().collect();
    let mut memo = vec![vec![None; pattern.len() + 1]; value.len() + 1];

    fn visit(
        value: &[char],
        pattern: &[char],
        i: usize,
        j: usize,
        memo: &mut [Vec<Option<bool>>],
    ) -> bool {
        if let Some(result) = memo[i][j] {
            return result;
        }
        let result = if j == pattern.len() {
            i == value.len()
        } else if pattern[j] == '*' {
            visit(value, pattern, i, j + 1, memo)
                || (i < value.len() && visit(value, pattern, i + 1, j, memo))
        } else if i == value.len() {
            false
        } else if pattern[j] == '?' {
            visit(value, pattern, i + 1, j + 1, memo)
        } else if pattern[j] == '[' {
            let mut end = j + 1;
            while end < pattern.len() && pattern[end] != ']' {
                end += 1;
            }
            if end == pattern.len() {
                pattern[j] == value[i] && visit(value, pattern, i + 1, j + 1, memo)
            } else {
                let mut offset = j + 1;
                let negated = pattern.get(offset).is_some_and(|c| *c == '!' || *c == '^');
                if negated {
                    offset += 1;
                }
                let mut matched = false;
                while offset < end {
                    if offset + 2 < end && pattern[offset + 1] == '-' {
                        matched |= pattern[offset] <= value[i] && value[i] <= pattern[offset + 2];
                        offset += 3;
                    } else {
                        matched |= pattern[offset] == value[i];
                        offset += 1;
                    }
                }
                if negated {
                    matched = !matched;
                }
                matched && visit(value, pattern, i + 1, end + 1, memo)
            }
        } else {
            pattern[j] == value[i] && visit(value, pattern, i + 1, j + 1, memo)
        };
        memo[i][j] = Some(result);
        result
    }

    visit(&value, &pattern, 0, 0, &mut memo)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntRange {
    start: i64,
    end: i64,
}

impl IntRange {
    fn contains(self, value: i64) -> bool {
        self.start <= value && value <= self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String, usize),
    Escaped(String, usize),
    Number(i64, usize),
    Float(FloatValue, usize),
    String(String, usize),
    Operator(String, usize),
    LParen(usize),
    RParen(usize),
    Colon(usize),
    Dash(usize),
}

impl Token {
    fn position(&self) -> usize {
        match self {
            Self::Ident(_, position)
            | Self::Escaped(_, position)
            | Self::Number(_, position)
            | Self::Float(_, position)
            | Self::String(_, position)
            | Self::Operator(_, position)
            | Self::LParen(position)
            | Self::RParen(position)
            | Self::Colon(position)
            | Self::Dash(position) => *position,
        }
    }

    fn display(&self) -> String {
        match self {
            Self::Ident(value, _) => value.clone(),
            Self::Escaped(value, _) => format!("\\{value}"),
            Self::Number(value, _) => value.to_string(),
            Self::Float(value, _) => value.0.to_string(),
            Self::String(value, _) => format!("\"{value}\""),
            Self::Operator(value, _) => value.clone(),
            Self::LParen(_) => "(".to_owned(),
            Self::RParen(_) => ")".to_owned(),
            Self::Colon(_) => ":".to_owned(),
            Self::Dash(_) => "-".to_owned(),
        }
    }
}

struct Lexer<'a> {
    chars: std::str::CharIndices<'a>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.char_indices(),
        }
    }

    fn lex(mut self) -> Result<Vec<Token>, SelectionError> {
        let mut tokens = Vec::new();
        while let Some((position, character)) = self.chars.next() {
            if character.is_whitespace() {
                continue;
            }
            let token = match character {
                '(' => Token::LParen(position),
                ')' => Token::RParen(position),
                ':' => Token::Colon(position),
                '-' => Token::Dash(position),
                '<' | '>' | '=' | '!' => self.lex_operator(position, character),
                '\\' => self.lex_escaped_identifier(position)?,
                '\'' | '"' => self.lex_string(position, character)?,
                character if character.is_ascii_digit() => self.lex_number(position, character)?,
                '.' if self
                    .chars
                    .clone()
                    .next()
                    .is_some_and(|(_, next)| next.is_ascii_digit()) =>
                {
                    self.lex_number(position, character)?
                }
                character if is_identifier_start(character) => {
                    self.lex_identifier(position, character)
                }
                character => {
                    return Err(SelectionError::UnexpectedCharacter {
                        position,
                        character,
                    });
                }
            };
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn lex_number(&mut self, position: usize, first: char) -> Result<Token, SelectionError> {
        let mut value = String::from(first);
        let mut is_float = first == '.';
        while let Some((_, character)) = self.chars.clone().next() {
            if character.is_ascii_digit() {
                value.push(character);
                self.chars.next();
            } else if character == '.' || character == 'e' || character == 'E' {
                is_float = true;
                value.push(character);
                self.chars.next();
                if matches!(character, 'e' | 'E')
                    && let Some((_, sign @ ('+' | '-'))) = self.chars.clone().next()
                {
                    value.push(sign);
                    self.chars.next();
                }
            } else {
                break;
            }
        }
        if is_float {
            let number = value
                .parse::<f64>()
                .map_err(|_| SelectionError::InvalidValue {
                    predicate: "number".to_owned(),
                    value,
                })?;
            Ok(Token::Float(FloatValue(number), position))
        } else {
            let number = value
                .parse::<i64>()
                .map_err(|_| SelectionError::InvalidValue {
                    predicate: "number".to_owned(),
                    value,
                })?;
            Ok(Token::Number(number, position))
        }
    }

    fn lex_operator(&mut self, position: usize, first: char) -> Token {
        let mut value = String::from(first);
        if let Some((_, '=')) = self.chars.clone().next()
            && (first == '<' || first == '>' || first == '!' || first == '=')
        {
            value.push('=');
            self.chars.next();
        }
        Token::Operator(value, position)
    }

    fn lex_identifier(&mut self, position: usize, first: char) -> Token {
        let mut value = String::from(first);
        while let Some((_, character)) = self.chars.clone().next() {
            if !is_identifier_continue(character) {
                break;
            }
            value.push(character);
            self.chars.next();
        }
        Token::Ident(value, position)
    }

    fn lex_escaped_identifier(&mut self, position: usize) -> Result<Token, SelectionError> {
        let Some((_, first)) = self.chars.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "escaped value",
            });
        };
        Ok(match self.lex_identifier(position, first) {
            Token::Ident(value, _) => Token::Escaped(value, position),
            token => token,
        })
    }

    fn lex_string(&mut self, position: usize, quote: char) -> Result<Token, SelectionError> {
        let mut value = String::new();
        while let Some((_, character)) = self.chars.next() {
            if character == quote {
                return Ok(Token::String(value, position));
            }
            if character == '\\' {
                let Some((_, escaped)) = self.chars.next() else {
                    return Err(SelectionError::UnexpectedEnd {
                        context: "quoted value",
                    });
                };
                value.push(escaped);
            } else {
                value.push(character);
            }
        }
        Err(SelectionError::UnexpectedEnd {
            context: "quoted value",
        })
    }
}

fn is_identifier_start(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(character, '_' | '*' | '?' | '.' | '+' | '/' | '[')
}

fn is_identifier_continue(character: char) -> bool {
    is_identifier_start(character) || matches!(character, '#' | '@' | '%' | ']' | '!' | '-')
}

fn is_selection_keyword(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "all"
            | "none"
            | "and"
            | "or"
            | "not"
            | "protein"
            | "backbone"
            | "water"
            | "nucleic"
            | "nucleicbackbone"
            | "nucleicbase"
            | "nucleicsugar"
            | "global"
            | "byres"
            | "bonded"
            | "group"
            | "name"
            | "resname"
            | "resid"
            | "resnum"
            | "index"
            | "bynum"
            | "element"
            | "chainid"
            | "segid"
            | "type"
            | "prop"
            | "around"
            | "point"
            | "same"
            | "atom"
            | "as"
    )
}

fn parse_comparison(value: &str) -> Option<Comparison> {
    match value {
        "<" => Some(Comparison::Less),
        "<=" => Some(Comparison::LessEqual),
        "==" => Some(Comparison::Equal),
        "!=" => Some(Comparison::NotEqual),
        ">" => Some(Comparison::Greater),
        ">=" => Some(Comparison::GreaterEqual),
        _ => None,
    }
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, cursor: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.cursor).cloned();
        self.cursor += usize::from(token.is_some());
        token
    }

    fn parse_expression(&mut self) -> Result<Expr, SelectionError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, SelectionError> {
        let mut expression = self.parse_and()?;
        while self.consume_keyword("or") {
            expression = Expr::Or(Box::new(expression), Box::new(self.parse_and()?));
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> Result<Expr, SelectionError> {
        let mut expression = self.parse_unary()?;
        while self.consume_keyword("and") {
            expression = Expr::And(Box::new(expression), Box::new(self.parse_unary()?));
        }
        Ok(expression)
    }

    fn parse_unary(&mut self) -> Result<Expr, SelectionError> {
        if self.consume_keyword("not") {
            return Ok(Expr::Not(Box::new(self.parse_unary()?)));
        }
        if matches!(self.peek(), Some(Token::LParen(_))) {
            let _ = self.next();
            let expression = self.parse_expression()?;
            match self.next() {
                Some(Token::RParen(_)) => return Ok(expression),
                Some(token) => {
                    return Err(SelectionError::UnexpectedToken {
                        position: token.position(),
                        token: token.display(),
                    });
                }
                None => {
                    return Err(SelectionError::UnexpectedEnd {
                        context: "closing parenthesis",
                    });
                }
            }
        }
        self.parse_predicate()
    }

    fn parse_predicate(&mut self) -> Result<Expr, SelectionError> {
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "predicate",
            });
        };
        let predicate = match token {
            Token::Ident(value, _) => value,
            token => {
                return Err(SelectionError::UnexpectedToken {
                    position: token.position(),
                    token: token.display(),
                });
            }
        };
        match predicate.to_ascii_lowercase().as_str() {
            "all" => Ok(Expr::All),
            "none" => Ok(Expr::None),
            "protein" => Ok(Expr::Protein),
            "backbone" => Ok(Expr::Backbone),
            "water" => Ok(Expr::Water),
            "nucleic" => Ok(Expr::Nucleic),
            "nucleicbackbone" => Ok(Expr::NucleicBackbone),
            "nucleicbase" => Ok(Expr::NucleicBase),
            "nucleicsugar" => Ok(Expr::NucleicSugar),
            "global" => Ok(Expr::Global(Box::new(self.parse_unary()?))),
            "byres" => Ok(Expr::ByRes(Box::new(self.parse_unary()?))),
            "bonded" => Ok(Expr::Bonded(Box::new(self.parse_unary()?))),
            "group" => Ok(Expr::Group(self.parse_one_string("group")?)),
            "name" => Ok(Expr::Predicate(Predicate::Name(
                self.parse_string_values("name")?,
            ))),
            "resname" => Ok(Expr::Predicate(Predicate::Resname(
                self.parse_string_values("resname")?,
            ))),
            "element" => Ok(Expr::Predicate(Predicate::Element(
                self.parse_string_values("element")?,
            ))),
            "chainid" => Ok(Expr::Predicate(Predicate::ChainId(
                self.parse_string_values("chainID")?,
            ))),
            "segid" => Ok(Expr::Predicate(Predicate::Segid(
                self.parse_string_values("segid")?,
            ))),
            "type" => Ok(Expr::Predicate(Predicate::Type(
                self.parse_string_values("type")?,
            ))),
            "resid" | "resnum" => Ok(Expr::Predicate(Predicate::Resid(
                self.parse_ranges("resid")?,
            ))),
            "index" => Ok(Expr::Predicate(Predicate::Index(
                self.parse_ranges("index")?,
            ))),
            "bynum" => Ok(Expr::Predicate(Predicate::ByNum(
                self.parse_ranges("bynum")?,
            ))),
            "prop" => self.parse_prop(),
            "around" => {
                let cutoff = self.parse_float("around")?;
                let selection = self.parse_unary()?;
                Ok(Expr::Around {
                    cutoff: FloatValue(cutoff),
                    selection: Box::new(selection),
                })
            }
            "point" => {
                let x = self.parse_float("point")?;
                let y = self.parse_float("point")?;
                let z = self.parse_float("point")?;
                let cutoff = self.parse_float("point")?;
                Ok(Expr::Point {
                    point: [FloatValue(x), FloatValue(y), FloatValue(z)],
                    cutoff: FloatValue(cutoff),
                })
            }
            "same" => self.parse_same(),
            "atom" => {
                let segid = self.parse_one_string("atom")?;
                let resid = self.parse_signed_integer("atom")?;
                let name = self.parse_one_string("atom")?;
                Ok(Expr::Atom { segid, resid, name })
            }
            _ => Err(SelectionError::UnknownPredicate(predicate)),
        }
    }

    fn parse_string_values(&mut self, predicate: &str) -> Result<Vec<String>, SelectionError> {
        let mut values = Vec::new();
        while let Some(token) = self.peek() {
            let value = match token {
                Token::Ident(value, _) if !is_selection_keyword(value) => value.clone(),
                Token::Escaped(value, _) => value.clone(),
                Token::String(value, _) => value.clone(),
                Token::Number(number, _) => number.to_string(),
                _ => break,
            };
            let _ = self.next();
            values.push(value);
        }
        if values.is_empty() {
            return Err(SelectionError::UnexpectedEnd {
                context: "predicate value",
            });
        }
        let _ = predicate;
        Ok(values)
    }

    fn parse_one_string(&mut self, predicate: &str) -> Result<String, SelectionError> {
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "predicate value",
            });
        };
        match token {
            Token::Ident(value, _) | Token::Escaped(value, _) | Token::String(value, _) => {
                Ok(value)
            }
            Token::Number(number, _) => Ok(number.to_string()),
            token => Err(SelectionError::InvalidValue {
                predicate: predicate.to_owned(),
                value: token.display(),
            }),
        }
    }

    fn parse_ranges(&mut self, predicate: &str) -> Result<Vec<IntRange>, SelectionError> {
        let mut ranges = Vec::new();
        loop {
            let can_start = matches!(self.peek(), Some(Token::Number(_, _) | Token::Dash(_)));
            if !can_start {
                break;
            }
            let start = self.parse_signed_integer(predicate)?;
            let end = if self.consume_range_separator() {
                self.parse_signed_integer(predicate)?
            } else {
                start
            };
            if start > end {
                return Err(SelectionError::InvalidRange { start, end });
            }
            if matches!(predicate, "index" | "bynum") && start < 0 {
                return Err(SelectionError::InvalidValue {
                    predicate: predicate.to_owned(),
                    value: start.to_string(),
                });
            }
            ranges.push(IntRange { start, end });
        }
        if ranges.is_empty() {
            return Err(SelectionError::UnexpectedEnd {
                context: "numeric value",
            });
        }
        Ok(ranges)
    }

    fn parse_float(&mut self, predicate: &str) -> Result<f64, SelectionError> {
        let negative = matches!(self.peek(), Some(Token::Dash(_)));
        if negative {
            let _ = self.next();
        }
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "numeric value",
            });
        };
        let mut value = match token {
            Token::Number(number, _) => number as f64,
            Token::Float(number, _) => number.0,
            token => {
                return Err(SelectionError::InvalidValue {
                    predicate: predicate.to_owned(),
                    value: token.display(),
                });
            }
        };
        if negative {
            value = -value;
        }
        Ok(value)
    }

    fn parse_prop(&mut self) -> Result<Expr, SelectionError> {
        let mut absolute = false;
        if self.consume_keyword("abs") {
            absolute = true;
        }
        let axis_name = self.parse_one_string("prop")?;
        let axis = match axis_name.to_ascii_lowercase().as_str() {
            "x" => Axis::X,
            "y" => Axis::Y,
            "z" => Axis::Z,
            _ => {
                return Err(SelectionError::InvalidValue {
                    predicate: "prop".to_owned(),
                    value: axis_name,
                });
            }
        };
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "operator",
            });
        };
        let operator = match token {
            Token::Operator(value, _) => {
                parse_comparison(&value).ok_or_else(|| SelectionError::InvalidValue {
                    predicate: "prop".to_owned(),
                    value,
                })?
            }
            token => {
                return Err(SelectionError::InvalidValue {
                    predicate: "prop".to_owned(),
                    value: token.display(),
                });
            }
        };
        let value = self.parse_float("prop")?;
        Ok(Expr::Predicate(Predicate::Prop {
            axis,
            operator,
            value: FloatValue(value),
            absolute,
        }))
    }

    fn parse_same(&mut self) -> Result<Expr, SelectionError> {
        let property = self.parse_one_string("same")?;
        let property = match property.to_ascii_lowercase().as_str() {
            "x" => SameProperty::X,
            "y" => SameProperty::Y,
            "z" => SameProperty::Z,
            "resid" | "resnum" => SameProperty::Resid,
            "residue" => SameProperty::Residue,
            "resname" => SameProperty::Resname,
            "name" => SameProperty::Name,
            "element" => SameProperty::Element,
            "type" => SameProperty::Type,
            "segid" | "segment" => SameProperty::Segid,
            "mass" => SameProperty::Mass,
            "charge" => SameProperty::Charge,
            _ => {
                return Err(SelectionError::InvalidValue {
                    predicate: "same".to_owned(),
                    value: property,
                });
            }
        };
        if !self.consume_keyword("as") {
            return Err(SelectionError::UnexpectedToken {
                position: self.peek().map_or(0, Token::position),
                token: self.peek().map_or_else(|| "".to_owned(), Token::display),
            });
        }
        let selection = self.parse_or()?;
        Ok(Expr::Same {
            property,
            selection: Box::new(selection),
        })
    }

    fn parse_signed_integer(&mut self, predicate: &str) -> Result<i64, SelectionError> {
        let negative = matches!(self.peek(), Some(Token::Dash(_)));
        if negative {
            let _ = self.next();
        }
        let Some(token) = self.next() else {
            return Err(SelectionError::UnexpectedEnd {
                context: "numeric value",
            });
        };
        let Token::Number(number, _) = token else {
            return Err(SelectionError::InvalidValue {
                predicate: predicate.to_owned(),
                value: token.display(),
            });
        };
        if negative {
            number
                .checked_neg()
                .ok_or_else(|| SelectionError::InvalidValue {
                    predicate: predicate.to_owned(),
                    value: format!("-{number}"),
                })
        } else {
            Ok(number)
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        let Some(Token::Ident(value, _)) = self.peek() else {
            return false;
        };
        if value.eq_ignore_ascii_case(keyword) {
            let _ = self.next();
            true
        } else {
            false
        }
    }

    fn consume_range_separator(&mut self) -> bool {
        if matches!(self.peek(), Some(Token::Colon(_) | Token::Dash(_))) {
            let _ = self.next();
            return true;
        }
        self.consume_keyword("to")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AtomLike, Selection, SelectionError, select, select_with_bonds, select_with_groups,
    };

    #[derive(Debug)]
    struct TestAtom {
        name: &'static str,
        resname: &'static str,
        resid: i32,
        index: usize,
        element: Option<&'static str>,
        atom_type: Option<&'static str>,
        chain_id: &'static str,
        segid: &'static str,
        position: [f64; 3],
    }

    impl AtomLike for TestAtom {
        fn name(&self) -> &str {
            self.name
        }
        fn resname(&self) -> &str {
            self.resname
        }
        fn resid(&self) -> i32 {
            self.resid
        }
        fn index(&self) -> usize {
            self.index
        }
        fn element(&self) -> Option<&str> {
            self.element
        }
        fn atom_type(&self) -> Option<&str> {
            self.atom_type
        }
        fn chain_id(&self) -> &str {
            self.chain_id
        }
        fn segid(&self) -> &str {
            self.segid
        }
        fn position(&self) -> [f64; 3] {
            self.position
        }
    }

    fn atoms() -> Vec<TestAtom> {
        vec![
            TestAtom {
                name: "CA",
                resname: "ALA",
                resid: 1,
                index: 0,
                element: Some("C"),
                atom_type: Some("CT1"),
                chain_id: "A",
                segid: "PROT",
                position: [0.0, 0.0, 0.0],
            },
            TestAtom {
                name: "N",
                resname: "ALA",
                resid: 1,
                index: 1,
                element: Some("N"),
                atom_type: Some("NH1"),
                chain_id: "A",
                segid: "PROT",
                position: [1.0, 0.0, 0.0],
            },
            TestAtom {
                name: "OW",
                resname: "HOH",
                resid: 8,
                index: 2,
                element: Some("O"),
                atom_type: Some("OT"),
                chain_id: "B",
                segid: "WAT",
                position: [0.0, 0.0, 3.0],
            },
        ]
    }

    #[test]
    fn predicates_and_ranges_match_atoms() {
        let atoms = atoms();
        for (expression, expected) in [
            ("all", 3),
            ("none", 0),
            ("name CA", 1),
            ("resname ALA", 2),
            ("resid 1-1", 2),
            ("index 1:2", 2),
            ("element O", 1),
            ("chainID B", 1),
            ("segid PROT", 2),
        ] {
            let selection = Selection::parse(expression).unwrap();
            assert_eq!(selection.apply(&atoms).len(), expected, "{expression}");
        }
        assert_eq!(select(&atoms, "name CA").unwrap().len(), 1);
    }

    #[test]
    fn wildcard_names_are_supported() {
        let atoms = [
            TestAtom {
                name: "CA",
                resname: "ALA",
                resid: 1,
                index: 0,
                element: Some("C"),
                atom_type: Some("CT1"),
                chain_id: "A",
                segid: "PROT",
                position: [0.0, 0.0, 0.0],
            },
            TestAtom {
                name: "CB",
                resname: "ALA",
                resid: 1,
                index: 1,
                element: Some("C"),
                atom_type: Some("CT2"),
                chain_id: "A",
                segid: "PROT",
                position: [1.0, 0.0, 0.0],
            },
        ];
        assert_eq!(select(&atoms, "name C?").unwrap().len(), 2);
        assert_eq!(select(&atoms, "name C*").unwrap().len(), 2);
        assert_eq!(select(&atoms, "name C[AB]").unwrap().len(), 2);
        assert_eq!(select(&atoms, "name C[!A]").unwrap().len(), 1);
    }

    #[test]
    fn extended_builtin_and_value_selectors_work() {
        let atoms = atoms();
        assert_eq!(select(&atoms, "protein").unwrap().len(), 2);
        assert_eq!(select(&atoms, "backbone").unwrap().len(), 2);
        assert_eq!(select(&atoms, "water").unwrap().len(), 1);
        assert_eq!(select(&atoms, "nucleic").unwrap().len(), 0);
        assert_eq!(select(&atoms, "type CT1").unwrap().len(), 1);
        assert_eq!(select(&atoms, "name CA N").unwrap().len(), 2);
        assert_eq!(select(&atoms, "resid 1 8").unwrap().len(), 3);
        assert_eq!(select(&atoms, "bynum 1").unwrap().len(), 1);
        assert_eq!(select(&atoms, "bynum 1:2").unwrap().len(), 2);
        assert_eq!(select(&atoms, "resname \\protein").unwrap().len(), 0);
        assert_eq!(select(&atoms, "atom PROT 1 CA").unwrap().len(), 1);
    }

    #[test]
    fn spatial_property_and_same_selectors_work() {
        let atoms = atoms();
        assert_eq!(select(&atoms, "prop x <= 0.5").unwrap().len(), 2);
        assert_eq!(select(&atoms, "prop x == 1").unwrap().len(), 1);
        assert_eq!(select(&atoms, "prop abs z < 2").unwrap().len(), 2);
        assert_eq!(select(&atoms, "point 0 0 0 1.01").unwrap().len(), 2);
        assert_eq!(select(&atoms, "point .5 0 0 .6").unwrap().len(), 2);
        assert_eq!(select(&atoms, "around 1.01 name CA").unwrap().len(), 1);
        assert_eq!(select(&atoms, "same resname as resid 1").unwrap().len(), 2);
        assert_eq!(select(&atoms, "same element as index 0").unwrap().len(), 1);
        assert_eq!(select(&atoms, "same x as index 0").unwrap().len(), 2);
        assert_eq!(select(&atoms, "global backbone").unwrap().len(), 2);
        assert_eq!(select(&atoms, "byres name CA").unwrap().len(), 2);
    }

    #[test]
    fn bonded_selectors_use_the_supplied_topology() {
        let atoms = atoms();
        let bonds = [(0, 1), (1, 2)];
        assert_eq!(
            select_with_bonds(&atoms, "bonded name N", &bonds)
                .unwrap()
                .iter()
                .map(|atom| atom.index())
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            select_with_bonds(&atoms, "type CT1 and bonded name N", &bonds)
                .unwrap()
                .iter()
                .map(|atom| atom.index())
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert!(select(&atoms, "bonded name N").unwrap().is_empty());
    }

    #[test]
    fn named_groups_match_indices_and_report_missing_names() {
        let atoms = atoms();
        let group_indices = [0, 2];
        assert_eq!(
            select_with_groups(&atoms, "not group solvent", &[("solvent", &group_indices)])
                .unwrap()
                .iter()
                .map(|atom| atom.index())
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            select_with_groups(&atoms, "group solvent", &[("solvent", &group_indices)])
                .unwrap()
                .len(),
            2
        );
        assert!(matches!(
            select_with_groups(&atoms, "group missing", &[]),
            Err(SelectionError::UnknownGroup(name)) if name == "missing"
        ));
    }

    #[test]
    fn boolean_precedence_and_parentheses_work() {
        let atoms = atoms();
        let selection = Selection::parse("name CA or name N and resname HOH").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);

        let selection = Selection::parse("(name CA or name N) and not chainID B").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 2);
    }

    #[test]
    fn quoted_values_and_negative_resids_work() {
        let mut atoms = atoms();
        atoms[0].resid = -2;
        let selection = Selection::parse("resid -2--1").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);

        let selection = Selection::parse("name 'CA'").unwrap();
        assert_eq!(selection.apply(&atoms).len(), 1);
    }

    #[test]
    fn invalid_expressions_are_reported() {
        assert_eq!(Selection::parse(""), Err(SelectionError::EmptyExpression));
        assert!(matches!(
            Selection::parse("wat CA"),
            Err(SelectionError::UnknownPredicate(_))
        ));
        assert!(matches!(
            Selection::parse("index -1"),
            Err(SelectionError::InvalidValue { .. })
        ));
        assert!(matches!(
            Selection::parse("resid 5-1"),
            Err(SelectionError::InvalidRange { .. })
        ));
        assert!(Selection::parse("(name CA").is_err());
    }
}
