//! Topology attribute guessing utilities.
//!
//! The routines in this module intentionally operate on plain strings and
//! coordinate slices.  This makes them useful while constructing a topology,
//! before [`crate::core::Atom`] values have been assembled.  Element symbols
//! are returned in upper case (for example, `"Cl"` is returned as `"CL"`).

use std::fmt;

/// Atomic masses (in unified atomic mass units) for the naturally occurring
/// elements and the standard representative masses for synthetic elements.
///
/// The table is public so callers that need to display or serialize the
/// reference data do not need to duplicate it.  Symbols are upper case to
/// match the rest of the crate's topology API.
pub const ELEMENT_MASSES: &[(&str, f64)] = &[
    ("H", 1.008),
    ("HE", 4.002_602),
    ("LI", 6.94),
    ("BE", 9.012_183_1),
    ("B", 10.81),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998_403_163),
    ("NE", 20.1797),
    ("NA", 22.989_769_28),
    ("MG", 24.305),
    ("AL", 26.981_538_5),
    ("SI", 28.085),
    ("P", 30.973_761_998),
    ("S", 32.06),
    ("CL", 35.45),
    ("AR", 39.948),
    ("K", 39.0983),
    ("CA", 40.078),
    ("SC", 44.955_908),
    ("TI", 47.867),
    ("V", 50.9415),
    ("CR", 51.9961),
    ("MN", 54.938_044),
    ("FE", 55.845),
    ("CO", 58.933_194),
    ("NI", 58.6934),
    ("CU", 63.546),
    ("ZN", 65.38),
    ("GA", 69.723),
    ("GE", 72.63),
    ("AS", 74.921_595),
    ("SE", 78.971),
    ("BR", 79.904),
    ("KR", 83.798),
    ("RB", 85.4678),
    ("SR", 87.62),
    ("Y", 88.905_84),
    ("ZR", 91.224),
    ("NB", 92.906_37),
    ("MO", 95.95),
    ("TC", 98.0),
    ("RU", 101.07),
    ("RH", 102.905_50),
    ("PD", 106.42),
    ("AG", 107.8682),
    ("CD", 112.414),
    ("IN", 114.818),
    ("SN", 118.710),
    ("SB", 121.760),
    ("TE", 127.60),
    ("I", 126.904_47),
    ("XE", 131.293),
    ("CS", 132.905_451_96),
    ("BA", 137.327),
    ("LA", 138.905_47),
    ("CE", 140.116),
    ("PR", 140.907_66),
    ("ND", 144.242),
    ("PM", 145.0),
    ("SM", 150.36),
    ("EU", 151.964),
    ("GD", 157.25),
    ("TB", 158.925_35),
    ("DY", 162.500),
    ("HO", 164.930_33),
    ("ER", 167.259),
    ("TM", 168.934_22),
    ("YB", 173.045),
    ("LU", 174.9668),
    ("HF", 178.49),
    ("TA", 180.947_88),
    ("W", 183.84),
    ("RE", 186.207),
    ("OS", 190.23),
    ("IR", 192.217),
    ("PT", 195.084),
    ("AU", 196.966_569),
    ("HG", 200.592),
    ("TL", 204.38),
    ("PB", 207.2),
    ("BI", 208.980_40),
    ("PO", 209.0),
    ("AT", 210.0),
    ("RN", 222.0),
    ("FR", 223.0),
    ("RA", 226.0),
    ("AC", 227.0),
    ("TH", 232.0377),
    ("PA", 231.035_88),
    ("U", 238.028_91),
    ("NP", 237.0),
    ("PU", 244.0),
    ("AM", 243.0),
    ("CM", 247.0),
    ("BK", 247.0),
    ("CF", 251.0),
    ("ES", 252.0),
    ("FM", 257.0),
    ("MD", 258.0),
    ("NO", 259.0),
    ("LR", 266.0),
    ("RF", 267.0),
    ("DB", 268.0),
    ("SG", 269.0),
    ("BH", 270.0),
    ("HS", 269.0),
    ("MT", 278.0),
    ("DS", 281.0),
    ("RG", 282.0),
    ("CN", 285.0),
    ("NH", 286.0),
    ("FL", 289.0),
    ("MC", 290.0),
    ("LV", 293.0),
    ("TS", 294.0),
    ("OG", 294.0),
    // A dummy atom type used by a number of force fields.
    ("DUMMY", 0.0),
];

/// Errors returned by topology guessing operations.
#[derive(Clone, Debug, PartialEq)]
pub enum GuesserError {
    /// The number of element labels does not match the number of coordinates.
    LengthMismatch { coordinates: usize, elements: usize },
    /// The cutoff must be finite and non-negative.
    InvalidCutoff(f64),
    /// An element or atom type is not present in the reference table.
    UnknownElement(String),
    /// A coordinate contains a non-finite component.
    InvalidCoordinate { index: usize, coordinate: [f64; 3] },
}

impl fmt::Display for GuesserError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                coordinates,
                elements,
            } => write!(
                formatter,
                "guessing bonds requires one element per coordinate (got {coordinates} coordinates and {elements} elements)"
            ),
            Self::InvalidCutoff(cutoff) => {
                write!(
                    formatter,
                    "bond cutoff must be finite and non-negative (got {cutoff})"
                )
            }
            Self::UnknownElement(element) => {
                write!(formatter, "unknown element or atom type {element:?}")
            }
            Self::InvalidCoordinate { index, coordinate } => write!(
                formatter,
                "coordinate {index} contains a non-finite component: {coordinate:?}"
            ),
        }
    }
}

impl std::error::Error for GuesserError {}

/// A stateless topology guesser with a configurable default bond cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guesser {
    cutoff: f64,
}

impl Default for Guesser {
    fn default() -> Self {
        Self { cutoff: 1.7 }
    }
}

impl Guesser {
    /// Construct a guesser using a 1.7 distance-unit default cutoff.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a guesser with an explicit default bond cutoff.
    pub fn with_cutoff(cutoff: f64) -> Result<Self, GuesserError> {
        validate_cutoff(cutoff)?;
        Ok(Self { cutoff })
    }

    /// Return this guesser's default bond cutoff.
    #[must_use]
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Guess one element from an atom name, optional type, and residue name.
    ///
    /// An explicitly supplied atom type takes precedence.  Residue names are
    /// used only when no atom name is available, which avoids interpreting a
    /// protein residue such as `ALA` as aluminium.  The fallback follows the
    /// historical MDAnalysis behavior and returns the first alphabetic token
    /// when no complete element symbol can be identified.
    pub fn guess_element(
        &self,
        name: &str,
        atom_type: Option<&str>,
        resname: Option<&str>,
    ) -> Result<String, GuesserError> {
        Ok(guess_element_impl(name, atom_type, resname))
    }

    /// Guess the atomic mass of an element or force-field atom type.
    pub fn guess_mass(&self, element: &str) -> Result<f64, GuesserError> {
        guess_mass(element)
    }

    /// Guess all bonds using this guesser's default cutoff.
    pub fn guess_bonds<E: AsRef<str>>(
        &self,
        coords: &[[f64; 3]],
        elements: &[E],
    ) -> Result<Vec<(usize, usize)>, GuesserError> {
        guess_bonds(coords, elements, self.cutoff)
    }

    /// Guess elements for a collection of atom names.
    pub fn guess_elements<S: AsRef<str>>(&self, names: &[S]) -> Vec<String> {
        names
            .iter()
            .map(|name| guess_element_impl(name.as_ref(), None, None))
            .collect()
    }

    /// Guess masses for a collection of element/type labels.
    pub fn guess_masses<S: AsRef<str>>(&self, elements: &[S]) -> Result<Vec<f64>, GuesserError> {
        elements
            .iter()
            .map(|element| guess_mass(element.as_ref()))
            .collect()
    }
}

/// Guess an element using an atom name and optional topology context.
pub fn guess_element(
    name: &str,
    atom_type: Option<&str>,
    resname: Option<&str>,
) -> Result<String, GuesserError> {
    Guesser::new().guess_element(name, atom_type, resname)
}

/// Guess an element from an atom name using MDAnalysis-compatible heuristics.
///
/// This convenience function is infallible and is useful when translating
/// legacy code that used `guess_atom_element`.  Empty names return an empty
/// string and unknown names use the historical first-letter fallback.
#[must_use]
pub fn guess_atom_element(name: &str) -> String {
    guess_element_impl(name, None, None)
}

/// Alias for [`guess_atom_element`].
#[must_use]
pub fn guess_atom_type(name: &str) -> String {
    guess_atom_element(name)
}

/// Guess elements for a collection of atom names.
#[must_use]
pub fn guess_types<S: AsRef<str>>(names: &[S]) -> Vec<String> {
    names
        .iter()
        .map(|name| guess_atom_element(name.as_ref()))
        .collect()
}

/// Return the standard atomic mass for an element or atom type.
pub fn guess_mass(element: &str) -> Result<f64, GuesserError> {
    let key = element.trim().to_ascii_uppercase();
    ELEMENT_MASSES
        .iter()
        .find_map(|(symbol, mass)| (*symbol == key).then_some(*mass))
        .ok_or(GuesserError::UnknownElement(element.trim().to_owned()))
}

/// Return the standard masses for a collection of elements or atom types.
pub fn guess_masses<S: AsRef<str>>(elements: &[S]) -> Result<Vec<f64>, GuesserError> {
    elements
        .iter()
        .map(|element| guess_mass(element.as_ref()))
        .collect()
}

/// Return the standard atomic mass, or zero for an unknown atom type.
///
/// This mirrors the pre-3.0 MDAnalysis compatibility helper.  New code that
/// needs to distinguish unknown values should use [`guess_mass`].
#[must_use]
pub fn get_atom_mass(element: &str) -> f64 {
    guess_mass(element).unwrap_or(0.0)
}

/// Guess a mass by first inferring the element from an atom name.
#[must_use]
pub fn guess_atom_mass(name: &str) -> f64 {
    get_atom_mass(&guess_atom_element(name))
}

/// Guess bonds from Cartesian coordinates and element labels.
///
/// Every unique pair `(i, j)` with `i < j` whose Euclidean distance is at
/// most `cutoff` is returned.  Labels are validated against the standard
/// element table so malformed topology metadata is reported instead of being
/// silently ignored.  Coordinates and cutoff use the same distance units as
/// the caller (typically Angstroms).
pub fn guess_bonds<E: AsRef<str>>(
    coords: &[[f64; 3]],
    elements: &[E],
    cutoff: f64,
) -> Result<Vec<(usize, usize)>, GuesserError> {
    validate_cutoff(cutoff)?;
    if coords.len() != elements.len() {
        return Err(GuesserError::LengthMismatch {
            coordinates: coords.len(),
            elements: elements.len(),
        });
    }

    for (index, &coordinate) in coords.iter().enumerate() {
        if !coordinate.iter().all(|component| component.is_finite()) {
            return Err(GuesserError::InvalidCoordinate { index, coordinate });
        }
    }
    for element in elements {
        validate_element(element.as_ref())?;
    }

    let cutoff_squared = cutoff * cutoff;
    let mut bonds = Vec::new();
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared <= cutoff_squared {
                bonds.push((i, j));
            }
        }
    }
    Ok(bonds)
}

fn validate_cutoff(cutoff: f64) -> Result<(), GuesserError> {
    if cutoff.is_finite() && cutoff >= 0.0 {
        Ok(())
    } else {
        Err(GuesserError::InvalidCutoff(cutoff))
    }
}

fn validate_element(element: &str) -> Result<(), GuesserError> {
    let key = element.trim().to_ascii_uppercase();
    is_mass_element(&key)
        .then_some(())
        .ok_or_else(|| GuesserError::UnknownElement(element.trim().to_owned()))
}

fn guess_element_impl(name: &str, atom_type: Option<&str>, resname: Option<&str>) -> String {
    // Explicit force-field types contain useful information even when the
    // display name is generic (e.g. `CT` or `OW`).
    if let Some(atom_type) = atom_type.filter(|value| !value.trim().is_empty()) {
        let guessed = infer_from_token(atom_type);
        if is_guessable_element(&guessed) {
            return guessed;
        }
    }

    if !name.trim().is_empty() {
        let guessed = infer_from_token(name);
        if is_guessable_element(&guessed) || guessed.len() == 1 {
            return guessed;
        }
    }

    // Residue names are intentionally a last resort.  They are useful for
    // isolated ions (`resname = "NA"`) but should not override atom names in
    // ordinary residues (`resname = "ALA"`).
    if name.trim().is_empty()
        && let Some(resname) = resname.filter(|value| !value.trim().is_empty())
    {
        // `CA` is the conventional residue atom name for an alpha carbon,
        // but is also the calcium symbol.  In the absence of an atom name a
        // residue called CA is much more likely to denote a calcium ion.
        if resname.trim().eq_ignore_ascii_case("CA") {
            return "CA".to_owned();
        }
        let guessed = infer_from_token(resname);
        if is_guessable_element(&guessed) {
            return guessed;
        }
    }
    infer_from_token(name)
}

fn infer_from_token(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let upper = trimmed.to_ascii_uppercase();
    if let Some(element) = special_atom_type(&upper) {
        return element.to_owned();
    }

    // Strip charge and wildcard symbols, then use the first alphabetic part
    // before a numeric suffix.  This handles names such as `HB1`, `1HG2`, and
    // `C0U` in the same way as MDAnalysis' historical guesser.
    let no_symbols: String = upper
        .chars()
        .filter(|character| !matches!(character, '*' | '+' | '-'))
        .collect();
    let token = no_symbols
        .split(|character: char| character.is_ascii_digit())
        .find(|part| !part.is_empty())
        .unwrap_or(&no_symbols);
    if let Some(element) = special_atom_type(token) {
        return element.to_owned();
    }
    // In atom names `CA` conventionally means the alpha carbon in an amino
    // acid.  Calcium atom names are represented by explicit forms such as
    // `CAL` or `CA2+` in the force-field tables above.
    if token == "CA" {
        return "C".to_owned();
    }
    if is_guessable_element(token) {
        return token.to_owned();
    }

    // Prefer removing a trailing character over a leading character.  This
    // resolves force-field names such as `AO5` -> O and `HG2` -> H.
    let mut candidate = token;
    while !candidate.is_empty() {
        let without_last = candidate
            .get(..candidate.len().saturating_sub(1))
            .unwrap_or("");
        if is_guessable_element(without_last) {
            return without_last.to_owned();
        }
        let without_first = candidate.get(1..).unwrap_or("");
        if is_guessable_element(without_first) {
            return without_first.to_owned();
        }
        if candidate.chars().count() <= 2 {
            return candidate.chars().next().unwrap_or_default().to_string();
        }
        candidate = without_last;
    }
    no_symbols
}

fn is_mass_element(element: &str) -> bool {
    ELEMENT_MASSES.iter().any(|(symbol, _)| *symbol == element)
}

fn is_guessable_element(element: &str) -> bool {
    // This is the conservative set used by the original MDAnalysis atom-name
    // heuristic.  Several two-letter symbols (notably HE and HO) are common
    // force-field prefixes and are therefore deliberately resolved through
    // their leading H rather than treated as complete element symbols.
    matches!(
        element,
        "H" | "LI"
            | "BE"
            | "B"
            | "C"
            | "N"
            | "O"
            | "F"
            | "NA"
            | "MG"
            | "AL"
            | "P"
            | "SI"
            | "S"
            | "CL"
            | "K"
            | "CA"
            | "FE"
            | "ZN"
            | "CU"
            | "BR"
            | "I"
            | "CS"
            | "RB"
            | "CE"
            | "DUMMY"
    )
}

fn special_atom_type(token: &str) -> Option<&'static str> {
    Some(match token {
        // Halides and common metal ion names.
        "BR" => "BR",
        "CAL" | "C0" | "CA2+" => "CA",
        "CES" | "CS+" => "CS",
        "CLA" | "CLAL" | "CL" | "CL-" => "CL",
        "IOD" => "I",
        "FE2" => "FE",
        "LIT" | "LI+" | "QL" => "LI",
        "MG2+" => "MG",
        "POT" | "K+" | "QK" => "K",
        "SOD" | "NA+" | "QN" => "NA",
        "ZN" => "ZN",
        "CU2+" => "CU",
        "QC" => "CE",
        "QR" => "RB",
        // Amber special carbon and virtual-site names.
        "BC" | "AC" => "C",
        "MW" => "DUMMY",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "{left} != {right}");
    }

    #[test]
    fn common_atom_names_are_guessed() {
        assert_eq!(guess_atom_element("MG2+"), "MG");
        assert_eq!(guess_atom_element("HB1"), "H");
        assert_eq!(guess_atom_element("AO5*"), "O");
        assert_eq!(guess_atom_element("C0U"), "C");
        assert_eq!(guess_atom_element("CA"), "C");
        assert_eq!(guess_atom_element("HO"), "H");
        assert_eq!(guess_atom_element("he"), "H");
        assert_eq!(guess_atom_element("zn"), "ZN");
        assert_eq!(guess_atom_element("Ca2+"), "CA");
        assert_eq!(guess_atom_element(""), "");
    }

    #[test]
    fn explicit_context_takes_precedence() {
        assert_eq!(guess_element("X", Some("NA"), None).unwrap(), "NA");
        assert_eq!(guess_element("", None, Some("CL")).unwrap(), "CL");
        assert_eq!(guess_element("CA", None, Some("CAL")).unwrap(), "C");
    }

    #[test]
    fn masses_are_case_insensitive_and_complete_for_common_elements() {
        close(guess_mass("c").unwrap(), 12.011);
        close(guess_mass("CL").unwrap(), 35.45);
        close(guess_mass("Fe").unwrap(), 55.845);
        close(guess_mass("DUMMY").unwrap(), 0.0);
        assert!(matches!(
            guess_mass("not-an-element"),
            Err(GuesserError::UnknownElement(_))
        ));
    }

    #[test]
    fn bonds_are_unique_pairs_with_inclusive_cutoff() {
        let coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.0, 0.0]];
        let elements = ["C", "H", "O"];
        assert_eq!(
            guess_bonds(&coordinates, &elements, 1.05).unwrap(),
            vec![(0, 1)]
        );
        assert_eq!(
            Guesser::with_cutoff(1.05)
                .unwrap()
                .guess_bonds(&coordinates, &elements)
                .unwrap(),
            vec![(0, 1)]
        );
    }

    #[test]
    fn malformed_bond_inputs_are_rejected() {
        let coordinates = [[0.0, 0.0, 0.0]];
        assert!(matches!(
            guess_bonds(&coordinates, &[] as &[&str], 1.0),
            Err(GuesserError::LengthMismatch { .. })
        ));
        assert!(matches!(
            guess_bonds(&coordinates, &["C"], f64::NAN),
            Err(GuesserError::InvalidCutoff(_))
        ));
        assert!(matches!(
            guess_bonds(&coordinates, &["X"], 1.0),
            Err(GuesserError::UnknownElement(_))
        ));
    }
}
