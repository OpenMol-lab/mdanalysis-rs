//! Physical units and unit conversion helpers.
//!
//! MDAnalysis uses Angstroms (A), picoseconds (ps), kilojoules per mole
//! (kJ/mol), and elementary charges (e) as its base units.  The enums in this
//! module make those conventions explicit while [`convert`] also accepts the
//! string spellings used by MDAnalysis input files.

use std::fmt;
use std::str::FromStr;

/// Avogadro constant (CODATA 2010 value used by MDAnalysis).
pub const N_AVOGADRO: f64 = 6.022_141_29e23;
/// Elementary charge in coulombs (CODATA 2010 value used by MDAnalysis).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_565e-19;
/// Thermochemical calorie in joules.
pub const CALORIE: f64 = 4.184;
/// Boltzmann constant in kJ/(mol K).
pub const BOLTZMANN_CONSTANT: f64 = 8.314_462_159e-3;
/// Electric constant in As/(Angstrom V), as used by MDAnalysis.
pub const ELECTRIC_CONSTANT: f64 = 5.526_350e-3;

/// Named physical constants.  This is an array rather than a hash map so the
/// table is available in `const` contexts and does not require dependencies.
pub const CONSTANTS: &[(&str, f64)] = &[
    ("N_Avogadro", N_AVOGADRO),
    ("elementary_charge", ELEMENTARY_CHARGE),
    ("calorie", CALORIE),
    ("Boltzmann_constant", BOLTZMANN_CONSTANT),
    ("electric_constant", ELECTRIC_CONSTANT),
];

/// Look up one of the constants in [`CONSTANTS`].
#[must_use]
pub fn constant(name: &str) -> Option<f64> {
    // Keep the misspelling accepted by MDAnalysis 2.x for source
    // compatibility. The Python implementation emits a deprecation warning;
    // Rust callers can migrate by using the correctly-spelled key.
    if name == "Boltzman_constant" {
        return Some(BOLTZMANN_CONSTANT);
    }
    CONSTANTS
        .iter()
        .find_map(|(key, value)| (*key == name).then_some(*value))
}

/// Base units used by MDAnalysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseUnits {
    pub length: &'static str,
    pub time: &'static str,
    pub energy: &'static str,
    pub charge: &'static str,
    pub force: &'static str,
    pub speed: &'static str,
}

/// The MDAnalysis base-unit table.
pub const MDANALYSIS_BASE_UNITS: BaseUnits = BaseUnits {
    length: "A",
    time: "ps",
    energy: "kJ/mol",
    charge: "e",
    force: "kJ/(mol*A)",
    speed: "A/ps",
};

/// Length units.  Factors are expressed in Angstroms per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LengthUnit {
    Angstrom,
    Nanometer,
    Picometer,
    Femtometer,
    Meter,
}

#[allow(non_upper_case_globals)]
impl LengthUnit {
    /// Short-name aliases useful when translating MDAnalysis code.
    pub const A: Self = Self::Angstrom;
    pub const Angstroms: Self = Self::Angstrom;
    pub const Nm: Self = Self::Nanometer;
    pub const NM: Self = Self::Nanometer;
    pub const Pm: Self = Self::Picometer;
    pub const PM: Self = Self::Picometer;
    pub const Fm: Self = Self::Femtometer;
    pub const FM: Self = Self::Femtometer;
    pub const M: Self = Self::Meter;

    #[must_use]
    pub const fn factor_to_angstrom(self) -> f64 {
        match self {
            Self::Angstrom => 1.0,
            Self::Nanometer => 10.0,
            Self::Picometer => 0.01,
            Self::Femtometer => 0.000_01,
            Self::Meter => 1.0e10,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Angstrom => "A",
            Self::Nanometer => "nm",
            Self::Picometer => "pm",
            Self::Femtometer => "fm",
            Self::Meter => "m",
        }
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Time units.  Factors are expressed in picoseconds per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Picosecond,
    Femtosecond,
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Akma,
}

#[allow(non_upper_case_globals)]
impl TimeUnit {
    pub const Ps: Self = Self::Picosecond;
    pub const PS: Self = Self::Picosecond;
    pub const Fs: Self = Self::Femtosecond;
    pub const FS: Self = Self::Femtosecond;
    pub const Ns: Self = Self::Nanosecond;
    pub const NS: Self = Self::Nanosecond;
    pub const Us: Self = Self::Microsecond;
    pub const US: Self = Self::Microsecond;
    pub const Ms: Self = Self::Millisecond;
    pub const MS: Self = Self::Millisecond;
    pub const S: Self = Self::Second;
    pub const AKMA: Self = Self::Akma;

    #[must_use]
    pub const fn factor_to_picosecond(self) -> f64 {
        match self {
            Self::Picosecond => 1.0,
            Self::Femtosecond => 1.0e-3,
            Self::Nanosecond => 1.0e3,
            Self::Microsecond => 1.0e6,
            Self::Millisecond => 1.0e9,
            Self::Second => 1.0e12,
            // 1 AKMA = 4.888821e-14 s = 0.04888821 ps.
            Self::Akma => 4.888_821e-2,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Picosecond => "ps",
            Self::Femtosecond => "fs",
            Self::Nanosecond => "ns",
            Self::Microsecond => "us",
            Self::Millisecond => "ms",
            Self::Second => "s",
            Self::Akma => "AKMA",
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Energy units.  Factors are expressed in kJ/mol per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnergyUnit {
    KilojoulePerMol,
    KilocaloriePerMol,
    Joule,
    ElectronVolt,
}

#[allow(non_upper_case_globals)]
impl EnergyUnit {
    pub const KJPerMol: Self = Self::KilojoulePerMol;
    pub const KJ_MOL: Self = Self::KilojoulePerMol;
    pub const KcalPerMol: Self = Self::KilocaloriePerMol;
    pub const KCAL_MOL: Self = Self::KilocaloriePerMol;
    pub const J: Self = Self::Joule;
    pub const EV: Self = Self::ElectronVolt;

    #[must_use]
    pub const fn factor_to_kj_per_mol(self) -> f64 {
        match self {
            Self::KilojoulePerMol => 1.0,
            Self::KilocaloriePerMol => CALORIE,
            // MDAnalysis' historical "J" unit denotes joules per molecule.
            Self::Joule => N_AVOGADRO / 1.0e3,
            Self::ElectronVolt => ELEMENTARY_CHARGE * N_AVOGADRO / 1.0e3,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::KilojoulePerMol => "kJ/mol",
            Self::KilocaloriePerMol => "kcal/mol",
            Self::Joule => "J",
            Self::ElectronVolt => "eV",
        }
    }
}

impl fmt::Display for EnergyUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Speed/velocity units.  Factors are expressed in Angstroms per picosecond.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VelocityUnit {
    AngstromPerPs,
    AngstromPerFs,
    AngstromPerNs,
    AngstromPerUs,
    AngstromPerMs,
    AngstromPerAkma,
    NanometerPerPs,
    NanometerPerNs,
    PicometerPerPs,
    MeterPerSecond,
}

/// Synonym used in MDAnalysis documentation.
pub type SpeedUnit = VelocityUnit;

#[allow(non_upper_case_globals)]
impl VelocityUnit {
    pub const APerPs: Self = Self::AngstromPerPs;
    pub const APerFs: Self = Self::AngstromPerFs;
    pub const APerNs: Self = Self::AngstromPerNs;
    pub const APerUs: Self = Self::AngstromPerUs;
    pub const APerMs: Self = Self::AngstromPerMs;
    pub const APerAkma: Self = Self::AngstromPerAkma;
    pub const NmPerPs: Self = Self::NanometerPerPs;
    pub const NmPerNs: Self = Self::NanometerPerNs;
    pub const PmPerPs: Self = Self::PicometerPerPs;
    pub const MPerS: Self = Self::MeterPerSecond;

    #[must_use]
    pub const fn factor_to_angstrom_per_ps(self) -> f64 {
        match self {
            Self::AngstromPerPs => 1.0,
            Self::AngstromPerFs => 1.0e3,
            Self::AngstromPerNs => 1.0e-3,
            Self::AngstromPerUs => 1.0e-6,
            Self::AngstromPerMs => 1.0e-9,
            Self::AngstromPerAkma => 1.0 / 4.888_821e-2,
            Self::NanometerPerPs => 10.0,
            Self::NanometerPerNs => 10.0e-3,
            Self::PicometerPerPs => 1.0e-2,
            Self::MeterPerSecond => 1.0e-2,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::AngstromPerPs => "A/ps",
            Self::AngstromPerFs => "A/fs",
            Self::AngstromPerNs => "A/ns",
            Self::AngstromPerUs => "A/us",
            Self::AngstromPerMs => "A/ms",
            Self::AngstromPerAkma => "A/AKMA",
            Self::NanometerPerPs => "nm/ps",
            Self::NanometerPerNs => "nm/ns",
            Self::PicometerPerPs => "pm/ps",
            Self::MeterPerSecond => "m/s",
        }
    }
}

impl fmt::Display for VelocityUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Force units.  Factors are expressed in kJ/(mol A) per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ForceUnit {
    KilojoulePerMolAngstrom,
    KilojoulePerMolNanometer,
    Newton,
    JoulePerMeter,
    KilocaloriePerMolAngstrom,
}

#[allow(non_upper_case_globals)]
impl ForceUnit {
    pub const KJPerMolAngstrom: Self = Self::KilojoulePerMolAngstrom;
    pub const KJPerMolNanometer: Self = Self::KilojoulePerMolNanometer;
    pub const N: Self = Self::Newton;
    pub const JPerM: Self = Self::JoulePerMeter;
    pub const KcalPerMolAngstrom: Self = Self::KilocaloriePerMolAngstrom;

    #[must_use]
    pub const fn factor_to_kj_per_mol_angstrom(self) -> f64 {
        match self {
            Self::KilojoulePerMolAngstrom => 1.0,
            Self::KilojoulePerMolNanometer => 0.1,
            Self::Newton | Self::JoulePerMeter => N_AVOGADRO * 1.0e-13,
            Self::KilocaloriePerMolAngstrom => CALORIE,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::KilojoulePerMolAngstrom => "kJ/(mol*A)",
            Self::KilojoulePerMolNanometer => "kJ/(mol*nm)",
            Self::Newton => "N",
            Self::JoulePerMeter => "J/m",
            Self::KilocaloriePerMolAngstrom => "kcal/(mol*A)",
        }
    }
}

impl fmt::Display for ForceUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Charge units.  Factors are expressed in elementary charges per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChargeUnit {
    ElementaryCharge,
    Amber,
    Coulomb,
}

#[allow(non_upper_case_globals)]
impl ChargeUnit {
    pub const E: Self = Self::ElementaryCharge;
    pub const C: Self = Self::Coulomb;

    #[must_use]
    pub const fn factor_to_elementary_charge(self) -> f64 {
        match self {
            Self::ElementaryCharge => 1.0,
            Self::Amber => 18.2223,
            Self::Coulomb => 1.0 / ELEMENTARY_CHARGE,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::ElementaryCharge => "e",
            Self::Amber => "Amber",
            Self::Coulomb => "C",
        }
    }
}

impl fmt::Display for ChargeUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Mass units.  The MDAnalysis base mass unit is the atomic mass unit (amu).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MassUnit {
    AtomicMassUnit,
    Kilogram,
    GramPerMol,
}

#[allow(non_upper_case_globals)]
impl MassUnit {
    pub const Amu: Self = Self::AtomicMassUnit;
    pub const AMU: Self = Self::AtomicMassUnit;
    pub const Da: Self = Self::AtomicMassUnit;
    pub const Dalton: Self = Self::AtomicMassUnit;
    pub const Kg: Self = Self::Kilogram;
    pub const GPerMol: Self = Self::GramPerMol;

    #[must_use]
    pub const fn factor_to_amu(self) -> f64 {
        match self {
            Self::AtomicMassUnit => 1.0,
            // 1 u = 1.66053906660e-27 kg (exact enough for f64 use here).
            Self::Kilogram => 1.0 / 1.660_539_066_60e-27,
            // Numerically, g/mol and u are equivalent.
            Self::GramPerMol => 1.0,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::AtomicMassUnit => "amu",
            Self::Kilogram => "kg",
            Self::GramPerMol => "g/mol",
        }
    }
}

impl fmt::Display for MassUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

/// Dimension of a unit, used to reject incompatible conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitKind {
    Length,
    Time,
    Energy,
    Velocity,
    Force,
    Charge,
    Mass,
}

#[allow(non_upper_case_globals)]
impl UnitKind {
    /// Historical spelling used by callers that call velocity "speed".
    pub const Speed: Self = Self::Velocity;
}

impl fmt::Display for UnitKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Length => "length",
            Self::Time => "time",
            Self::Energy => "energy",
            Self::Velocity => "velocity",
            Self::Force => "force",
            Self::Charge => "charge",
            Self::Mass => "mass",
        })
    }
}

/// A dynamically typed unit, useful for conversions where the dimensions are
/// known only at runtime (for example when reading a trajectory header).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    Length(LengthUnit),
    Time(TimeUnit),
    Energy(EnergyUnit),
    Velocity(VelocityUnit),
    Force(ForceUnit),
    Charge(ChargeUnit),
    Mass(MassUnit),
}

impl Unit {
    #[must_use]
    pub const fn kind(self) -> UnitKind {
        match self {
            Self::Length(_) => UnitKind::Length,
            Self::Time(_) => UnitKind::Time,
            Self::Energy(_) => UnitKind::Energy,
            Self::Velocity(_) => UnitKind::Velocity,
            Self::Force(_) => UnitKind::Force,
            Self::Charge(_) => UnitKind::Charge,
            Self::Mass(_) => UnitKind::Mass,
        }
    }

    #[must_use]
    pub const fn factor_to_base(self) -> f64 {
        match self {
            Self::Length(unit) => unit.factor_to_angstrom(),
            Self::Time(unit) => unit.factor_to_picosecond(),
            Self::Energy(unit) => unit.factor_to_kj_per_mol(),
            Self::Velocity(unit) => unit.factor_to_angstrom_per_ps(),
            Self::Force(unit) => unit.factor_to_kj_per_mol_angstrom(),
            Self::Charge(unit) => unit.factor_to_elementary_charge(),
            Self::Mass(unit) => unit.factor_to_amu(),
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(unit) => unit.fmt(formatter),
            Self::Time(unit) => unit.fmt(formatter),
            Self::Energy(unit) => unit.fmt(formatter),
            Self::Velocity(unit) => unit.fmt(formatter),
            Self::Force(unit) => unit.fmt(formatter),
            Self::Charge(unit) => unit.fmt(formatter),
            Self::Mass(unit) => unit.fmt(formatter),
        }
    }
}

/// Errors returned by parsing or converting units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitError {
    /// The supplied spelling is not in the unit tables.
    UnknownUnit(String),
    /// Both units are known but represent different physical dimensions.
    Incompatible { from: UnitKind, to: UnitKind },
}

impl fmt::Display for UnitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownUnit(unit) => write!(formatter, "unit '{unit}' is not recognized"),
            Self::Incompatible { from, to } => {
                write!(
                    formatter,
                    "cannot convert between unit types {from} -> {to}"
                )
            }
        }
    }
}

impl std::error::Error for UnitError {}

/// Convert an enum or string unit into the dynamic representation.
pub trait IntoUnit {
    /// Resolve this value to a known physical unit.
    fn into_unit(self) -> Result<Unit, UnitError>;
}

impl IntoUnit for Unit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(self)
    }
}

impl IntoUnit for LengthUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Length(self))
    }
}

impl IntoUnit for TimeUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Time(self))
    }
}

impl IntoUnit for EnergyUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Energy(self))
    }
}

impl IntoUnit for VelocityUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Velocity(self))
    }
}

impl IntoUnit for ForceUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Force(self))
    }
}

impl IntoUnit for ChargeUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Charge(self))
    }
}

impl IntoUnit for MassUnit {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Ok(Unit::Mass(self))
    }
}

impl From<LengthUnit> for Unit {
    fn from(unit: LengthUnit) -> Self {
        Self::Length(unit)
    }
}

impl From<TimeUnit> for Unit {
    fn from(unit: TimeUnit) -> Self {
        Self::Time(unit)
    }
}

impl From<EnergyUnit> for Unit {
    fn from(unit: EnergyUnit) -> Self {
        Self::Energy(unit)
    }
}

impl From<VelocityUnit> for Unit {
    fn from(unit: VelocityUnit) -> Self {
        Self::Velocity(unit)
    }
}

impl From<ForceUnit> for Unit {
    fn from(unit: ForceUnit) -> Self {
        Self::Force(unit)
    }
}

impl From<ChargeUnit> for Unit {
    fn from(unit: ChargeUnit) -> Self {
        Self::Charge(unit)
    }
}

impl From<MassUnit> for Unit {
    fn from(unit: MassUnit) -> Self {
        Self::Mass(unit)
    }
}

impl IntoUnit for &str {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Unit::from_str(self)
    }
}

impl IntoUnit for String {
    fn into_unit(self) -> Result<Unit, UnitError> {
        Unit::from_str(&self)
    }
}

impl TryFrom<&str> for Unit {
    type Error = UnitError;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::from_str(name)
    }
}

impl TryFrom<String> for Unit {
    type Error = UnitError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::from_str(&name)
    }
}

impl FromStr for Unit {
    type Err = UnitError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        let key = name.trim().to_ascii_lowercase();
        let unit = match key.as_str() {
            "a" | "angstrom" | "angstroms" | "å" | "Å" => Self::Length(LengthUnit::Angstrom),
            "nm" | "nanometer" | "nanometers" => Self::Length(LengthUnit::Nanometer),
            "pm" | "picometer" | "picometers" => Self::Length(LengthUnit::Picometer),
            "fm" | "femtometer" | "femtometers" => Self::Length(LengthUnit::Femtometer),
            "m" | "meter" | "meters" => Self::Length(LengthUnit::Meter),
            "ps" | "picosecond" | "picoseconds" => Self::Time(TimeUnit::Picosecond),
            "fs" | "femtosecond" | "femtoseconds" => Self::Time(TimeUnit::Femtosecond),
            "ns" | "nanosecond" | "nanoseconds" => Self::Time(TimeUnit::Nanosecond),
            "us" | "μs" | "µs" | "microsecond" | "microseconds" => {
                Self::Time(TimeUnit::Microsecond)
            }
            "ms" | "millisecond" | "milliseconds" => Self::Time(TimeUnit::Millisecond),
            "s" | "sec" | "second" | "seconds" => Self::Time(TimeUnit::Second),
            "akma" => Self::Time(TimeUnit::Akma),
            "kj/mol" | "kj mol-1" | "kilojoule/mol" | "kilojoules/mol" => {
                Self::Energy(EnergyUnit::KilojoulePerMol)
            }
            "kcal/mol" | "kcal mol-1" | "kilocalorie/mol" | "kilocalories/mol" => {
                Self::Energy(EnergyUnit::KilocaloriePerMol)
            }
            "j" | "j/mol" | "joule" | "joules" | "joule/mol" | "joules/mol" => {
                Self::Energy(EnergyUnit::Joule)
            }
            "ev" | "electronvolt" => Self::Energy(EnergyUnit::ElectronVolt),
            "a/ps" | "å/ps" | "Å/ps" | "angstrom/ps" | "angstrom/picosecond" | "a/picosecond" => {
                Self::Velocity(VelocityUnit::AngstromPerPs)
            }
            "a/fs" | "å/fs" | "Å/fs" | "angstrom/fs" | "angstrom/femtosecond" | "a/femtosecond" => {
                Self::Velocity(VelocityUnit::AngstromPerFs)
            }
            "a/ns" | "å/ns" | "Å/ns" | "angstrom/ns" | "angstrom/nanosecond" | "a/nanosecond" => {
                Self::Velocity(VelocityUnit::AngstromPerNs)
            }
            "a/us"
            | "a/μs"
            | "a/µs"
            | "å/us"
            | "å/μs"
            | "Å/μs"
            | "angstrom/us"
            | "angstrom/microsecond" => Self::Velocity(VelocityUnit::AngstromPerUs),
            "a/ms" | "å/ms" | "Å/ms" | "angstrom/ms" | "angstrom/millisecond" => {
                Self::Velocity(VelocityUnit::AngstromPerMs)
            }
            "a/akma" | "å/akma" | "Å/akma" | "angstrom/akma" => {
                Self::Velocity(VelocityUnit::AngstromPerAkma)
            }
            "nm/ps" | "nanometer/ps" | "nanometer/picosecond" => {
                Self::Velocity(VelocityUnit::NanometerPerPs)
            }
            "nm/ns" | "nanometer/ns" | "nanometer/nanosecond" => {
                Self::Velocity(VelocityUnit::NanometerPerNs)
            }
            "pm/ps" | "picometer/ps" | "picometer/picosecond" => {
                Self::Velocity(VelocityUnit::PicometerPerPs)
            }
            "m/s" | "meter/second" | "meters/second" => {
                Self::Velocity(VelocityUnit::MeterPerSecond)
            }
            "kj/(mol*a)" | "kj/(mol*å)" | "kj/(mol*Å)" | "kj/(mol*angstrom)" => {
                Self::Force(ForceUnit::KilojoulePerMolAngstrom)
            }
            "kj/(mol*nm)" | "kj/(mol*nanometer)" => {
                Self::Force(ForceUnit::KilojoulePerMolNanometer)
            }
            "n" | "newton" | "newtons" => Self::Force(ForceUnit::Newton),
            "j/m" | "joule/m" | "joule/meter" | "joules/meter" => {
                Self::Force(ForceUnit::JoulePerMeter)
            }
            "kcal/(mol*a)" | "kcal/(mol*å)" | "kcal/(mol*Å)" | "kcal/(mol*angstrom)" => {
                Self::Force(ForceUnit::KilocaloriePerMolAngstrom)
            }
            "e" | "electron charge" | "elementary_charge" => {
                Self::Charge(ChargeUnit::ElementaryCharge)
            }
            "amber" => Self::Charge(ChargeUnit::Amber),
            "c" | "coulomb" | "coulombs" | "as" => Self::Charge(ChargeUnit::Coulomb),
            "amu" | "u" | "da" | "dalton" | "daltons" => Self::Mass(MassUnit::AtomicMassUnit),
            "kg" | "kilogram" | "kilograms" => Self::Mass(MassUnit::Kilogram),
            "g/mol" | "gram/mol" | "grams/mol" => Self::Mass(MassUnit::GramPerMol),
            _ => return Err(UnitError::UnknownUnit(name.to_string())),
        };
        Ok(unit)
    }
}

/// Return the multiplicative factor for conversion from `from` to `to`.
pub fn get_conversion_factor<F: IntoUnit, T: IntoUnit>(from: F, to: T) -> Result<f64, UnitError> {
    let from = from.into_unit()?;
    let to = to.into_unit()?;
    if from.kind() != to.kind() {
        return Err(UnitError::Incompatible {
            from: from.kind(),
            to: to.kind(),
        });
    }
    Ok(from.factor_to_base() / to.factor_to_base())
}

/// Convert a scalar from one unit to another.
pub fn convert<F: IntoUnit, T: IntoUnit>(value: f64, from: F, to: T) -> Result<f64, UnitError> {
    Ok(value * get_conversion_factor(from, to)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < 1e-10 * scale,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn constants_and_base_units_match_mdanalysis() {
        close(constant("N_Avogadro").unwrap(), 6.02214129e23);
        close(constant("calorie").unwrap(), 4.184);
        assert_eq!(MDANALYSIS_BASE_UNITS.length, "A");
        assert_eq!(MDANALYSIS_BASE_UNITS.time, "ps");
        assert_eq!(MDANALYSIS_BASE_UNITS.energy, "kJ/mol");
    }

    #[test]
    fn length_and_time_conversions() {
        close(convert(12.34567, "nm", "A").unwrap(), 123.4567);
        close(
            convert(123.4567, LengthUnit::Angstrom, LengthUnit::Nanometer).unwrap(),
            12.34567,
        );
        close(convert(1.0, "fs", "ns").unwrap(), 1.0e-6);
        close(convert(1.0, "ps", "AKMA").unwrap(), 20.45482949774598);
    }

    #[test]
    fn energy_force_speed_and_mass_conversions() {
        close(convert(1.0, "kcal/mol", "kJ/mol").unwrap(), 4.184);
        close(convert(1.0, "kcal/mol", "eV").unwrap(), 0.0433641022929);
        close(convert(2.5, "kJ/(mol*nm)", "kJ/(mol*A)").unwrap(), 0.25);
        close(convert(1.0, "A/ps", "m/s").unwrap(), 100.0);
        close(convert(1.0, "m/s", "A/fs").unwrap(), 1.0e-5);
        close(convert(1.0, "kg", "amu").unwrap(), 6.022140762081123e26);
    }

    #[test]
    fn incompatible_and_unknown_units_are_errors() {
        assert!(matches!(
            convert(1.0, "A", "ps"),
            Err(UnitError::Incompatible { .. })
        ));
        assert!(matches!(
            convert(1.0, "Stone", "nm"),
            Err(UnitError::UnknownUnit(_))
        ));
    }
}
