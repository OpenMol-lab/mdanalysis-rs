//! Binary-state time-correlation utilities.
//!
//! These routines mirror the reusable core of
//! `MDAnalysis.lib.correlations`: each frame is represented by a set of
//! identifiers that currently satisfy a property (for example, waters within
//! a cutoff or hydrogen bonds that are formed).  The implementation is generic
//! over the identifier type and does not require a particular topology.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

/// Errors returned by correlation calculations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CorrelationError {
    /// No frame sets were supplied.
    EmptyInput,
    /// `tau_max` is larger than the number of input frames.
    TauExceedsFrames { tau_max: usize, frames: usize },
    /// A window step of zero cannot advance time origins.
    InvalidWindowStep,
    /// Intermittency must be zero or a non-negative integer (the Rust type
    /// already enforces non-negativity, so this variant is reserved for API
    /// symmetry and diagnostics).
    InvalidIntermittency,
}

impl fmt::Display for CorrelationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => {
                formatter.write_str("correlation input must contain at least one frame")
            }
            Self::TauExceedsFrames { tau_max, frames } => {
                write!(formatter, "tau_max {tau_max} exceeds {frames} input frames")
            }
            Self::InvalidWindowStep => formatter.write_str("window_step must be positive"),
            Self::InvalidIntermittency => formatter.write_str("intermittency is invalid"),
        }
    }
}

impl std::error::Error for CorrelationError {}

/// Autocorrelation output: lag values, mean survival values, and raw
/// time-origin samples for each positive lag.
pub type AutocorrelationResult = (Vec<usize>, Vec<f64>, Vec<Vec<f64>>);

/// Calculate a continuous binary autocorrelation (survival probability).
///
/// The returned tuple contains lag values `0..=tau_max`, the mean survival
/// probability at each lag, and the per-time-origin values used to calculate
/// each mean. Empty time origins (frames with no active identifiers) are
/// skipped. Lag zero is defined as exactly one, matching MDAnalysis.
pub fn autocorrelation<T>(
    frames: &[HashSet<T>],
    tau_max: usize,
    window_step: usize,
) -> Result<AutocorrelationResult, CorrelationError>
where
    T: Eq + Hash + Clone,
{
    if frames.is_empty() {
        return Err(CorrelationError::EmptyInput);
    }
    if tau_max > frames.len() {
        return Err(CorrelationError::TauExceedsFrames {
            tau_max,
            frames: frames.len(),
        });
    }
    if window_step == 0 {
        return Err(CorrelationError::InvalidWindowStep);
    }
    let mut raw = vec![Vec::new(); tau_max];
    for origin in (0..frames.len()).step_by(window_step) {
        let initial = &frames[origin];
        if initial.is_empty() {
            continue;
        }
        for tau in 1..=tau_max {
            let end = origin + tau;
            if end >= frames.len() {
                break;
            }
            let mut surviving = initial.clone();
            for frame in &frames[origin + 1..=end] {
                surviving.retain(|item| frame.contains(item));
            }
            raw[tau - 1].push(surviving.len() as f64 / initial.len() as f64);
        }
    }
    let means: Vec<f64> = raw
        .iter()
        .map(|values| {
            if values.is_empty() {
                f64::NAN
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        })
        .collect();
    let mut taus = Vec::with_capacity(tau_max + 1);
    let mut values = Vec::with_capacity(tau_max + 1);
    taus.push(0);
    values.push(1.0);
    taus.extend(1..=tau_max);
    values.extend(means);
    Ok((taus, values, raw))
}

/// Fill short absences in a sequence of active-identifier sets.
///
/// An identifier missing for at most `intermittency` consecutive frames and
/// then observed again is inserted into that gap.  The input is cloned, so
/// callers retain their original observations. With zero intermittency the
/// cloned sequence is returned unchanged.
pub fn correct_intermittency<T>(
    frames: &[HashSet<T>],
    intermittency: usize,
) -> Result<Vec<HashSet<T>>, CorrelationError>
where
    T: Eq + Hash + Clone,
{
    if frames.is_empty() {
        return Err(CorrelationError::EmptyInput);
    }
    if intermittency == 0 {
        return Ok(frames.to_vec());
    }
    let mut corrected = frames.to_vec();
    for origin in 0..frames.len() {
        let candidates: Vec<T> = frames[origin].iter().cloned().collect();
        for item in candidates {
            let mut gap = 0usize;
            for end in origin + 1..frames.len() {
                if corrected[end].contains(&item) {
                    if gap > 0 && gap <= intermittency {
                        for frame in &mut corrected[end - gap..end] {
                            frame.insert(item.clone());
                        }
                    }
                    break;
                }
                gap += 1;
                if gap > intermittency {
                    break;
                }
            }
        }
    }
    Ok(corrected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[u32]) -> HashSet<u32> {
        values.iter().copied().collect()
    }

    #[test]
    fn autocorrelation_matches_continuous_survival_definition() {
        let frames = vec![set(&[1, 2]), set(&[1]), set(&[1, 2]), set(&[2])];
        let (taus, values, raw) = autocorrelation(&frames, 3, 1).unwrap();
        assert_eq!(taus, vec![0, 1, 2, 3]);
        assert_eq!(raw[0], vec![0.5, 1.0, 0.5]);
        assert_eq!(raw[1], vec![0.5, 0.0]);
        assert_eq!(raw[2], vec![0.0]);
        assert_eq!(values[0], 1.0);
        assert!((values[1] - (2.0 / 3.0)).abs() < 1.0e-12);
        assert!((values[2] - 0.25).abs() < 1.0e-12);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn intermittency_fills_only_short_returning_gaps() {
        let frames = vec![set(&[7]), set(&[]), set(&[]), set(&[7]), set(&[])];
        let corrected = correct_intermittency(&frames, 2).unwrap();
        assert_eq!(
            corrected,
            vec![set(&[7]), set(&[7]), set(&[7]), set(&[7]), set(&[])]
        );
        let unchanged = correct_intermittency(&frames, 0).unwrap();
        assert_eq!(unchanged, frames);
    }

    #[test]
    fn invalid_correlation_inputs_are_reported() {
        assert_eq!(
            autocorrelation::<u32>(&[], 0, 1),
            Err(CorrelationError::EmptyInput)
        );
        let frames = vec![set(&[1])];
        assert_eq!(
            autocorrelation(&frames, 2, 1),
            Err(CorrelationError::TauExceedsFrames {
                tau_max: 2,
                frames: 1
            })
        );
        assert_eq!(
            autocorrelation(&frames, 0, 0),
            Err(CorrelationError::InvalidWindowStep)
        );
        assert_eq!(
            correct_intermittency::<u32>(&[], 1),
            Err(CorrelationError::EmptyInput)
        );
    }
}
