//! Timezone-aware day boundaries for "today" windows.
//!
//! All timestamps in the database are stored as naive UTC. User-facing
//! "today" windows (briefings, digests, sparklines) must nevertheless be
//! calendar-day boundaries **in the user's local timezone**, expressed back
//! as naive UTC for querying. This module centralizes that conversion so the
//! rest of the codebase never computes day boundaries by hand.

use chrono::offset::LocalResult;
use chrono::{DateTime, Duration, Local, NaiveDateTime, TimeZone, Utc};

/// A half-open `[start, end)` window of naive-UTC timestamps covering one
/// calendar day in some timezone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DayWindow {
    /// Naive UTC instant of local midnight starting the day.
    pub start: NaiveDateTime,
    /// Naive UTC instant of local midnight starting the next day.
    pub end: NaiveDateTime,
}

impl DayWindow {
    /// Window shifted backwards by `days` whole local days.
    ///
    /// For fixed-offset zones this is exact arithmetic. For zones with DST
    /// the shift is computed on the underlying local midnights by the
    /// caller's construction path; the simple subtraction here matches the
    /// fixed-offset and UTC semantics used in tests and is a safe
    /// approximation only when the caller does not rely on DST-exact edges.
    #[must_use]
    pub fn shifted_back(&self, days: i64) -> DayWindow {
        DayWindow {
            start: self.start - Duration::days(days),
            end: self.end - Duration::days(days),
        }
    }

    /// The start of the window `days_back` days before this one's day.
    /// Useful for "last N days" cutoffs ending at the current local day.
    #[must_use]
    pub fn cutoff_days_back(&self, days_back: i64) -> NaiveDateTime {
        self.start - Duration::days(days_back)
    }
}

/// Compute the current local-day window in `tz` for the instant `now`.
///
/// Both bounds are returned as naive UTC so they can be compared directly
/// against stored timestamps.
pub fn day_window_in_tz<Tz>(now: DateTime<Utc>, tz: &Tz) -> DayWindow
where
    Tz: TimeZone,
    Tz::Offset: std::fmt::Display,
{
    let local_now = now.with_timezone(tz);
    let local_midnight = local_now
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .expect("00:00:00 is always representable");
    let start = local_midnight_to_utc(local_midnight, tz).unwrap_or_else(|| now.naive_utc());
    let next_midnight = local_midnight + Duration::days(1);
    let end = local_midnight_to_utc(next_midnight, tz).unwrap_or(start + Duration::days(1));
    DayWindow { start, end }
}

/// The current local-day window in the system timezone.
pub fn local_day_window(now: DateTime<Utc>) -> DayWindow {
    day_window_in_tz(now, &Local)
}

/// Convert a local midnight to its naive-UTC instant.
///
/// Returns `None` only if the timezone database cannot resolve the instant
/// within a 3-hour scan (practically unreachable for real zones).
fn local_midnight_to_utc<Tz>(local: NaiveDateTime, tz: &Tz) -> Option<NaiveDateTime>
where
    Tz: TimeZone,
{
    match tz.from_local_datetime(&local) {
        LocalResult::Single(dt) | LocalResult::Ambiguous(dt, _) => Some(dt.naive_utc()),
        LocalResult::None => {
            // DST spring-forward gap: midnight does not exist (e.g. some
            // historical zones). Scan forward for the first valid instant.
            for minutes in 1..=180 {
                let candidate = local + Duration::minutes(minutes);
                if let LocalResult::Single(dt) | LocalResult::Ambiguous(dt, _) =
                    tz.from_local_datetime(&candidate)
                {
                    return Some(dt.naive_utc());
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        chrono::NaiveDate::from_ymd_opt(y, mo, d)
            .unwrap()
            .and_hms_opt(h, mi, 0)
            .unwrap()
            .and_utc()
    }

    #[test]
    fn window_for_utc_zone_is_utc_midnight() {
        let now = utc(2026, 8, 22, 15, 30);
        let win = day_window_in_tz(now, &Utc);
        assert_eq!(win.start, utc(2026, 8, 22, 0, 0).naive_utc());
        assert_eq!(win.end, utc(2026, 8, 23, 0, 0).naive_utc());
    }

    #[test]
    fn window_for_negative_offset_starts_later_in_utc() {
        // America/New_York in August is UTC-4. Local 2026-08-22 20:00 is
        // 2026-08-23 00:00 UTC — the *previous* UTC day, which is exactly
        // the bug this module exists to fix.
        let tz = FixedOffset::west_opt(4 * 3600).unwrap();
        let now = utc(2026, 8, 23, 0, 30); // 2026-08-22 20:30 local
        let win = day_window_in_tz(now, &tz);
        // Local day 2026-08-22 starts at 04:00 UTC.
        assert_eq!(win.start, utc(2026, 8, 22, 4, 0).naive_utc());
        assert_eq!(win.end, utc(2026, 8, 23, 4, 0).naive_utc());
    }

    #[test]
    fn window_for_positive_offset_starts_earlier_in_utc() {
        let tz = FixedOffset::east_opt(2 * 3600).unwrap();
        let now = utc(2026, 8, 22, 22, 30); // 2026-08-23 00:30 local
        let win = day_window_in_tz(now, &tz);
        // Local day 2026-08-23 starts at 2026-08-22 22:00 UTC.
        assert_eq!(win.start, utc(2026, 8, 22, 22, 0).naive_utc());
        assert_eq!(win.end, utc(2026, 8, 23, 22, 0).naive_utc());
    }

    #[test]
    fn cutoff_days_back_subtracts_whole_days() {
        let now = utc(2026, 8, 22, 15, 0);
        let win = day_window_in_tz(now, &Utc);
        assert_eq!(win.cutoff_days_back(6), utc(2026, 8, 16, 0, 0).naive_utc());
    }

    #[test]
    fn shifted_back_moves_both_bounds() {
        let now = utc(2026, 8, 22, 15, 0);
        let win = day_window_in_tz(now, &Utc).shifted_back(1);
        assert_eq!(win.start, utc(2026, 8, 21, 0, 0).naive_utc());
        assert_eq!(win.end, utc(2026, 8, 22, 0, 0).naive_utc());
    }

    #[test]
    fn local_day_window_matches_utc_when_system_is_utc() {
        // We cannot force the system zone, but the call must never panic and
        // must always produce a non-empty window ordered start < end.
        let win = local_day_window(Utc::now());
        assert!(win.start < win.end);
    }
}
