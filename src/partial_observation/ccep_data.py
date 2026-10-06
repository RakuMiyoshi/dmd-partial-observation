"""Read CCEP ECoG data from OpenNeuro ds004080.

Arrays follow the (channels, time) convention. Raw data are loaded through
MNE from the local BrainVision copy specified by the caller.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import mne
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_BIDS_ROOT = (
    Path(os.environ["PARTIAL_OBSERVATION_BIDS_ROOT"]).expanduser().resolve()
    if os.environ.get("PARTIAL_OBSERVATION_BIDS_ROOT")
    else None
)

def set_bids_root(path: Path | str) -> None:
    """Set the local root of OpenNeuro ds004080 explicitly."""
    global _BIDS_ROOT
    _BIDS_ROOT = Path(path).expanduser().resolve()

def _require_bids_root() -> Path:
    if _BIDS_ROOT is None:
        raise RuntimeError("Set PARTIAL_OBSERVATION_BIDS_ROOT or pass --bids-root. Raw ds004080 data are not bundled.")
    return _BIDS_ROOT


# ------------------------------------------------------------------
# TSV helper
# ------------------------------------------------------------------

def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ------------------------------------------------------------------
# Subject / run discovery
# ------------------------------------------------------------------

def list_subjects() -> list[tuple[str, str, int]]:
    """Return participant identifier, session, and age records."""
    out = []
    for r in _read_tsv(_require_bids_root() / "participants.tsv"):
        ses = r.get("session", "ses-1")
        try:
            age = int(float(r.get("age", "-1") or "-1"))
        except (ValueError, TypeError):
            age = -1
        out.append((r["participant_id"], ses, age))
    return out


def discover_runs(sub: str, ses: str) -> list[Path]:
    """Return sorted task-SPESclin BrainVision header paths."""
    ieeg_dir = _require_bids_root() / sub / ses / "ieeg"
    return sorted(ieeg_dir.glob(f"{sub}_{ses}_task-SPESclin_run-*_ieeg.vhdr"))


# ------------------------------------------------------------------
# Run loading
# ------------------------------------------------------------------

def _sidecar(vhdr_path: Path, suffix: str) -> Path:
    """Return the sidecar path associated with a BrainVision header."""
    return vhdr_path.with_name(vhdr_path.name.replace("_ieeg.vhdr", suffix))


def run_sampling_frequency(vhdr_path: Path | str) -> float:
    """Read the sampling frequency recorded in the sidecar JSON."""
    vhdr_path = Path(vhdr_path)
    with _sidecar(vhdr_path, "_ieeg.json").open("r", encoding="utf-8") as fh:
        return float(json.load(fh)["SamplingFrequency"])


def discover_eligible_runs(sub: str, ses: str, *, sf: float = 2048.0,
                           atol: float = 0.1) -> list[Path]:
    """Return runs matching the requested sampling frequency without resampling."""
    return [p for p in discover_runs(sub, ses)
            if abs(run_sampling_frequency(p) - sf) <= atol]


def _good_ecog_channels(vhdr_path: Path, raw_ch_names: list[str]) -> list[str]:
    """Return good ECoG channels in the order used by the raw recording."""
    good = {
        r["name"]
        for r in _read_tsv(_sidecar(vhdr_path, "_channels.tsv"))
        if r.get("type", "").upper() == "ECOG"
        and r.get("status", "good").lower() == "good"
    }
    return [ch for ch in raw_ch_names if ch in good]


_BAD_TRIAL_TYPES = ("artefact", "seizure")


def _row_sample(r: dict[str, str], sf: float, *, end: bool = False) -> int | None:
    """Return a row start or end sample, preferring explicit sample columns."""
    key = "sample_end" if end else "sample_start"
    try:
        return int(round(float(r[key])))
    except (KeyError, ValueError, TypeError):
        pass
    try:
        return int(round(float(r["offset" if end else "onset"]) * sf))
    except (KeyError, ValueError, TypeError):
        return None


def _bad_intervals(rows: list[dict[str, str]], sf: float) -> list[tuple[int, int]]:
    """Return inclusive sample intervals marked as artefact or seizure."""
    intervals = []
    for r in rows:
        if r.get("trial_type", "") not in _BAD_TRIAL_TYPES:
            continue
        start = _row_sample(r, sf, end=False)
        if start is None:
            continue
        end = _row_sample(r, sf, end=True)
        if end is None or end < start:
            try:
                end = start + int(round(float(r["duration"]) * sf))
            except (KeyError, ValueError, TypeError):
                end = start
        intervals.append((start, max(start, end)))
    return intervals


def _parse_stim_events(vhdr_path: Path, sf: float) -> list[dict[str, Any]]:
    """Read stimulation events and exclude events inside bad intervals."""
    rows = _read_tsv(_sidecar(vhdr_path, "_events.tsv"))
    intervals = _bad_intervals(rows, sf)

    def in_bad(s: int) -> bool:
        return any(a <= s <= b for a, b in intervals)

    events = []
    for r in rows:
        site = r.get("electrical_stimulation_site", "n/a")
        if not site or site == "n/a":
            continue
        if r.get("trial_type", "") in _BAD_TRIAL_TYPES:
            continue
        try:
            onset_sample = int(round(float(r["sample_start"])))
        except (KeyError, ValueError, TypeError):
            try:
                onset_sample = int(round(float(r["onset"]) * sf))
            except (KeyError, ValueError, TypeError):
                continue
        if in_bad(onset_sample):
            continue
        events.append({"onset_sample": onset_sample, "site": site})
    return events


def load_run_events(vhdr_path: Path | str) -> dict[str, Any]:
    """Read event metadata without loading waveform data."""
    vhdr_path = Path(vhdr_path)
    sf = run_sampling_frequency(vhdr_path)
    return {
        "sf": sf,
        "stim_events": _parse_stim_events(vhdr_path, sf),
        "vhdr_path": vhdr_path,
    }


def load_run(vhdr_path: Path | str) -> dict[str, Any]:
    """Load one CCEP run and return good ECoG channels in microvolts."""
    vhdr_path = Path(vhdr_path)

    # Treat the sidecar JSON as the authoritative sampling frequency.
    sf = run_sampling_frequency(vhdr_path)

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    if abs(raw.info["sfreq"] - sf) > 0.1:
        print(f"  [WARN] MNE sfreq {raw.info['sfreq']} != json sf {sf}; using json")

    good_chs = _good_ecog_channels(vhdr_path, raw.ch_names)
    if not good_chs:
        raise RuntimeError(f"No good ECOG channels found for {vhdr_path.name}")

    data = raw.pick(good_chs).get_data(return_times=False, units="uV").astype(np.float64)

    return {
        "data": data,
        "sf": sf,
        "ch_names": good_chs,
        "stim_events": _parse_stim_events(vhdr_path, sf),
        "vhdr_path": vhdr_path,
    }


# Electrode metadata

def load_electrodes(sub: str, ses: str) -> dict[str, dict[str, Any]]:
    """Return electrode labels and coordinates keyed by channel name."""
    el_path = _require_bids_root() / sub / ses / "ieeg" / f"{sub}_{ses}_electrodes.tsv"
    out: dict[str, dict[str, Any]] = {}
    for r in _read_tsv(el_path):
        name = r.get("name", "")
        if not name:
            continue
        try:
            destrieux_nr = int(r.get("Destrieux_label", "-1") or "-1")
        except (ValueError, TypeError):
            destrieux_nr = -1
        try:
            xyz = np.array(
                [float(r.get(k, "nan") or "nan") for k in ("x", "y", "z")],
                dtype=np.float64,
            )
        except (ValueError, TypeError):
            xyz = np.full(3, np.nan)
        out[name] = {
            "destrieux_nr": destrieux_nr,
            "destrieux_text": r.get("Destrieux_label_text", ""),
            "xyz": xyz,
        }
    return out
