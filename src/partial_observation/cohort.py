"""Protocol-based cohort enumeration and per-site quality control.

``enumerate_cohort`` selects stimulation sites from the BIDS sidecars alone
(no waveforms are read); ``prepare_site`` loads the trials of one site, applies
the raw-epoch and window checks, and returns the analysed trial windows.
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .ccep_data import discover_eligible_runs, list_subjects, load_run_events
from .io import json_list
from .koopman import center, hard_window_qc, raw_epoch_qc, slice_trial


@dataclass(frozen=True)
class SiteCandidate:
    subject: str
    session: str
    age: int
    run: str
    run_sort: tuple[int, str]
    run_path: str
    site: str
    contacts: tuple[str, str]
    candidate_event_indices: tuple[int, ...]
    candidate_onset_samples: tuple[int, ...]
    candidate_prev_isi_s: tuple[float, ...]
    candidate_next_isi_s: tuple[float, ...]

    @property
    def n_candidate(self) -> int:
        return len(self.candidate_event_indices)


def run_id(path: Path) -> str:
    match = re.search(r"_run-([^_]+)_", path.name)
    return f"run-{match.group(1)}" if match else path.stem


def canonical_run_sort(run: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", run)
    return (int(match.group(1)) if match else sys.maxsize, run)


def sidecar(vhdr_path: Path, suffix: str) -> Path:
    return vhdr_path.with_name(vhdr_path.name.replace("_ieeg.vhdr", suffix))


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def row_sample(row: pd.Series, sf: float) -> float | None:
    value = finite_float(row.get("sample_start"))
    if value is not None:
        return value
    onset_s = finite_float(row.get("onset"))
    return None if onset_s is None else onset_s * sf


def parse_contacts(site: str) -> tuple[str, str] | None:
    parts = tuple(part.strip() for part in str(site).split("-") if part.strip())
    return parts if len(parts) == 2 else None


def evenly_subsample(items: list[Any], n: int) -> list[Any]:
    if len(items) < n:
        raise ValueError(f"Need {n} trials, found {len(items)}")
    if len(items) == n:
        return list(items)
    indices = np.linspace(0, len(items) - 1, n).round().astype(int)
    if len(np.unique(indices)) != n:
        raise RuntimeError("Even subsampling generated duplicate indices")
    return [items[int(index)] for index in indices]


def _protocol_rows(vhdr_path: Path, sf: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Return rows and the conservative clock used for all-stimulus ISI.

    In ds004080, stimulation-associated artefact/seizure annotations can replace
    or bracket an otherwise missing electrical-stimulation marker.  Therefore
    every timestamped raw events.tsv row remains in the exclusion clock.  Only
    fully described electrical-stimulation rows can become analysed trials.
    This is deliberately conservative and reproduces the audited 1788/50
    pre-waveform-QC cohort.
    """
    rows = read_tsv(sidecar(vhdr_path, "_events.tsv"))
    sample_values = []
    for _, row in rows.iterrows():
        sample = row_sample(row, sf)
        if sample is not None:
            sample_values.append(sample)
    return rows, np.unique(np.asarray(sample_values, dtype=float))


def _all_stim_isi(sample: float, all_samples: np.ndarray, sf: float) -> tuple[float, float]:
    position = int(np.searchsorted(all_samples, sample))
    if position >= len(all_samples) or not np.isclose(all_samples[position], sample,
                                                       rtol=0.0, atol=0.51):
        position = int(np.argmin(np.abs(all_samples - sample)))
    previous = np.inf if position == 0 else (sample - all_samples[position - 1]) / sf
    following = (np.inf if position == len(all_samples) - 1
                 else (all_samples[position + 1] - sample) / sf)
    return float(previous), float(following)


def enumerate_run_sites(subject: str, session: str, age: int,
                        vhdr_path: Path) -> list[SiteCandidate]:
    lightweight = load_run_events(vhdr_path)
    sf = float(lightweight["sf"])
    clean_events = list(lightweight["stim_events"])
    rows, all_stim_samples = _protocol_rows(vhdr_path, sf)
    channels = read_tsv(sidecar(vhdr_path, "_channels.tsv"))
    channel_types = {str(row["name"]): str(row.get("type", "")).upper()
                     for _, row in channels.iterrows()}

    protocol_by_key: dict[tuple[int, str], list[pd.Series]] = defaultdict(list)
    for _, row in rows.iterrows():
        site = str(row.get("electrical_stimulation_site", "")).strip()
        sample = row_sample(row, sf)
        if site and site.lower() not in {"n/a", "na"} and sample is not None:
            protocol_by_key[(int(round(sample)), site)].append(row)

    eligible: dict[str, list[tuple[int, int, float, float]]] = defaultdict(list)
    for event_index, event in enumerate(clean_events):
        site = str(event["site"])
        sample = int(round(float(event["onset_sample"])))
        contacts = parse_contacts(site)
        matching = protocol_by_key.get((sample, site), [])
        row = matching[0] if matching else None
        if contacts is None or row is None:
            continue
        if any(channel_types.get(contact) != "ECOG" for contact in contacts):
            continue
        current = finite_float(row.get("electrical_stimulation_current"))
        pulse_width = finite_float(row.get("electrical_stimulation_pulsewidth"))
        stimulation_type = str(row.get("electrical_stimulation_type", "")).lower()
        if current is None or not np.isclose(current, config.CURRENT_A, rtol=0, atol=1e-9):
            continue
        if pulse_width is None or not np.isclose(pulse_width, config.PULSE_WIDTH_S,
                                                 rtol=0, atol=1e-9):
            continue
        if stimulation_type != config.STIMULATION_TYPE:
            continue
        previous, following = _all_stim_isi(sample, all_stim_samples, sf)
        if previous + 1e-12 < config.MIN_ISI_S or following + 1e-12 < config.MIN_ISI_S:
            continue
        eligible[site].append((event_index, sample, previous, following))

    rid = run_id(vhdr_path)
    out = []
    for site, trials in eligible.items():
        if len(trials) < config.N_TRIALS:
            continue
        contacts = parse_contacts(site)
        assert contacts is not None
        out.append(SiteCandidate(
            subject=subject, session=session, age=int(age), run=rid,
            run_sort=canonical_run_sort(rid), run_path=str(vhdr_path), site=site,
            contacts=contacts,
            candidate_event_indices=tuple(item[0] for item in trials),
            candidate_onset_samples=tuple(item[1] for item in trials),
            candidate_prev_isi_s=tuple(item[2] for item in trials),
            candidate_next_isi_s=tuple(item[3] for item in trials),
        ))
    return out


def enumerate_cohort(subject_filter: set[str] | None = None,
                     max_subjects: int | None = None,
                     site_filter: set[str] | None = None,
                     progress: bool = True) -> tuple[list[SiteCandidate], pd.DataFrame]:
    subjects = [(str(s), str(se), int(a)) for s, se, a in list_subjects()]
    if subject_filter:
        subjects = [row for row in subjects if row[0] in subject_filter]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    all_candidates = []
    for subject_index, (subject, session, age) in enumerate(subjects, 1):
        if progress:
            print(f"[enumerate {subject_index}/{len(subjects)}] {subject}", flush=True)
        for path in discover_eligible_runs(subject, session, sf=config.SAMPLING_FREQUENCY_HZ):
            all_candidates.extend(enumerate_run_sites(subject, session, age, path))
    if site_filter:
        all_candidates = [item for item in all_candidates if item.site in site_filter]

    # One subject x site: most protocol-eligible trials, then canonical run ID.
    grouped: dict[tuple[str, str], list[SiteCandidate]] = defaultdict(list)
    for item in all_candidates:
        grouped[(item.subject, item.site)].append(item)
    selected = []
    selected_keys = set()
    for key, choices in grouped.items():
        choice = sorted(choices, key=lambda x: (-x.n_candidate, x.run_sort))[0]
        selected.append(choice)
        selected_keys.add((choice.subject, choice.site, choice.run))
    selected.sort(key=lambda x: (x.subject, x.site, x.run_sort))

    audit_rows = []
    for item in sorted(all_candidates,
                       key=lambda x: (x.subject, x.site, x.run_sort)):
        is_selected = (item.subject, item.site, item.run) in selected_keys
        audit_rows.append({
            "subject": item.subject, "session": item.session, "age": item.age,
            "run": item.run, "site": item.site,
            "stim_contact_1": item.contacts[0], "stim_contact_2": item.contacts[1],
            "n_candidate": item.n_candidate, "selected_run": is_selected,
            "selection_reason": ("max candidate count; canonical run tie-break"
                                 if is_selected else "duplicate subject x site"),
            "candidate_event_indices": json_list(item.candidate_event_indices),
            "candidate_onset_samples": json_list(item.candidate_onset_samples),
            "run_path": item.run_path,
        })
    return selected, pd.DataFrame(audit_rows)


def extract_window(segment: np.ndarray, t_ms: np.ndarray, channel_indices: np.ndarray,
                   span: tuple[float, float], max_delay: int) -> np.ndarray:
    mask = (t_ms >= span[0]) & (t_ms < span[1])
    out = np.asarray(segment[channel_indices][:, mask], dtype=float)
    if out.ndim != 2 or out.shape[0] < config.MIN_CHANNELS:
        raise ValueError(f"Only {out.shape[0]} recording channels remain")
    if out.shape[1] < max_delay + 3:
        raise ValueError(f"Window {span} has {out.shape[1]} samples")
    return center(out)


def prepare_site(candidate: SiteCandidate, run: dict[str, Any]) -> dict[str, Any]:
    sf = float(run["sf"])
    data = np.asarray(run["data"], dtype=float)
    names = [str(name) for name in run["ch_names"]]
    events = list(run["stim_events"])
    own_set = set(candidate.contacts)
    after_stim = np.asarray([i for i, name in enumerate(names) if name not in own_set],
                            dtype=int)
    if len(after_stim) < config.MIN_CHANNELS:
        raise RuntimeError("Too few good ECoG channels after stimulation-contact exclusion")
    after_stim_names = [names[i] for i in after_stim]

    # DMD needs 310 ms, while the current raw QC scans through +900 ms.
    guard_s = 1.0 / sf
    pre_s = max(abs(span[0]) for span in config.WINDOWS_MS.values()) / 1e3 + guard_s
    post_s = max(max(span[1] for span in config.WINDOWS_MS.values()),
                 config.RAW_QC["post_scan_ms"][1], config.RAW_QC["dc_window_ms"][1]) / 1e3 + guard_s

    epochs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    channel_frames = []
    boundary_errors: dict[int, str] = {}
    for event_index in candidate.candidate_event_indices:
        try:
            event = events[event_index]
            segment, t_ms = slice_trial(data, int(round(event["onset_sample"])), sf,
                                        pre_s=pre_s, post_s=post_s)
            raw = np.asarray(segment[after_stim], dtype=float)
            frame = raw_epoch_qc(raw, t_ms, sf, after_stim_names,
                                 event=event_index, **config.RAW_QC)
            frame.insert(0, "onset_sample", int(round(event["onset_sample"])))
            channel_frames.append(frame)
            epochs[event_index] = (np.asarray(segment), np.asarray(t_ms))
        except Exception as exc:
            boundary_errors[event_index] = f"{type(exc).__name__}: {exc}"

    if not channel_frames:
        raise RuntimeError("No candidate trial covers the raw-QC epoch")
    qc_channels = pd.concat(channel_frames, ignore_index=True)
    recurrence_n = max(config.RECURRENT_CHANNEL_MIN_TRIALS,
                       int(math.ceil(config.RECURRENT_CHANNEL_FRACTION * candidate.n_candidate)))
    hard_counts = qc_channels.groupby("channel", sort=False)["hard_flag"].sum()
    recurrent_names = hard_counts[hard_counts >= recurrence_n].index.astype(str).tolist()
    recurrent_set = set(recurrent_names)
    qc_channels["recurrent_hard_channel"] = qc_channels["channel"].isin(recurrent_set)
    final_indices = np.asarray([i for i in after_stim if names[i] not in recurrent_set],
                               dtype=int)
    final_names = [names[i] for i in final_indices]
    if len(final_indices) < config.MIN_CHANNELS:
        raise RuntimeError("Too few channels after recurrent hard-channel exclusion")

    accepted = []
    trial_rows = []
    for event_index, onset_sample, prev_isi, next_isi in zip(
            candidate.candidate_event_indices, candidate.candidate_onset_samples,
            candidate.candidate_prev_isi_s, candidate.candidate_next_isi_s):
        row = {
            "event_index": event_index, "onset_sample": onset_sample,
            "candidate": True, "prev_all_stim_isi_s": prev_isi,
            "next_all_stim_isi_s": next_isi, "raw_epoch_available": event_index in epochs,
            "raw_warning": False, "raw_hard_after_recurrent_exclusion": True,
            "pre_window_qc": False, "early_window_qc": False,
            "qc_passed": False, "selected": False,
            "failure_reason": boundary_errors.get(event_index, ""),
        }
        if event_index not in epochs:
            trial_rows.append(row)
            continue
        event_qc = qc_channels[qc_channels["event"] == event_index]
        remaining = event_qc[~event_qc["channel"].isin(recurrent_set)]
        row["raw_warning"] = bool(remaining["warning_flag"].any())
        row["raw_hard_after_recurrent_exclusion"] = bool(remaining["hard_flag"].any())
        if row["raw_hard_after_recurrent_exclusion"]:
            row["failure_reason"] = "raw_hard_after_recurrent_channel_exclusion"
            trial_rows.append(row)
            continue
        segment, t_ms = epochs[event_index]
        try:
            windows = {name: extract_window(segment, t_ms, final_indices, span,
                                             max(config.ROUTE_DELAYS.values()))
                       for name, span in config.WINDOWS_MS.items()}
            row["pre_window_qc"] = hard_window_qc(windows["pre"])
            row["early_window_qc"] = hard_window_qc(windows["early"])
            row["qc_passed"] = bool(row["pre_window_qc"] and row["early_window_qc"])
            if row["qc_passed"]:
                accepted.append({"event_index": event_index,
                                 "onset_sample": onset_sample, "windows": windows})
            else:
                row["failure_reason"] = "hard_window_qc"
        except Exception as exc:
            row["failure_reason"] = f"window_error: {type(exc).__name__}: {exc}"
        trial_rows.append(row)

    selected = evenly_subsample(accepted, config.N_TRIALS) if len(accepted) >= config.N_TRIALS else []
    selected_ids = {item["event_index"] for item in selected}
    for row in trial_rows:
        row["selected"] = row["event_index"] in selected_ids
    return {
        "selected": selected, "qc_trials": pd.DataFrame(trial_rows),
        "qc_channels": qc_channels, "initial_names": names,
        "after_stim_names": after_stim_names, "final_names": final_names,
        "final_indices": final_indices, "recurrent_names": recurrent_names,
        "recurrence_n": recurrence_n, "pre_s": pre_s, "post_s": post_s,
    }

