"""Step 1: fit every analysed stimulation site with single-trial Hankel DMD.

For each site, the ten selected trials of each window (pre- and post-stimulus)
are fitted one at a time at every delay depth, and the fixed eigenpairs of each
fit are evaluated on the other nine trials ("one-to-nine" validation). All
eigenpairs are stored: no stability, frequency, residual or decay filter is
applied here. The multi-step decay diagnostics on the held-out trials are
computed here as well; the validation rule is applied downstream
(``02_select_and_count.py``).

Outputs, under ``<output-root>/<analysis-id>/``:

* ``candidate_runs.csv``, ``candidate_summary.json``: protocol-based cohort;
* ``per_mode/<site>.csv.gz``: one row per eigenpair and fit, with held-out
  residuals and modal amplitudes on each of the nine held-out trials;
* ``npz/<site>.npz``: physical modes, singular values and fit metadata;
* ``decay/<site>__<route>.npz``: held-out one-step residuals and k-step
  coefficients of every eigenpair (the input of the decay check);
* ``qc/``, ``per_site/``, ``manifests/``: QC audit and completion markers.

Sites are written atomically, so an interrupted run resumes where it stopped.
Use ``--dry-run`` to reproduce the cohort enumeration without reading waveforms.
"""
from __future__ import annotations

import os

# Pin BLAS to one thread before NumPy loads; parallelism is across sites.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import concurrent.futures
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from partial_observation import config
from partial_observation.ccep_data import load_run, set_bids_root
from partial_observation.cohort import SiteCandidate, enumerate_cohort, prepare_site
from partial_observation.io import (
    REPOSITORY_ROOT, atomic_csv, atomic_json, atomic_npz, json_list, provenance,
    safe_name, utc_now,
)
from partial_observation.koopman import (
    EPS, evaluate_fixed_modes, fit_koopman, held_out_decay_diagnostics,
)

# Output schema tag; unchanged since the frozen reference artifacts were written.
SCHEMA_VERSION = "ccep-one-to-nine-primary300-v1"
SOURCES = [Path(__file__)] + [REPOSITORY_ROOT / "src/partial_observation" / name
                              for name in ("config.py", "ccep_data.py", "cohort.py",
                                           "koopman.py")]


def unit_modes(modes_phys: np.ndarray) -> np.ndarray:
    """Unit-norm physical modes, phase-anchored so the largest channel is real positive."""
    modes = np.asarray(modes_phys, dtype=np.complex128).copy()
    for mode in modes:
        norm = float(np.linalg.norm(mode))
        if not np.isfinite(norm) or norm <= EPS:
            mode[:] = np.nan + 0j
            continue
        mode /= norm
        anchor = int(np.argmax(np.abs(mode)))
        mode *= np.exp(-1j * np.angle(mode[anchor]))
        if mode[anchor].real < 0:
            mode *= -1
    return modes.astype(np.complex64)


def fit_site(candidate: SiteCandidate, prepared: dict[str, Any], *, routes: dict[str, int],
             rank: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit each selected trial alone and evaluate its eigenpairs on the other nine.

    Returns the per-mode table, the site artifact (physical modes and fit
    metadata), and per route the held-out multi-step decay diagnostics.
    """
    selected = prepared["selected"]
    event_ids = np.asarray([item["event_index"] for item in selected], dtype=np.int64)
    onset_samples = np.asarray([item["onset_sample"] for item in selected], dtype=np.int64)
    sf = config.SAMPLING_FREQUENCY_HZ
    rows = []
    raw_mode_blocks = []
    unit_mode_blocks = []
    mode_uids = []
    fit_mode_offsets = [0]
    fit_sval_offsets = [0]
    singular_values = []
    fit_window = []
    fit_route = []
    fit_delay = []
    fit_train_position = []
    fit_train_event = []
    fit_test_positions = []
    fit_test_events = []
    fit_actual_rank = []
    fit_uncapped_rank = []
    fit_rank_cap = []
    fit_rank_capped = []
    fit_retained_energy = []
    fit_error = []
    horizons_k = [max(1, int(round(h * sf / 1000))) for h in config.DECAY_HORIZONS_MS]
    decay_blocks: dict[str, list[dict[str, Any]]] = {route: [] for route in routes}

    for window_name in config.WINDOWS_MS:
        trials = [item["windows"][window_name] for item in selected]
        for route, delay in routes.items():
            for train_position in range(config.N_TRIALS):
                test_positions = np.asarray([i for i in range(config.N_TRIALS)
                                             if i != train_position], dtype=np.int64)
                model = fit_koopman([trials[train_position]], sf, d=delay,
                                    rank=rank)
                heldout = [evaluate_fixed_modes(
                    model, trials[int(test_position)], d=delay, sf=sf,
                    initial_ms=config.INITIAL_MODAL_AMPLITUDE_MS,
                ) for test_position in test_positions]
                n_common = trials[train_position].shape[1] - max(config.ROUTE_DELAYS.values()) + 1
                decay = held_out_decay_diagnostics(
                    model, [trials[int(q)] for q in test_positions], d=delay,
                    horizons_k=horizons_k, n_common=n_common)
                raw_modes = np.asarray(model.modes_phys, dtype=np.complex64)
                normalized_modes = unit_modes(model.modes_phys)
                raw_mode_blocks.append(raw_modes)
                unit_mode_blocks.append(normalized_modes)
                singular_values.extend(np.asarray(model.svals, dtype=float).tolist())

                fit_window.append(window_name)
                fit_route.append(route)
                fit_delay.append(delay)
                fit_train_position.append(train_position)
                fit_train_event.append(event_ids[train_position])
                fit_test_positions.append(test_positions)
                fit_test_events.append(event_ids[test_positions])
                fit_actual_rank.append(model.rank)
                fit_uncapped_rank.append(model.rank_uncapped)
                fit_rank_cap.append(model.rank_cap)
                fit_rank_capped.append(model.rank_capped)
                fit_retained_energy.append(model.retained_energy)
                fit_error.append(model.fit_error)

                for mode_index in range(len(model.eigs)):
                    residuals = np.asarray([item["residual"][mode_index]
                                            for item in heldout], dtype=float)
                    amplitudes = np.asarray([item["modal_amplitude_uV"][mode_index]
                                             for item in heldout], dtype=float)
                    initial_amplitudes = np.asarray([
                        item["initial_modal_amplitude_uV"][mode_index]
                        for item in heldout], dtype=float)
                    uid = "/".join((candidate.subject, candidate.run, candidate.site,
                                    window_name, route, str(event_ids[train_position]),
                                    str(mode_index)))
                    common = {
                        "schema_version": SCHEMA_VERSION,
                        "mode_uid": uid, "subject": candidate.subject,
                        "session": candidate.session, "age": candidate.age,
                        "run": candidate.run, "site": candidate.site,
                        "window": window_name,
                        "window_start_ms": config.WINDOWS_MS[window_name][0],
                        "window_stop_ms": config.WINDOWS_MS[window_name][1],
                        "route": route, "hankel_d": delay,
                        "train_trial_position": train_position,
                        "train_event_index": int(event_ids[train_position]),
                        "train_onset_sample": int(onset_samples[train_position]),
                        "mode_index": mode_index,
                        "eig_real": float(model.eigs[mode_index].real),
                        "eig_imag": float(model.eigs[mode_index].imag),
                        "eig_abs": float(abs(model.eigs[mode_index])),
                        "frequency_hz": float(model.freq_hz[mode_index]),
                        "decay_rate_s_inv": float(model.decay_rate[mode_index]),
                        "tau_ms": float(model.tau_ms[mode_index]),
                        "train_residual": float(model.res_train[mode_index]),
                        "train_activation": float(model.activation[mode_index]),
                        "train_weight": float(model.weight[mode_index]),
                        "phys_norm": float(model.phys_norm[mode_index]),
                        "physical_mode_normalization": "raw plus unit_l2_phase_anchored",
                        "phase_convention": "largest_abs_channel_real_positive",
                        "requested_rank": rank,
                        "actual_rank": int(model.rank),
                        "uncapped_rank": int(model.rank_uncapped),
                        "rank_cap": int(model.rank_cap),
                        "rank_capped": bool(model.rank_capped),
                        "retained_energy": float(model.retained_energy),
                        "fit_error": float(model.fit_error),
                        "median_heldout_residual": float(np.nanmedian(residuals)),
                        "median_heldout_modal_amplitude_uV": float(np.nanmedian(amplitudes)),
                        "median_heldout_initial_modal_amplitude_uV": float(
                            np.nanmedian(initial_amplitudes)),
                    }
                    for heldout_index, test_position in enumerate(test_positions):
                        common[f"heldout_{heldout_index}_trial_position"] = int(test_position)
                        common[f"heldout_{heldout_index}_event_index"] = int(
                            event_ids[test_position])
                        common[f"heldout_{heldout_index}_residual"] = float(
                            residuals[heldout_index])
                        common[f"heldout_{heldout_index}_modal_amplitude_uV"] = float(
                            amplitudes[heldout_index])
                        common[f"heldout_{heldout_index}_initial_modal_amplitude_uV"] = float(
                            initial_amplitudes[heldout_index])
                    rows.append(common)
                    mode_uids.append(uid)
                n_modes = len(model.eigs)
                decay_blocks[route].append(dict(
                    **decay, eig=model.eigs,
                    mode_uid=np.asarray(mode_uids[-n_modes:], dtype=str),
                    mode_index=np.arange(n_modes),
                    train_trial_position=np.full(n_modes, train_position),
                    train_event_index=np.full(n_modes, event_ids[train_position]),
                    window=np.full(n_modes, window_name),
                    test_trial_position=np.tile(test_positions, (n_modes, 1)),
                    test_event_index=np.tile(event_ids[test_positions], (n_modes, 1))))
                fit_mode_offsets.append(fit_mode_offsets[-1] + len(model.eigs))
                fit_sval_offsets.append(fit_sval_offsets[-1] + len(model.svals))

    artifact = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "mode_uid": np.asarray(mode_uids, dtype=str),
        "physical_mode_raw": np.concatenate(raw_mode_blocks, axis=0),
        "physical_mode_unit_l2_phase_anchored": np.concatenate(unit_mode_blocks, axis=0),
        "fit_mode_offset": np.asarray(fit_mode_offsets, dtype=np.int64),
        "fit_singular_value_offset": np.asarray(fit_sval_offsets, dtype=np.int64),
        "fit_singular_value": np.asarray(singular_values, dtype=np.float64),
        "fit_window": np.asarray(fit_window, dtype=str),
        "fit_route": np.asarray(fit_route, dtype=str),
        "fit_hankel_d": np.asarray(fit_delay, dtype=np.int64),
        "fit_train_trial_position": np.asarray(fit_train_position, dtype=np.int64),
        "fit_train_event_index": np.asarray(fit_train_event, dtype=np.int64),
        "fit_test_trial_positions": np.asarray(fit_test_positions, dtype=np.int64),
        "fit_test_event_indices": np.asarray(fit_test_events, dtype=np.int64),
        "fit_requested_rank": np.full(len(fit_window), rank, dtype=np.int64),
        "fit_actual_rank": np.asarray(fit_actual_rank, dtype=np.int64),
        "fit_uncapped_rank": np.asarray(fit_uncapped_rank, dtype=np.int64),
        "fit_rank_cap": np.asarray(fit_rank_cap, dtype=np.int64),
        "fit_rank_capped": np.asarray(fit_rank_capped, dtype=bool),
        "fit_retained_energy": np.asarray(fit_retained_energy, dtype=np.float64),
        "fit_error": np.asarray(fit_error, dtype=np.float64),
        "channel_name_initial_good_ecog": np.asarray(prepared["initial_names"], dtype=str),
        "channel_name_after_stim_exclusion": np.asarray(prepared["after_stim_names"], dtype=str),
        "channel_name_final": np.asarray(prepared["final_names"], dtype=str),
        "candidate_event_indices": np.asarray(candidate.candidate_event_indices,
                                              dtype=np.int64),
        "candidate_onset_samples": np.asarray(candidate.candidate_onset_samples,
                                              dtype=np.int64),
        "qc_passed_event_indices": prepared["qc_trials"].loc[
            prepared["qc_trials"]["qc_passed"], "event_index"].to_numpy(dtype=np.int64),
        "selected_event_indices": event_ids,
        "selected_onset_samples": onset_samples,
    }
    decay_arrays = {}
    for route, blocks in decay_blocks.items():
        arrays = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        arrays.update(
            horizon_k=np.array(horizons_k),
            horizon_requested_ms=np.array(config.DECAY_HORIZONS_MS),
            horizon_realised_ms=1000 * np.array(horizons_k) / sf, sf_hz=np.array(sf),
            range_name=np.array(["full", "common_d50_snapshot_count"]),
            subject=np.array(candidate.subject), run=np.array(candidate.run),
            site=np.array(candidate.site), route=np.array(route),
            delay=np.array(routes[route]), selected_event_indices=event_ids,
            channel_name=np.array(prepared["final_names"]))
        decay_arrays[route] = arrays
    return pd.DataFrame(rows), artifact, decay_arrays


def site_stem(candidate: SiteCandidate) -> str:
    return "__".join(map(safe_name, (candidate.subject, candidate.run, candidate.site)))


def site_paths(output: Path, candidate: SiteCandidate,
               routes: Iterable[str] = ()) -> dict[str, Path]:
    stem = site_stem(candidate)
    decay = {f"decay_{route}": output / "decay" / f"{stem}__{route}.npz"
             for route in routes}
    return decay | {
        "site": output / "per_site" / f"{stem}.csv",
        "mode": output / "per_mode" / f"{stem}.csv.gz",
        "npz": output / "npz" / f"{stem}.npz",
        "trial_qc": output / "qc" / f"{stem}__trial.csv.gz",
        "channel_qc": output / "qc" / f"{stem}__channel.csv.gz",
        "manifest": output / "manifests" / f"{stem}.json",
        "failure": output / "failures" / f"{stem}.json",
    }


def validate_complete_site(paths: dict[str, Path], deep: bool = False) -> bool:
    if not paths["manifest"].exists():
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        expected = {key for key in paths if key not in {"manifest", "failure"}}
        if set(manifest.get("files", {})) != expected:
            return False
        for key in expected:
            if not paths[key].exists() or paths[key].stat().st_size == 0:
                return False
        if deep:
            modes = pd.read_csv(paths["mode"], usecols=["mode_uid"])
            with np.load(paths["npz"], allow_pickle=False) as artifact:
                if not np.array_equal(modes["mode_uid"].astype(str).to_numpy(),
                                      artifact["mode_uid"].astype(str)):
                    return False
            if len(modes) != int(manifest["n_modes"]):
                return False
        return True
    except Exception:
        return False


def validate_terminal_site(paths: dict[str, Path], deep: bool = False) -> bool:
    """A complete fit or deterministic QC exclusion is resumable terminal state."""
    if not paths["manifest"].exists():
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return validate_complete_site(paths, deep=deep)
        if (manifest.get("status") == "qc_excluded"
                and manifest.get("schema_version") == SCHEMA_VERSION):
            return all(paths[key].exists() and paths[key].stat().st_size > 0
                       for key in ("site", "trial_qc", "channel_qc"))
        return False
    except Exception:
        return False


def site_metadata_frame(candidate: SiteCandidate, prepared: dict[str, Any],
                        status: str, *, routes: dict[str, int], rank: int) -> pd.DataFrame:
    qc_passed = prepared["qc_trials"].loc[
        prepared["qc_trials"]["qc_passed"], "event_index"].astype(int).tolist()
    selected = [item["event_index"] for item in prepared["selected"]]
    exclusion_reasons = {name: "recurrent_raw_epoch_hard_qc"
                         for name in prepared["recurrent_names"]}
    exclusion_reasons.update({name: "stimulation_contact"
                              for name in candidate.contacts
                              if name in prepared["initial_names"]})
    return pd.DataFrame([{
        "schema_version": SCHEMA_VERSION, "status": status,
        "subject": candidate.subject, "session": candidate.session,
        "age": candidate.age, "run": candidate.run, "site": candidate.site,
        "stim_contact_1": candidate.contacts[0],
        "stim_contact_2": candidate.contacts[1],
        "sampling_frequency_hz": config.SAMPLING_FREQUENCY_HZ,
        "n_candidate": candidate.n_candidate, "n_qc_passed": len(qc_passed),
        "n_selected": len(selected),
        "candidate_event_indices": json_list(candidate.candidate_event_indices),
        "qc_passed_event_indices": json_list(qc_passed),
        "selected_event_indices": json_list(selected),
        "channels_initial_good_ecog": json_list(prepared["initial_names"]),
        "channels_after_stim_exclusion": json_list(prepared["after_stim_names"]),
        "channels_final": json_list(prepared["final_names"]),
        "channel_exclusion_reasons": json.dumps(exclusion_reasons, ensure_ascii=False),
        "recurrent_hard_threshold_n": prepared["recurrence_n"],
        "windows_ms": json.dumps(config.WINDOWS_MS), "routes": json.dumps(routes),
        "requested_rank": rank, "epoch_pre_s": prepared["pre_s"],
        "epoch_post_s": prepared["post_s"], "run_path": candidate.run_path,
    }])


def process_site(candidate: SiteCandidate, run: dict[str, Any], output: Path, *,
                 routes: dict[str, int], rank: int, resume: bool = True) -> dict[str, Any]:
    paths = site_paths(output, candidate, routes)
    if resume and validate_terminal_site(paths):
        terminal_status = json.loads(paths["manifest"].read_text(
            encoding="utf-8"))["status"]
        return {"status": f"skipped_{terminal_status}", "subject": candidate.subject,
                "run": candidate.run, "site": candidate.site}
    # The manifest is the transaction commit marker.  Never leave an older
    # marker visible while recomputing an incomplete/forced site.
    paths["manifest"].unlink(missing_ok=True)
    try:
        prepared = prepare_site(candidate, run)
        n_qc_passed = int(prepared["qc_trials"]["qc_passed"].sum())
        if n_qc_passed < config.N_TRIALS:
            # A forced recomputation may replace an older complete fit.
            paths["mode"].unlink(missing_ok=True)
            for key in paths:
                if key == "npz" or key.startswith("decay_"):
                    paths[key].unlink(missing_ok=True)
            site_frame = site_metadata_frame(candidate, prepared, "qc_excluded",
                                             routes=routes, rank=rank)
            atomic_csv(site_frame, paths["site"])
            atomic_csv(prepared["qc_trials"], paths["trial_qc"])
            atomic_csv(prepared["qc_channels"], paths["channel_qc"])
            manifest = {
                "schema_version": SCHEMA_VERSION, "status": "qc_excluded",
                "created_utc": utc_now(), "subject": candidate.subject,
                "run": candidate.run, "site": candidate.site,
                "reason": f"QC-passed trials {n_qc_passed} < {config.N_TRIALS}",
                "n_candidate": candidate.n_candidate,
                "n_qc_passed": n_qc_passed, "n_selected": 0,
                "files": {key: str(paths[key].relative_to(output))
                          for key in ("site", "trial_qc", "channel_qc")},
            }
            atomic_json(manifest, paths["manifest"])
            paths["failure"].unlink(missing_ok=True)
            return {"status": "qc_excluded", "subject": candidate.subject,
                    "run": candidate.run, "site": candidate.site,
                    "n_qc_passed": n_qc_passed}
        modes, artifact, decay = fit_site(candidate, prepared, routes=routes, rank=rank)
        if modes["mode_uid"].duplicated().any():
            raise RuntimeError("mode_uid is not unique within site")
        if len(modes) != len(artifact["mode_uid"]):
            raise RuntimeError("CSV/NPZ mode count mismatch")

        qc_passed = prepared["qc_trials"].loc[
            prepared["qc_trials"]["qc_passed"], "event_index"].astype(int).tolist()
        selected = [item["event_index"] for item in prepared["selected"]]
        site_row = site_metadata_frame(candidate, prepared, "complete",
                                       routes=routes, rank=rank)
        atomic_csv(site_row, paths["site"])
        atomic_csv(modes, paths["mode"])
        atomic_npz(paths["npz"], **artifact)
        for route, arrays in decay.items():
            atomic_npz(paths[f"decay_{route}"], **arrays)
        atomic_csv(prepared["qc_trials"], paths["trial_qc"])
        atomic_csv(prepared["qc_channels"], paths["channel_qc"])
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "complete",
            "created_utc": utc_now(), "subject": candidate.subject,
            "run": candidate.run, "site": candidate.site,
            "n_modes": len(modes), "n_fits": len(config.WINDOWS_MS) * len(routes) * config.N_TRIALS,
            "n_candidate": candidate.n_candidate, "n_qc_passed": len(qc_passed),
            "n_selected": len(selected),
            "files": {key: str(path.relative_to(output)) for key, path in paths.items()
                      if key not in {"manifest", "failure"}},
        }
        atomic_json(manifest, paths["manifest"])
        paths["failure"].unlink(missing_ok=True)
        if not validate_complete_site(paths, deep=True):
            raise RuntimeError("Deep validation failed after atomic site write")
        return {"status": "complete", "subject": candidate.subject,
                "run": candidate.run, "site": candidate.site,
                "n_modes": len(modes), "n_qc_passed": len(qc_passed)}
    except Exception as exc:
        paths["manifest"].unlink(missing_ok=True)
        failure = {"schema_version": SCHEMA_VERSION, "status": "failed",
                   "created_utc": utc_now(), "subject": candidate.subject,
                   "run": candidate.run, "site": candidate.site,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        atomic_json(failure, paths["failure"])
        return failure


def process_run_group(items: list[SiteCandidate], output: Path, *, routes: dict[str, int],
                      rank: int, resume: bool, site_jobs: int = 1) -> list[dict[str, Any]]:
    pending = [item for item in items
               if not (resume and validate_terminal_site(site_paths(output, item, routes)))]
    results = []
    for item in items:
        if item not in pending:
            status = json.loads(site_paths(output, item, routes)["manifest"].read_text(
                encoding="utf-8"))["status"]
            results.append({"status": f"skipped_{status}", "subject": item.subject,
                            "run": item.run, "site": item.site})
    if not pending:
        return results
    try:
        run = load_run(Path(pending[0].run_path))
    except Exception as exc:
        for item in pending:
            paths = site_paths(output, item, routes)
            failure = {"schema_version": SCHEMA_VERSION, "status": "failed",
                       "created_utc": utc_now(), "subject": item.subject,
                       "run": item.run, "site": item.site,
                       "error": f"run_load: {type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            atomic_json(failure, paths["failure"])
            results.append(failure)
        return results
    def execute(item: SiteCandidate) -> dict[str, Any]:
        print(f"[site] {item.subject} {item.run} {item.site}", flush=True)
        result = process_site(item, run, output, routes=routes, rank=rank,
                              resume=resume)
        print(f"  -> {item.site}: {result['status']}", flush=True)
        return result

    # Keep only one waveform run resident at a time.  Site fits share that
    # read-only array through threads; BLAS is pinned to one thread above, so
    # --n-jobs can use many cores without loading 64 multi-GB runs at once.
    if site_jobs <= 1 or len(pending) == 1:
        results.extend(execute(item) for item in pending)
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(site_jobs, len(pending))) as executor:
            results.extend(executor.map(execute, pending))
    return results


def aggregate_site_tables(output: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((output / "per_site").glob("*.csv"))]
    sites = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_csv(sites, output / "sites.csv")
    if not sites.empty:
        for subject, frame in sites.groupby("subject", sort=True):
            atomic_csv(frame.reset_index(drop=True),
                       output / "per_subject" / f"{safe_name(subject)}.csv")
    return sites


def validate_output(output: Path, deep: bool = True) -> dict[str, Any]:
    manifests = sorted((output / "manifests").glob("*.json"))
    failures = sorted((output / "failures").glob("*.json"))
    invalid = []
    threshold_examples = []
    status_counts: dict[str, int] = defaultdict(int)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(manifest.get("status", "unknown"))
        status_counts[status] += 1
        paths = {key: output / value for key, value in manifest.get("files", {}).items()}
        paths["manifest"] = manifest_path
        paths["failure"] = output / "failures" / manifest_path.name
        if not validate_terminal_site(paths, deep=deep):
            invalid.append(manifest_path.name)
            continue
        if status == "complete" and deep and len(threshold_examples) < 2:
            residual_columns = [f"heldout_{i}_residual" for i in range(9)]
            modes = pd.read_csv(paths["mode"], usecols=residual_columns)
            residual = modes[residual_columns].to_numpy(float)
            threshold_examples.append({
                "site_manifest": manifest_path.name,
                "n_modes": len(modes),
                "mean_support_fraction_at_0.05": float(np.mean(residual <= 0.05)),
                "mean_support_fraction_at_0.10": float(np.mean(residual <= 0.10)),
                "mean_support_fraction_at_0.20": float(np.mean(residual <= 0.20)),
            })
    return {"schema_version": SCHEMA_VERSION,
            "n_complete": status_counts.get("complete", 0),
            "n_qc_excluded": status_counts.get("qc_excluded", 0),
            "n_failures": len(failures), "invalid_manifests": invalid,
            "posthoc_threshold_examples": threshold_examples}


def resolved_config(args: argparse.Namespace, routes: dict[str, int]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
        "selection": {
            "sampling_frequency_hz": config.SAMPLING_FREQUENCY_HZ, "current_a": config.CURRENT_A,
            "pulse_width_s": config.PULSE_WIDTH_S, "stimulation_type": config.STIMULATION_TYPE,
            "both_stim_contacts_type": "ECOG", "all_actual_stimulation_min_isi_s": config.MIN_ISI_S,
            "isi_clock": "all timestamped raw events.tsv rows (audited conservative clock)",
            "candidate_min_trials_within_run": config.N_TRIALS,
            "subject_site_run_selection": "max candidate count, canonical run ID tie-break",
            "selected_trials": config.N_TRIALS, "trial_subsample": "evenly spaced after QC",
        },
        "recording_channels": {
            "initial": "type=ECOG and status=good", "exclude_own_contacts": True,
            "recurrent_hard_fraction": config.RECURRENT_CHANNEL_FRACTION,
            "recurrent_hard_min_trials": config.RECURRENT_CHANNEL_MIN_TRIALS,
            "minimum_final_channels": config.MIN_CHANNELS,
        },
        "raw_epoch_qc": config.RAW_QC, "windows_ms": config.WINDOWS_MS,
        "routes": routes, "requested_svd_rank": args.rank,
        "rank_condition_floor": config.RANK_CONDITION_FLOOR,
        "preprocessing": {"resample": False, "filter": False, "notch": False,
                          "car": False, "baseline": False,
                          "within_window_channel_mean_removal": True},
        "mode_storage_filters": {"stable_only": False, "fmax": None,
                                 "positive_frequency_only": False,
                                 "residual_threshold": None,
                                 "support_threshold": None},
        "initial_modal_amplitude_ms": config.INITIAL_MODAL_AMPLITUDE_MS,
        "cli": vars(args), "provenance": provenance(SOURCES),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, help="Local root of OpenNeuro ds004080; alternatively set PARTIAL_OBSERVATION_BIDS_ROOT")
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "outputs/ecog")
    parser.add_argument("--analysis-id", default=None,
                        help="Output folder name (default: primary300_rank<RANK>)")
    parser.add_argument("--rank", type=int, default=config.REQUESTED_RANK,
                        help=f"SVD truncation rank (primary {config.REQUESTED_RANK}; "
                             f"Fig. S5 uses {config.ROBUSTNESS_RANK})")
    parser.add_argument("--routes", nargs="+", choices=list(config.ROUTE_DELAYS),
                        default=list(config.ROUTE_DELAYS),
                        help="Delay depths to fit (default: all)")
    parser.add_argument("--subject", action="append",
                        help="Repeat to select subjects; default is the full cohort")
    parser.add_argument("--site", action="append", help="Optional exact site filter")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate candidates and write audit/config without loading waveforms")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel sites within one resident run (maximum 64 recommended)")
    parser.add_argument("--strict-expected-counts", action="store_true",
                        help="Require 1788 sites/50 subjects (full unfiltered cohort only)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bids_root is not None:
        set_bids_root(args.bids_root)
    routes = {route: config.ROUTE_DELAYS[route] for route in args.routes}
    analysis_id = args.analysis_id or f"primary300_rank{args.rank}"
    output = args.output_root.resolve() / safe_name(analysis_id)
    output.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        report = validate_output(output, deep=True)
        atomic_json(report, output / "validation.json")
        print(json.dumps(report, indent=2))
        return int(bool(report["invalid_manifests"]))

    atomic_json(resolved_config(args, routes), output / "config.json")
    selected, audit = enumerate_cohort(
        subject_filter=set(args.subject) if args.subject else None,
        max_subjects=args.max_subjects,
        site_filter=set(args.site) if args.site else None,
    )
    atomic_csv(audit, output / "candidate_runs.csv")
    selected_subjects = len({item.subject for item in selected})
    summary = {"n_selected_subject_site": len(selected),
               "n_selected_subjects": selected_subjects,
               "n_candidate_run_sites_before_dedup": len(audit)}
    atomic_json(summary, output / "candidate_summary.json")
    print(json.dumps(summary, indent=2), flush=True)
    full_unfiltered = not args.subject and args.max_subjects is None and not args.site
    if args.strict_expected_counts and full_unfiltered:
        if (len(selected), selected_subjects) != (config.EXPECTED_CANDIDATE_SITES,
                                                  config.EXPECTED_CANDIDATE_SUBJECTS):
            raise RuntimeError(
                f"Candidate count mismatch: got {len(selected)} sites/{selected_subjects} "
                f"subjects; expected {config.EXPECTED_CANDIDATE_SITES}/{config.EXPECTED_CANDIDATE_SUBJECTS}")
    if args.dry_run:
        return 0

    grouped: dict[tuple[str, str], list[SiteCandidate]] = defaultdict(list)
    for item in selected:
        grouped[(item.subject, item.run_path)].append(item)
    run_groups = list(grouped.values())
    if args.n_jobs < 1 or args.n_jobs > 64:
        raise ValueError("--n-jobs must be in 1..64")
    nested = [process_run_group(group, output, routes=routes, rank=args.rank,
                                resume=not args.no_resume, site_jobs=args.n_jobs)
              for group in run_groups]
    results = [item for group in nested for item in group]
    aggregate_site_tables(output)
    validation = validate_output(output, deep=True)
    final = {"created_utc": utc_now(), "candidate_summary": summary,
             "status_counts": pd.Series([item["status"] for item in results]).value_counts().to_dict(),
             "validation": validation}
    atomic_json(final, output / "batch_manifest.json")
    print(json.dumps(final, indent=2), flush=True)
    return int(bool(validation["invalid_manifests"]))


if __name__ == "__main__":
    raise SystemExit(main())
