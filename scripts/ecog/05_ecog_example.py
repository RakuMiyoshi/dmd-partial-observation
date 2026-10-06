"""Step 5: export the recording example of Fig. 3A-B.

For one stimulation site, the display contacts are the ones with the largest
median evoked response over the ten selected trials, and the displayed trial is
the selected trial closest to that median response. The trace is a single raw
trial without filtering or averaging. The electrode positions are stored with
it, so the figure renders without raw data; the fsaverage surface itself is
fetched by the plotting code and not stored.

Writes ``fig3/ecog_example.npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from partial_observation.ccep_data import load_electrodes, load_run, set_bids_root
from partial_observation.cohort import enumerate_cohort, prepare_site
from partial_observation.koopman import slice_trial

SUBJECT = "sub-ccepAgeUMCU28"
SITE = "F28-F29"
N_DISPLAY_CHANNELS = 14
EPOCH_S = .35                         # displayed span before and after stimulation
PRE_MS = (-310, -10)                  # reference level for each contact
RESPONSE_MS = (15, 150)               # window used to rank contacts and trials


def load_example(subject: str, site: str) -> dict[str, object]:
    """Select the display contacts and the displayed trial for one site."""
    selected_sites, _ = enumerate_cohort(subject_filter={subject}, site_filter={site},
                                         progress=False)
    candidate, = selected_sites
    run = load_run(Path(candidate.run_path))
    prepared = prepare_site(candidate, run)
    if len(prepared["selected"]) != 10:
        raise RuntimeError("The example site no longer has ten selected trials")
    final_names = list(prepared["final_names"])
    final_indices = np.asarray(prepared["final_indices"], dtype=int)

    trial_segments, trial_times = [], []
    for item in prepared["selected"]:
        segment, trial_time = slice_trial(run["data"], int(item["onset_sample"]), run["sf"],
                                          pre_s=EPOCH_S, post_s=EPOCH_S)
        trial_segments.append(segment[final_indices])
        trial_times.append(trial_time)
    if not all(np.array_equal(trial_times[0], t) for t in trial_times[1:]):
        raise RuntimeError("Selected trials do not share a common time grid")
    time_ms = trial_times[0]
    all_trials = np.stack(trial_segments)
    pre = (time_ms >= PRE_MS[0]) & (time_ms < PRE_MS[1])
    response = (time_ms >= RESPONSE_MS[0]) & (time_ms <= RESPONSE_MS[1])
    centred = all_trials - np.nanmedian(all_trials[:, :, pre], axis=2)[:, :, None]
    median_response = np.nanmedian(centred, axis=0)
    response_score = np.nanpercentile(np.abs(median_response[:, response]), 95, axis=1)
    positions = np.argsort(response_score)[-N_DISPLAY_CHANNELS:][::-1]
    # Rows from farthest (back) to nearest (front) the stimulation pair.
    electrodes = load_electrodes(candidate.subject, candidate.session)
    stim_centre = np.mean([electrodes[n]["xyz"] for n in candidate.contacts], axis=0)
    distance = np.asarray([np.linalg.norm(np.asarray(electrodes[final_names[int(i)]]["xyz"])
                                          - stim_centre) for i in positions])
    positions = positions[np.argsort(-np.nan_to_num(distance, nan=np.inf))]

    # The selected trial closest to the median response on the displayed contacts.
    scale = np.nanpercentile(np.abs(median_response[positions][:, response]), 95, axis=1)
    residual = ((centred[:, positions][:, :, response]
                 - median_response[positions][None, :, response])
                / np.maximum(scale[None, :, None], 1.0))
    trial_position = int(np.nanargmin(np.nanmean(residual ** 2, axis=(1, 2))))
    selected = prepared["selected"][trial_position]

    recording_xyz = np.asarray([electrodes[name]["xyz"] for name in final_names])
    stim_xyz = np.asarray([electrodes[name]["xyz"] for name in candidate.contacts])
    finite = np.isfinite(recording_xyz).all(axis=1)
    return {
        "candidate": candidate, "time_ms": time_ms,
        "traces_uv": all_trials[trial_position, positions],
        "channel_names": [final_names[int(i)] for i in positions],
        "trial_position": trial_position, "selected": selected,
        "recording_xyz": recording_xyz[finite], "stim_xyz": stim_xyz,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, help="Local root of OpenNeuro ds004080")
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/reference_results/fig3/ecog_example.npz"))
    args = parser.parse_args()
    if args.bids_root is not None:
        set_bids_root(args.bids_root)
    example = load_example(args.subject, args.site)
    candidate = example["candidate"]
    hemi = "rh" if np.nanmedian(example["recording_xyz"][:, 0]) >= 0 else "lh"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        time_ms=example["time_ms"].astype(np.float64),
        traces_uv=example["traces_uv"].astype(np.float32),
        channel_names=np.asarray(example["channel_names"]),
        recording_xyz=example["recording_xyz"].astype(np.float64),
        stim_xyz=example["stim_xyz"].astype(np.float64),
        meta=np.asarray(json.dumps({
            "subject": candidate.subject, "session": candidate.session,
            "run": candidate.run, "site": candidate.site,
            "hemisphere": hemi, "event_index": int(example["selected"]["event_index"]),
            "onset_sample": int(example["selected"]["onset_sample"]),
            "trial_position_within_selected_ten": example["trial_position"],
            "n_display_channels": N_DISPLAY_CHANNELS,
            "display_rows": "back (first) to front (last); front is nearest "
                            "the stimulation pair",
            "trace_processing": "single raw unaveraged trial; no filtering",
            "generator": "scripts/ecog/05_ecog_example.py",
        })),
    )
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
