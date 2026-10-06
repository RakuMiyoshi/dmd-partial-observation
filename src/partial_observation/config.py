"""Primary analysis specification of the ECoG cohort (Methods, main text).

This module is the single source of every fixed parameter used by the ECoG
pipeline in ``scripts/ecog/`` and by the ECoG figure scripts. Nothing here is
tuned per participant or per site.
"""

# ----------------------------------------------------------------------
# Stimulation protocol and cohort (OpenNeuro ds004080)
# ----------------------------------------------------------------------
SAMPLING_FREQUENCY_HZ = 2048.0      # runs recorded at other rates are not used
CURRENT_A = 0.008
PULSE_WIDTH_S = 0.001
STIMULATION_TYPE = "monophasic"
MIN_ISI_S = 4.75                    # to every other stimulation, before and after
N_TRIALS = 10                       # trials analysed per stimulation site
MIN_CHANNELS = 4                    # recording channels required after exclusion

# Expected cohort size after protocol selection, before waveform QC.
EXPECTED_CANDIDATE_SITES = 1788
EXPECTED_CANDIDATE_SUBJECTS = 50

# ----------------------------------------------------------------------
# Raw-epoch quality control (amplifier clipping and persistent offsets)
# ----------------------------------------------------------------------
RAW_QC = {
    "pre_ms": (-310.0, -10.0),
    "post_scan_ms": (10.0, 900.0),
    "scale_floor_uv": 1.0,
    "plateau_eps_uv": 1.0,
    "plateau_warn_z": 10.0,
    "plateau_hard_z": 20.0,
    "plateau_warn_ms": 20.0,
    "plateau_hard_ms": 50.0,
    "dc_window_ms": (300.0, 900.0),
    "dc_warn_z": 20.0,
    "dc_hard_uv": 1000.0,
    "dc_absolute_hard_uv": 2000.0,
    "dc_persist_ms": 300.0,
}
# A channel failing the hard raw-epoch check in at least this fraction of the
# candidate trials (and in at least RECURRENT_CHANNEL_MIN_TRIALS trials) is
# excluded from every trial of the site.
RECURRENT_CHANNEL_FRACTION = 0.50
RECURRENT_CHANNEL_MIN_TRIALS = 2
# Window-level check: a window is rejected only for non-finite or flat channels.
FINITE_FRACTION_MIN = 0.999
FLAT_STD_UV = 1e-10

# ----------------------------------------------------------------------
# Hankel DMD
# ----------------------------------------------------------------------
WINDOWS_MS = {"pre": (-310.0, -10.0), "early": (10.0, 310.0)}
WINDOW_LABELS = {"pre": "pre-stimulus", "early": "post-stimulus"}
ROUTE_DELAYS = {                    # delay depth m of each estimator
    "standard": 1,
    "hankel_d5": 5,
    "hankel_d10": 10,
    "hankel_d20": 20,
    "hankel_d50": 50,
}
PRIMARY_ROUTES = ("standard", "hankel_d50")   # compared in the main text
REQUESTED_RANK = 50                 # SVD truncation rank of the Hankel matrix
ROBUSTNESS_RANK = 30                # sensitivity analysis (Fig. S5)
RANK_CONDITION_FLOOR = 1e-4         # keep singular values >= floor * s_max
INITIAL_MODAL_AMPLITUDE_MS = 20.0

# ----------------------------------------------------------------------
# Validation of eigenvalue estimates
# ----------------------------------------------------------------------
MAX_FREQUENCY_HZ = 100.0            # candidates: stable, 0 <= f <= 100 Hz
RESIDUAL_THRESHOLD = 0.1            # one-step held-out residual
REQUIRED_HELD_OUT_TRIALS = 5        # ... met in at least 5 of the 9 held-out trials
N_HELD_OUT_TRIALS = N_TRIALS - 1
DECAY_HORIZONS_MS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)   # multi-step diagnostics
DECAY_VALIDATION_HORIZON_MS = 5.0
DECAY_VALIDATION_TOLERANCE = 0.05   # one-sided radial gap

# ----------------------------------------------------------------------
# Reporting thresholds (Fig. 3 and Figs. S3-S5)
# ----------------------------------------------------------------------
FAST_DECAY_PER_S = -10.0            # fast-decaying: decay rate <= -10 / s (tau <= 100 ms)
FREQUENCY_SPLIT_HZ = 8.0
