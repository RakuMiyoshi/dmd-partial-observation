"""Fixed color palette used by the released simulation figures."""

# Mode groups: persistent (rho ~ 1) vs fast-decaying oscillators.
GROUP_COLORS: dict[str, str] = {
    "persistent": "#9467bd",
    "fast-decay": "#e08214",
}

# Analysis windows: pre-stimulus baseline and early post-stimulus response.
WINDOW_COLORS: dict[str, str] = {
    "pre": "#7f7f7f",
    "early": "#d62728",
}

# Window display names used by the ECoG plotting helpers.
WINDOW_LABELS: dict[str, str] = {
    "pre": "passive",
    "early": "perturbed",
    "late": "late",
}
