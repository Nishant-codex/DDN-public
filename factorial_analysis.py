"""
Factorial analysis of reservoir spectral structure.

For each (task, delay, decay_mode, decay_scope) configuration, compute
summary statistics of the Jacobian spectrum, then produce the effect
table that decomposes variance in spectral structure across the four
factors.


Usage
-----
results = factorial_analysis(eigvals_by_config)  # dict of (task, delay,
                                                 # decay_mode, decay_scope)
                                                 # -> eigvals ndarray
"""

import numpy as np
import pandas as pd
import itertools


FACTOR_ORDER = ["task", "delay", "decay_mode", "decay_scope"]
TASK_LEVELS = ["MG", "NARMA"]
DELAY_LEVELS = ["ESN", "DDN"]
DECAY_MODE_LEVELS = ["fixed", "dist"]
DECAY_SCOPE_LEVELS = ["per_cluster", "net_wide"]


def mode_timescales(eigvals, eps=1e-12):
    """Timescale of each mode; slower modes (|mu| -> 1) get larger tau.

    Returns tau, the mode time constants, in the same order as eigvals.
    Zero/negligible modes get tau = 0.
    """
    vals = np.asarray(eigvals, dtype=complex)
    mag = np.abs(vals)
    tau = np.zeros(mag.shape, dtype=float)
    valid = mag > eps
    tau[valid] = -1.0 / np.log(mag[valid])
    return tau


def spectrum_summary(eigvals):
    """Summary statistics for one spectrum."""
    vals = np.asarray(eigvals, dtype=complex)
    mag = np.abs(vals)
    im = np.abs(np.imag(vals))
    tau = mode_timescales(vals)
    tau_pos = tau[tau > 0]

    def pr(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n == 0 or (x ** 2).sum() == 0:
            return 0.0
        # PR is bounded by n for a nonnegative vector; clip for safety.
        return float(np.clip((x.sum() ** 2) / (x ** 2).sum(), 0.0, n))

    return {
        "rho": mag.max() if mag.size else 0.0,
        "frac_complex": float((im > 1e-6).mean()) if mag.size else 0.0,
        "pr_magnitude": pr(mag),
        "pr_imag": pr(im),
        "logtau_iqr": (
            float(np.percentile(np.log10(tau_pos), 75)
                  - np.percentile(np.log10(tau_pos), 25))
            if tau_pos.size else 0.0
        ),
        "logtau_range": (
            float(np.log10(tau_pos.max()) - np.log10(tau_pos.min()))
            if tau_pos.size else 0.0
        ),
    }


def factorial_analysis(eigvals_by_config):
    """Compute per-config summaries, then effect table for each factor.

    Parameters
    ----------
    eigvals_by_config : dict
        Keys must be tuples (task, delay, decay_mode, decay_scope)
        with task in {MG, NARMA}, delay in {ESN, DDN},
        decay_mode in {fixed, distributed}, decay_scope in
        {per_cluster, net_wide}. Values are complex eigenvalue arrays.


    Returns
    -------
    config_df : pd.DataFrame
        One row per configuration with spectrum_summary columns.
    effect_df : pd.DataFrame
        One row per (factor, metric) pair, with the marginal mean
        difference and the conditional effect within each cell.
    """
    rows = []
    for key in itertools.product(
        TASK_LEVELS, DELAY_LEVELS,
        DECAY_MODE_LEVELS, DECAY_SCOPE_LEVELS,
    ):
        if key not in eigvals_by_config:
            continue
        summary = spectrum_summary(eigvals_by_config[key])
        summary.update(dict(zip(FACTOR_ORDER, key)))
        rows.append(summary)

    config_df = pd.DataFrame(rows)

    if config_df.empty:
        return config_df, pd.DataFrame()

    metrics = [
        "rho", "frac_complex", "pr_magnitude",
        "pr_imag", "logtau_iqr", "logtau_range",
    ]

    effect_rows = []
    for factor in FACTOR_ORDER:
        level_cols = [c for c in FACTOR_ORDER if c != factor]
        # Marginal effect: mean of each metric within each level of `factor`.
        grouped = config_df.groupby(factor)[metrics].mean()
        # Variance decomposition: variance of the marginal means.
        marginal_var = grouped.var()
        total_var = config_df[metrics].var()

        # Within-cell effect: for each other-config combo, the
        # within-cell difference between the two levels of `factor`.
        # This isolates `factor` from the other three.
        within_cells = []
        for cell_vals in config_df.groupby(level_cols).groups.values():
            cell = config_df.loc[cell_vals]
            levels_in_cell = cell[factor].unique()
            if len(levels_in_cell) == 2:
                a = cell[cell[factor] == levels_in_cell[0]][metrics]
                b = cell[cell[factor] == levels_in_cell[1]][metrics]
                within_cells.append(
                    (b.mean() - a.mean()).to_dict()
                )

        within_effect = pd.DataFrame(within_cells) if within_cells else pd.DataFrame()

        for metric in metrics:
            effect_rows.append({
                "factor": factor,
                "metric": metric,
                "marginal_var_ratio": (
                    marginal_var[metric] / total_var[metric]
                    if total_var[metric] > 0 else 0.0
                ),
                "within_cell_mean_diff": (
                    within_effect[metric].mean()
                    if not within_effect.empty else 0.0
                ),
                "within_cell_std": (
                    within_effect[metric].std()
                    if not within_effect.empty else 0.0
                ),
            })

    effect_df = pd.DataFrame(effect_rows)
    return config_df, effect_df


def participation_ratio_summary(state_matrix):
    """PR of the reservoir states (activity dimension), for comparison."""
    s = np.asarray(state_matrix, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    norm2 = (s ** 2).sum(axis=1)
    if norm2.sum() == 0:
        return 0.0
    return (norm2.sum() ** 2) / (norm2 ** 2).sum()


if __name__ == "__main__":
    # --- Example: fill with your actual eigvals ---
    rng = np.random.default_rng(0)
    eigvals_by_config = {}
    for task, delay, mode, scope in itertools.product(
        TASK_LEVELS, DELAY_LEVELS,
        DECAY_MODE_LEVELS, DECAY_SCOPE_LEVELS,
    ):
        # Placeholder spectrum for demonstration; replace with real data.
        n = 200
        mag = rng.uniform(0.05, 0.95, n)
        phase = rng.uniform(-np.pi, np.pi, n)
        eigvals_by_config[(task, delay, mode, scope)] = (
            mag * np.exp(1j * phase)
        )

    config_df, effect_df = factorial_analysis(eigvals_by_config)
    print(config_df.to_string(index=False))
    print()
    print(effect_df.to_string(index=False))
