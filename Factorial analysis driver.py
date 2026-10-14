"""
Driver: fill eigvals_by_config from your actual experiment and run
the factorial analysis.

Fill in the `load_spectra` function for however your spectra are stored
(a directory of .npz files, a single pickle, or recomputed live from
the Jacobian function). The key format is (task, delay, decay_mode,
decay_scope) and the value is a complex eigenvalue array.
"""

import itertools
import numpy as np
from factorial_analysis import (
    factorial_analysis,
    spectrum_summary,
    TASK_LEVELS, DELAY_LEVELS,
    DECAY_MODE_LEVELS, DECAY_SCOPE_LEVELS,
)


def load_spectra():
    """Return {(task, delay, decay_mode, decay_scope): complex eigvals}.

    Replace this with your actual loader. Common patterns:
    - From a directory of .npz files, one per configuration.
    - Recompute live from your Jacobian function and block_companion.
    """
    eigvals_by_config = {}
    # Example using a directory of npz files:
    # import glob, os
    # for path in glob.glob("spectra/*.npz"):
    #     task, delay, mode, scope = os.path.basename(path).removesuffix(".npz").split("__")
    #     eigvals_by_config[(task, delay, mode, scope)] = np.load(path)["eigvals"]

    # Example recomputing live (uses your functions):
    # for (task, delay, mode, scope) in itertools.product(
    #         TASK_LEVELS, DELAY_LEVELS, DECAY_MODE_LEVELS, DECAY_SCOPE_LEVELS):
    #     net, x_star = build_network_for(task, delay, mode, scope)
    #     J_blocks, g_star = delayed_esn_jacobian_blocks(
    #         net.W_by_delay, net.decay, x_star,
    #         activation="tanh" if task == "NARMA" else "sigmoid",
    #     )
    #     eigvals_by_config[(task, delay, mode, scope)] = delayed_esn_modes(J_blocks)

    return eigvals_by_config


def main():
    eigvals_by_config = load_spectra()
    if not eigvals_by_config:
        raise SystemExit(
            "load_spectra returned nothing. Fill in the loader first."
        )

    config_df, effect_df = factorial_analysis(eigvals_by_config)

    print("=== Per-configuration summaries ===")
    print(config_df.to_string(index=False))
    print()
    print("=== Effect table (variance decomposition) ===")
    print(effect_df.to_string(index=False))

    config_df.to_csv("factorial_config_summary.csv", index=False)
    effect_df.to_csv("factorial_effect_table.csv", index=False)
    print("\nWrote factorial_config_summary.csv and factorial_effect_table.csv")


if __name__ == "__main__":
    main()
