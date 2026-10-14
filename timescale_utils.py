import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import os 
from simulator import NetworkSimulator

from reservoirpy.datasets import mackey_glass

import numpy as np

def settle_to_fixed_point(
    model,
    input_constant,
    max_steps=10000,
    tol=1e-12,              # CHANGED: was 1e-9, now 1e-12
    consecutive_required=50,
    verbose=False,
):
    """
    Run under constant input and return a candidate fixed point.
    """
    state = np.array(model.A, dtype=float, copy=True)
    consecutive = 0

    for step in range(max_steps):
        model.update_step(input_constant)
        state_next = np.array(model.A, dtype=float, copy=True)

        # For a delayed network: check the full history too.
        if state_next.ndim == 2:
            # DDN, state is (N+1, D): all columns must match column 0.
            history_spread = np.linalg.norm(state_next - state_next[:, [0]])
        else:
            history_spread = 0.0

        update_error = np.linalg.norm(state_next - state) + history_spread
        state = state_next

        if update_error < tol:
            consecutive += 1
        else:
            consecutive = 0

        if verbose and step % 1000 == 0:
            print(
                f"Step {step + 1}: update error = {update_error:.3e}, "
                f"consecutive = {consecutive}"
            )

        if consecutive >= consecutive_required:
            if verbose:
                print(
                    f"Converged to fixed point after {step + 1} steps "
                    f"(consecutive = {consecutive})."
                )
            return state.copy(), True
    return state.copy(), False

def _newton_refine_esn(
    model,
    x0,
    input_constant,
    max_iter=50,
    tol=1e-14,
    verbose=False,
):
    """Newton refinement for a non-delayed network.

    Uses the Jacobian esn_jacobian (which you already have) to take
    Newton steps: x <- x - J^{-1} (F(x) - x).
    """
    from scipy.linalg import solve

    x = np.array(x0, dtype=float, copy=True)
    success = False
    residual = np.inf

    for i in range(max_iter):
        # Evaluate F(x) - x, the fixed-point residual.
        model.A = x.copy()
        model.update_step(input_constant)
        Fx = np.array(model.A, dtype=float, copy=True)
        residual_vec = Fx - x
        residual = np.linalg.norm(residual_vec)

        if verbose:
            print(f"Newton iter {i + 1}: residual = {residual:.3e}")

        if residual < tol:
            success = True
            break

        # Compute the Jacobian at the current point.
        # NOTE: esn_jacobian needs `net`, `x_star`, and `activation`.
        # You may need to adapt this call to your model's interface.
        try:
            J, _ = esn_jacobian(
                model, x,
                activation=model.activation_func.__name__.split('_')[0],  # "sigmoid" or "tanh"
            )
        except (AttributeError, NameError):
            # Fall back to a finite-difference Jacobian if esn_jacobian
            # isn't available or doesn't match this model's interface.
            J = _finite_difference_jacobian(model, x, input_constant)

        # Newton step: x <- x - J^{-1} (F(x) - x).
        try:
            dx = solve(J, -residual_vec)
        except np.linalg.LinAlgError:
            # Singular Jacobian; use least squares.
            dx, *_ = np.linalg.lstsq(J, -residual_vec, rcond=None)

        x = x + dx

    return success, x, residual

def _finite_difference_jacobian(model, x, input_constant, eps=1e-7):
    """Finite-difference Jacobian of F(x) - x, used as a fallback."""
    N = x.size
    J = np.zeros((N, N))

    model.A = x.copy()
    model.update_step(input_constant)
    F0 = np.array(model.A, dtype=float, copy=True)

    for j in range(N):
        x_pert = x.copy()
        x_pert[j] += eps
        model.A = x_pert
        model.update_step(input_constant)
        F1 = np.array(model.A, dtype=float, copy=True)
        J[:, j] = (F1 - F0) / eps

    return J - np.eye(N)

def delayed_esn_jacobian_blocks(
    W_by_delay,
    alpha,
    x_star,
    activation="tanh",
):
    """
    Compute state-delay Jacobian blocks J_0,...,J_D
    at a fixed point of a delayed leaky ESN.


    Parameters
    ----------
    W_by_delay : list of ndarray
        W_by_delay[d] is W_d, shape (N, N), for d = 0,...,D.
        W_d[i, j] must mean connection source j -> target i.
    alpha : float or array-like, shape (N,)
        Scalar common leak rate, or per-neuron leak rates.
    x_star : array-like, shape (N,) or (N, 1)
        Reservoir fixed point (post-activation state).
    activation : str or callable, optional
        "sigmoid"  : expects x_star in [0, 1] (sigmoid output).
        "tanh"     : expects x_star in [-1, 1] (tanh output). Default.
        callable  : derivative function evaluated on x_star,
                    must return a length-N vector. Use this for any
                    other activation.


    Returns
    -------
    J_blocks : list of ndarray
        [J_0,...,J_D], each of shape (N, N).
    g_star : ndarray, shape (N,)
        Activation derivative at the fixed point.
    """
    W = [np.asarray(Wd, dtype=float) for Wd in W_by_delay]
    N = W[0].shape[0]
    D = len(W) - 1


    if any(Wd.shape != (N, N) for Wd in W):
        raise ValueError("Every W_by_delay[d] must have shape (N, N).")


    x_star = np.asarray(x_star, dtype=float).reshape(-1)
    if x_star.size != N:
        raise ValueError("x_star must contain exactly N entries.")


    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 0:
        alpha = np.full(N, float(alpha))
    else:
        alpha = alpha.reshape(-1)


    if alpha.size != N:
        raise ValueError("alpha must be scalar or a length-N vector.")
    if np.any((alpha <= 0.0) | (alpha > 1.0)):
        raise ValueError("alpha must be in (0, 1].")


    # --- Activation derivative at the fixed point ---
    # The state stored in the reservoir is the POST-activation value,
    # so for the standard activations the derivative can be written
    # directly in terms of x_star. Any other activation can be passed
    # as a callable derivative.
    if activation == "sigmoid":
        if np.any((x_star < 0.0) | (x_star > 1.0)):
            raise ValueError(
                "activation='sigmoid' but x_star has entries outside [0, 1]. "
                "Your network probably uses tanh -- pass activation='tanh'."
            )
        g_star = x_star * (1.0 - x_star)
    elif activation == "tanh":
        if np.any(np.abs(x_star) > 1.0):
            raise ValueError(
                "activation='tanh' but x_star has entries outside [-1, 1]. "
                "Your network probably uses sigmoid -- pass activation='sigmoid'."
            )
        g_star = 1.0 - x_star**2
    elif callable(activation):
        g_star = np.asarray(activation(x_star), dtype=float).reshape(-1)
        if g_star.size != N:
            raise ValueError("activation callable must return length-N vector.")
    else:
        raise NotImplementedError(
            "Use 'sigmoid', 'tanh', or pass a callable derivative."
        )


    # --- Delay Jacobian blocks ---
    # For each row i: J_d[i, :] = alpha_i * g_star[i] * W_d[i, :],
    # with the leak term (1 - alpha_i) on the diagonal only for d = 0.
    row_gain = alpha * g_star


    J_blocks = []
    J0 = np.diag(1.0 - alpha) + row_gain[:, None] * W[0]
    J_blocks.append(J0)


    for d in range(1, D + 1):
        Jd = row_gain[:, None] * W[d]
        J_blocks.append(Jd)


    return J_blocks, g_star

def augmented_jacobian(J_blocks):
    """
    Construct the companion/augmented Jacobian.

    J_blocks = [J_0, ..., J_D], each shape (N, N).

    Returns
    -------
    J_aug : ndarray, shape (N*(D+1), N*(D+1)).
    """
    J_blocks = [np.asarray(J, dtype=float) for J in J_blocks]

    N = J_blocks[0].shape[0]
    D = len(J_blocks) - 1

    if any(J.shape != (N, N) for J in J_blocks):
        raise ValueError("All Jacobian blocks must have identical (N, N) shape.")

    J_aug = np.zeros((N * (D + 1), N * (D + 1)), dtype=float)

    # Top block row: state update.
    for d, Jd in enumerate(J_blocks):
        J_aug[:N, d*N:(d+1)*N] = Jd

    # Lower block rows: delay-history shift register.
    for d in range(1, D + 1):
        J_aug[d*N:(d+1)*N, (d-1)*N:d*N] = np.eye(N)

    return J_aug

def timescales_from_eigenvalues(net,J_aug_a, dt=1.0, zero_tol=1e-10):
    """
    Compute stable modal decay timescales from discrete-time eigenvalues.

    Returns a dictionary aligned with all input eigenvalues.
    Unstable, unit-circle, and near-zero modes get NaN timescales.
    """
    mu = np.linalg.eigvals(J_aug_a)          # one matrix only
    r = np.abs(mu)

    print("rho:", r.max(), " -> tau_max:", -net.dt/np.log(r.max()))
    print("top 10 |mu|:", np.sort(r)[-10:])

    meaningful = (r > 1e-6) & (r < 1 - 1e-12)
    tau = -net.dt/np.log(r[meaningful])

    return {
        "eigenvalues": mu,
        "magnitudes": r,
        "timescales": tau,
        "stable_mask": meaningful,
        "spectral_radius": r.max(),
    }

def plot_spectrum(mu, mu_full=None, zero_tol=1e-6, title="Augmented Jacobian spectrum"):
    r = np.abs(mu)
    rho = r.max()
    keep = r > zero_tol            # drop shift-register zeros for clarity

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- full view ---
    th = np.linspace(0, 2*np.pi, 400)
    ax[0].plot(np.cos(th), np.sin(th), "k--", lw=1, label="unit circle")
    ax[0].plot(rho*np.cos(th), rho*np.sin(th), "r:", lw=1,
               label=fr"$\rho={rho:.5f}$")

    ax[0].scatter(mu.real[keep], mu.imag[keep], s=18, alpha=0.8,
                  label=f"{keep.sum()} nonzero modes")
    n_zero = (~keep).sum()
    ax[0].scatter([0], [0], marker="x", c="gray", s=40,
                  label=f"{n_zero} near-zero modes")

    if mu_full is not None:
        marg = np.abs(mu_full) > 1 - 1e-9
        ax[0].scatter(mu_full.real[marg], mu_full.imag[marg], marker="s",
                      facecolors="none", edgecolors="orange", s=60,
                      label=f"{marg.sum()} marginal (frozen units)")

    ax[0].set_aspect("equal")
    ax[0].set_xlabel("Re($\\mu$)"); ax[0].set_ylabel("Im($\\mu$)")
    ax[0].set_title(title); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=0.3)

    # --- zoom near z = 1 (slow modes) ---
    ax[1].plot(np.cos(th), np.sin(th), "k--", lw=1)
    ax[1].scatter(mu.real[keep], mu.imag[keep], s=22)
    top = np.argsort(-r)[:5]
    for k in top:
        ax[1].annotate(f"{r[k]:.4f}", (mu[k].real, mu[k].imag),
                       textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1].set_xlim(0.85, 1.01); ax[1].set_ylim(-0.1, 0.1)
    ax[1].set_xlabel("Re($\\mu$)"); ax[1].set_ylabel("Im($\\mu$)")
    ax[1].set_title("Zoom near $\\mu=1$"); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig

def plot_tau(mu, dt, zero_tol=1e-6, tau_lims=None):
    r = np.abs(mu)
    sel = (r > zero_tol) & (r < 1 - 1e-12)
    tau = -dt / np.log(r[sel])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(np.sort(tau)[::-1], "o-", ms=3)
    ax[0].set_xlabel("mode rank (slowest first)")
    ax[0].set_ylabel(r"$\tau$ (time units)")
    ax[0].grid(alpha=0.3)

    ax[1].hist(np.log10(tau), bins=30)
    if tau_lims is not None:
        ax[1].set_xlim(*tau_lims)
    else:
        ax[1].set_xlim(np.log10(tau).min(), np.log10(tau).max())
    ax[1].set_xlabel(r"$\log_{10}\tau$")
    ax[1].set_ylabel("count")
    plt.tight_layout()
    return fig

def run_zero_input(net, n_steps,n_switch, v_const=0.0):
    """
    step_fn(x_hist, v) -> x_next, where x_hist is the (N, D+1) delay buffer,
    column 0 = current state. Use your own update function.
    n_steps = number of updates to run, including warmup.
    n_switch = number of updates to run after switching input to v_const.
    """
    warmup  = 200

    narma_input = mackey_glass(n_steps, seed=0)
    narma_input[n_switch:,0]= narma_input[n_switch:,0]*v_const
    INPUT = narma_input
    sim = NetworkSimulator(net)
    sim.warmup(narma_input[:warmup])
    traj = sim.get_network_data(INPUT[warmup:]) # skip warmup, skip input column
    return np.array(traj)          # shape (n_steps+1, N)

def plot_relaxation(
    traj,
    x_star,
    rho,
    dt=1.0,
    n_units=10,
    seed=0,
    switch_step=None
):
    """
    Parameters
    ----------
    traj : ndarray, shape (N, T)
        Reservoir trajectory. Each column is the N-neuron state at one time.
    x_star : ndarray, shape (N,) or (N, 1)
        Fixed-point state of the physical neurons.
    rho : float
        Spectral radius of the active-unit augmented Jacobian.
    dt : float
        Time per reservoir update.
    n_units : int
        Number of individual neuron traces to display.
    """
    traj = np.asarray(traj, dtype=float)
    x_star = np.asarray(x_star, dtype=float).reshape(-1, 1)

    N, T = traj.shape

    if x_star.shape[0] != N:
        raise ValueError(
            f"x_star has {x_star.shape[0]} units, but traj has {N}."
        )

    delta_x = traj - x_star                  # (N, T)
    dist = np.linalg.norm(delta_x, axis=0)   # (T,)
    t = np.arange(T) * dt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Individual physical-neuron trajectories
    rng = np.random.default_rng(seed)
    units = rng.choice(N, size=min(n_units, N), replace=False)

    for i in units:
        ax[0].plot(t, traj[i, :], lw=0.8)

    ax[0].set_xlabel("time")
    ax[0].set_ylabel("reservoir state")
    ax[0].set_title("Selected reservoir units")

    # 2. Deviation from fixed point
    epsilon = np.finfo(float).tiny
    ax[1].semilogy(t, np.maximum(dist, epsilon), label="observed")

    predicted = dist[0] * rho ** np.arange(T) #
    ax[1].semilogy(
        t,
        np.maximum(predicted, epsilon),
        "r--",
        lw=1.5,
        label=rf"$\rho^n$, $\rho={rho:.6f}$",
    )

    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\|\mathbf{x}_n-\mathbf{x}^{\ast}\|_2$")
    ax[1].set_title("Relaxation toward fixed point")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # 3. PCA: use one shared projection for the entire trajectory
    X = delta_x.T
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc = X @ Vt[:2].T

    if switch_step is None:
        ax[2].plot(
            pc[:, 0], pc[:, 1],
            color="tab:blue", lw=0.5, alpha=0.7,
            label="trajectory",
        )
    else:
        # Include the switch point in both segments to keep them connected.
        ax[2].plot(
            pc[:switch_step + 1, 0],
            pc[:switch_step + 1, 1],
            color="tab:blue", lw=0.7, alpha=0.7, 
            label="MG input",
        )

        ax[2].plot(
            pc[switch_step:, 0],
            pc[switch_step:, 1],
            color="tab:orange", lw=0.7, alpha=0.9, linestyle="--",
            label="zero input",
        )

        ax[2].scatter(
            pc[switch_step, 0], pc[switch_step, 1],
            color="black", marker="X", s=50, zorder=5,
            label="input switch",
        )

    ax[2].scatter(
        pc[0, 0], pc[0, 1],
        color="green", s=35, zorder=5, label="start",
    )
    ax[2].scatter(
        pc[-1, 0], pc[-1, 1],
        color="red", s=35, zorder=5, label="end",
    )

    ax[2].set_xlabel("PC1")
    ax[2].set_ylabel("PC2")
    ax[2].set_title("Trajectory in PCA space")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    plt.tight_layout()

    return fig,

def activity_dimension(traj):
    X = np.asarray(traj, dtype=float)

    # Center each neuron's activity over the observation window.
    X = X - X.mean(axis=1, keepdims=True)

    # Squared singular values are proportional to covariance eigenvalues.
    s = np.linalg.svd(X, compute_uv=False)
    variance = s**2

    total = variance.sum()
    if total == 0:
        return {
            "participation_ratio": np.nan,
            "n_pc_95": 0,
        }

    participation_ratio = total**2 / np.sum(variance**2)

    cumulative = np.cumsum(variance) / total
    n_pc_95 = np.searchsorted(cumulative, 0.95) + 1

    return {
        "participation_ratio": participation_ratio,
        "n_pc_95": int(n_pc_95),
    }

def run_network_with_input(net, timesteps,v_const):
    
    narma_input = mackey_glass(timesteps, seed=0)
    INPUT = narma_input.T
    sim = NetworkSimulator(net)
    update = sim.get_network_data(INPUT.T*v_const)
    return update

def esn_jacobian(
    net,
    x_star,
    activation="sigmoid",
    active_only=False,
):
    """
    Jacobian of the non-delayed, zero-input update:

        x_next = (1 - alpha) * x
                 + alpha * f(W @ x + bias)

    where f is the activation you supply.


    Parameters
    ----------
    net : object
        Must expose .W (full reservoir matrix, including the input node
        if patched), .neurons_out (reservoir indices), .decay (leak
        rates for every unit in .W), .WBias (bias, optional), and
        .activation_func (the activation used to build the reservoir).
    x_star : array-like
        Fixed-point state, reservoir units only.
    activation : str or callable
        "sigmoid" : net.activation_func is the standard logistic
                    sigmoid, output in [0, 1].
        "tanh"    : net.activation_func is tanh, output in [-1, 1].
        callable : derivative function, called as f_prime(h) where h is
                   the PRE-activation vector. Use this for any other
                   activation.
    active_only : bool
        If True, keep only rows/columns of units with alpha > 0.


    Returns
    -------
    J : ndarray, shape (N, N)
    info : dict with the fixed-point residual, the activation
           derivative, and the indices of active units.
    """
    W = np.asarray(net.W[np.ix_(net.neurons_out, net.neurons_out)], dtype=float)
    x = np.asarray(x_star, dtype=float).reshape(-1)
    N = x.size


    if N == 0:
        raise ValueError("x_star must contain at least one reservoir state.")


    if W.shape != (N, N):
        raise ValueError(
            f"W has shape {W.shape}; expected {(N, N)}. "
            "Exclude the patched input node consistently from W and x_star."
        )


    a = np.asarray(net.decay, dtype=float)[net.neurons_out]


    # Bias: handle both "no bias attribute" and "bias stored as 2D".
    bias = getattr(net, "WBias", None)
    if bias is None:
        bias = np.zeros(N)
    else:
        bias = np.asarray(bias, dtype=float)
        if bias.ndim == 2:
            # e.g. shape (N+1, N) with the input node as the extra row.
            bias = bias[net.neurons_out]
        else:
            bias = bias[net.neurons_out].reshape(-1)
        if bias.size != N:
            raise ValueError(
                f"Bias has size {bias.size}; expected {N}. "
                "Check that net.neurons_out indexes the reservoir correctly."
            )


    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
        raise ValueError(
            "For this update, alpha must be finite and lie in [0, 1]."
        )


    # Pre-activation.
    h = W @ x + bias


    # --- Activation derivative ---
    # Two ways to supply it: by name (the post-activation state x
    # determines the derivative for the standard activations), or as
    # a callable acting on the PRE-activation h.
    if activation == "sigmoid":
        activation_out = np.asarray(
            net.activation_func(h), dtype=float
        ).reshape(-1)
        if activation_out.size != N:
            raise ValueError(
                "activation_func must return one value per reservoir unit."
            )
        if np.any((activation_out < 0.0) | (activation_out > 1.0)):
            raise ValueError(
                "activation='sigmoid' but activation_func output is outside "
                "[0, 1]. Your network probably uses tanh -- pass 'tanh'."
            )
        derivative = activation_out * (1.0 - activation_out)
    elif activation == "tanh":
        activation_out = np.asarray(
            net.activation_func(h), dtype=float
        ).reshape(-1)
        if activation_out.size != N:
            raise ValueError(
                "activation_func must return one value per reservoir unit."
            )
        if np.any(np.abs(activation_out) > 1.0):
            raise ValueError(
                "activation='tanh' but activation_func output is outside "
                "[-1, 1]. Your network probably uses sigmoid -- pass 'sigmoid'."
            )
        derivative = 1.0 - activation_out**2
    elif callable(activation):
        # A user-supplied derivative, evaluated on the PRE-activation.
        derivative = np.asarray(
            activation(h), dtype=float
        ).reshape(-1)
        if derivative.size != N:
            raise ValueError(
                "activation callable must return one value per reservoir unit."
            )
        activation_out = np.asarray(
            net.activation_func(h), dtype=float
        ).reshape(-1)
    else:
        raise NotImplementedError(
            "Use 'sigmoid', 'tanh', or pass a callable derivative."
        )


    J_full = (
        np.diag(1.0 - a)
        + (a * derivative)[:, None] * W
    )


    x_next = (1.0 - a) * x + a * activation_out
    residual = x_next - x


    active = np.flatnonzero(a > 0)


    J = (
        J_full[np.ix_(active, active)]
        if active_only
        else J_full
    )


    info = {
        "fixed_point_residual": np.linalg.norm(residual),
        "residual_per_unit": residual,
        "active_indices": active,
        "activation_derivative": derivative,
    }


    return J, info

def save_spectrum(eigvals, task, delay, decay_mode, decay_scope, out_dir="spectra"):
    """Save one configuration's eigenvalues to a .npz file."""
    os.makedirs(out_dir, exist_ok=True)
    key = f"{task}__{delay}__{decay_mode}__{decay_scope}"
    np.savez(
        os.path.join(out_dir, f"{key}.npz"),
        eigvals=np.asarray(eigvals, dtype=complex),
    )