"""
Monte Carlo Simulation Engine
Nutz, Webster & Zhao (2025) — Section 3

Simule les trajectoires optimales (X_t, Y_t, Z_t) sous la stratégie
q_t = f_t * X_t + g_t * Y_t + h_t * Z_t
"""

import numpy as np
from dataclasses import dataclass
from riccati import OWParams, solve_riccati


def simulate_optimal_strategy(params: OWParams,
                               n_paths: int = 1000,
                               n_steps: int = 20,
                               seed: int = 42) -> dict:
    """
    Simule n_paths trajectoires de la stratégie optimale.

    Parameters
    ----------
    params  : OWParams — paramètres du modèle
    n_paths : nombre de trajectoires Monte Carlo
    n_steps : nombre de chocs intraday (défaut 20, ~toutes les 20 min)
    seed    : graine aléatoire pour reproductibilité

    Returns
    -------
    dict avec les trajectoires moyennes et distributions des métriques
    """
    np.random.seed(seed)

    T = params.T
    dt = T / n_steps
    t_shocks = np.linspace(0, T, n_steps + 1)

    # Résout le système de Riccati (précomputation)
    sol = solve_riccati(params, n_points=1000)
    t_grid = sol['t']
    f_grid = sol['f']
    g_grid = sol['g']
    h_grid = sol['h']

    def interp_coeff(coeff, t):
        return np.interp(t, t_grid, coeff)

    # Initialisation
    # x0 = -z0 = -0.1 ADV (inventaire initial)
    z0 = 0.1
    x0 = -z0
    y0 = 0.0  # impact initial nul

    # Stockage des trajectoires
    all_X = np.zeros((n_paths, n_steps + 1))
    all_Y = np.zeros((n_paths, n_steps + 1))
    all_Z = np.zeros((n_paths, n_steps + 1))
    all_q = np.zeros((n_paths, n_steps))
    all_costs = np.zeros(n_paths)

    for p in range(n_paths):
        X = x0
        Y = y0
        Z = z0

        all_X[p, 0] = X
        all_Y[p, 0] = Y
        all_Z[p, 0] = Z

        cost = 0.0

        for i in range(n_steps):
            t = t_shocks[i]

            # Coefficients à l'instant t
            ft = interp_coeff(f_grid, t)
            gt = interp_coeff(g_grid, t)
            ht = interp_coeff(h_grid, t)

            # Vitesse de trading optimale
            q = ft * X + gt * Y + ht * Z

            # Choc sur le flux entrant (Gaussien iid)
            dZ = -params.theta * Z * dt + params.sigma * np.random.randn() * np.sqrt(dt)

            # Mise à jour des états
            X = X + q * dt - dZ
            Y = Y * (1 - params.beta * dt) + params.lam * q * dt
            Z = Z + dZ

            # Coût d'impact et de spread
            impact_cost = Y * q * dt
            spread_cost = 0.5 * params.eps * q**2 * dt
            cost += impact_cost + spread_cost

            all_X[p, i+1] = X
            all_Y[p, i+1] = Y
            all_Z[p, i+1] = Z
            all_q[p, i] = q

        all_costs[p] = cost

    # Métriques
    in_flow_tv = np.mean(np.sum(np.abs(np.diff(all_Z, axis=1)), axis=1))

    return {
        't': t_shocks,
        'X': all_X,
        'Y': all_Y,
        'Z': all_Z,
        'q': all_q,
        'costs': all_costs,
        'mean_cost': np.mean(all_costs),
        'in_flow_tv': in_flow_tv,
        'params': params
    }


def compute_internalization(results: dict) -> float:
    """
    Taux d'internalization — fraction du flux entrant nettée.
    Nutz et al. Eq (3.1).
    """
    Z = results['Z']
    q = results['q']
    dt = results['t'][1] - results['t'][0]

    in_flow_tv = np.mean(np.sum(np.abs(np.diff(Z, axis=1)), axis=1))
    out_flow_tv = np.mean(np.sum(np.abs(q) * dt, axis=1))

    internalization = 1 - out_flow_tv / in_flow_tv
    return internalization