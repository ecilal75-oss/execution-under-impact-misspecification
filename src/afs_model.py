"""
AFS Model — Alfonsi, Fruth & Schied (2010)
Implementation based on Hey, Bouchaud, Mastromatteo, Muhle-Karbe & Webster (2023)
"The Cost of Misspecifying Price Impact"
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AFSParams:
    """
    Parameters of the AFS price impact model.

    c     : impact concavity (0 < c <= 1). c=0.5 = square-root law, c=1 = linear
    tau   : impact decay timescale (in days)
    lam   : impact scale lambda (normalized via ADV and volatility)
    sigma : asset volatility
    V     : average daily volume (ADV)
    """
    c: float = 0.48
    tau: float = 0.2
    lam: float = 1.0
    sigma: float = 1.0
    V: float = 1.0


def optimal_impact(alpha: float | np.ndarray,
                   mu_alpha: float | np.ndarray,
                   params: AFSParams) -> np.ndarray:
    """
    Optimal impact state I* from Hey et al. Eq (2.4).

    I*_t = 1/(1+c) * (alpha_t - tau * mu_alpha_t)

    Parameters
    ----------
    alpha     : alpha signal level at time t
    mu_alpha  : drift of alpha (decay rate), mu_alpha = alpha'_t
    params    : AFS model parameters

    Returns
    -------
    I_star : optimal impact state
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    mu_alpha = np.atleast_1d(np.asarray(mu_alpha, dtype=float))

    I_star = (alpha - params.tau * mu_alpha) / (1 + params.c)
    return I_star


def pnl_optimal(alpha: float, params: AFSParams, T: float, tau_grid=None) -> float:
    """
    Hey et al. Section 4.2, formule exacte pour U(J(c); c).
    Alpha constant, tau correctement spécifié.
    """
    c = params.c
    tau = params.tau
    alpha_sr = alpha  # sigma=1

    U = (alpha_sr ** (1 + 1/c)) * (c / (1 + c)) * (
        T / (tau * (1 + c)) ** (1/c) + 1
    )
    return U


def pnl_misspecified(alpha: float,
                     params_true: AFSParams,
                     c_hat: float,
                     T: float) -> float:
    c = params_true.c
    tau = params_true.tau
    alpha_sr = alpha

    term1 = (alpha_sr ** (1 + 1/c_hat)) * (
        T / (tau * (1 + c_hat) ** (1/c_hat)) + 1
    )

    # g(c)/g(c_hat) = 1 quand g=1
    term2 = (alpha_sr ** ((1 + c)/c_hat)) * (
        T / (tau * (1 + c_hat) ** ((1+c)/c_hat)) + 1/(1+c)
    )

    return term1 - term2


def profit_ratio_concavity(alpha: float,
                           params_true: AFSParams,
                           c_hat_grid: np.ndarray,
                           T: float) -> np.ndarray:
    """
    Profit ratio U(J(c_hat); c) / U(J(c); c) across misspecified concavities.
    Reproduces Figure 4 (right panel) of Hey et al.

    Parameters
    ----------
    alpha        : constant alpha signal
    params_true  : true AFS parameters
    c_hat_grid   : array of misspecified concavity values to evaluate
    T            : trading horizon

    Returns
    -------
    ratios : profit ratio for each c_hat in c_hat_grid
    """
    U_opt = pnl_optimal(alpha, params_true, T, tau_grid=None)

    ratios = np.array([
        pnl_misspecified(alpha, params_true, c_hat, T) / U_opt
        for c_hat in c_hat_grid
    ])

    return ratios


def _prefactor_g(c: float, tau: float) -> float:
    """
    Normalized prefactor g(c, tau) from Hey et al. Eq (2.6).
    Placeholder — in practice calibrated from data.
    For simulation purposes we use g=1 as a normalization baseline.
    """
    # En pratique calibré depuis les données (Section 3 de Hey et al.)
    # Pour la reproduction des figures on normalise à 1
    return 1.0