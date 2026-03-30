"""
Riccati ODE Solver
Implémente le système d'EDO de Nutz, Webster & Zhao (2025)
"Unwinding Stochastic Order Flow: When to Warehouse Trades"

Le système résout les coefficients (A, B, C, D, E, F, K)
de la fonction valeur quadratique v(t, x, y, z).
La stratégie optimale est alors :
    q_t = f_t * X_t + g_t * Y_t + h_t * Z_t
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass


@dataclass
class OWParams:
    """
    Paramètres du modèle Obizhaeva-Wang généralisé.
    Valeurs par défaut calibrées depuis Nutz et al. Section 3.1.

    beta  : impact decay rate (1/jour). beta=8 → half-life ~40 minutes
    lam   : Kyle's lambda, impact par unité tradée
    eps   : spread cost parameter (coût instantané)
    theta : autocorrélation du flux entrant (OU parameter)
              theta < 0 → momentum
              theta = 0 → martingale (truth-telling)
              theta > 0 → mean-reversion
    sigma : volatilité du flux entrant
    T     : horizon de trading (1 jour)
    """
    beta: float = 8.0
    lam: float = 0.2
    eps: float = 0.01
    theta: float = 0.0
    sigma: float = 0.1
    T: float = 1.0


def riccati_rhs(t: float, P: np.ndarray, params: OWParams) -> np.ndarray:
    """
    Second membre du système de Riccati (2.20) de Nutz et al.
    P = [A, B, C, D, E, F, K]

    On résout en temps rebroussé : on part de T et on va vers 0.
    Les conditions terminales sont :
        A_T = lambda, B_T = -1, C_T = 1/lambda
        D_T = E_T = F_T = K_T = 0
    """
    A, B, C, D, E, F, K = P

    beta = params.beta
    lam = params.lam
    eps = params.eps
    theta = params.theta
    sigma = params.sigma

    # Termes intermédiaires
    MB = A + lam * B      # = -eps * f
    NB = B + lam * C      # = -eps * g
    HB = D + lam * E      # = -eps * h

    # Système de Riccati — Proposition 2.7 de Nutz et al.
    dA = eps**(-1) * MB**2
    dB = eps**(-1) * MB * NB + beta * B
    dC = eps**(-1) * NB**2 + 2 * beta * C - (1/lam) * (2 * beta)
    dD = eps**(-1) * MB * HB - theta * (A - D)
    dE = eps**(-1) * NB * HB - theta * (B - E) + beta * E
    dF = eps**(-1) * HB**2 - 2 * theta * (D - F)
    dK = -0.5 * sigma**2 * (A - 2*D + F)

    return np.array([dA, dB, dC, dD, dE, dF, dK])


def solve_riccati(params: OWParams, n_points: int = 1000) -> dict:
    """
    Résout le système de Riccati sur [0, T] par intégration numérique.

    Retourne les coefficients (A, B, C, D, E, F, K) et les
    coefficients de stratégie (f, g, h) sur la grille temporelle.
    """
    T = params.T
    lam = params.lam
    eps = params.eps

    # Conditions terminales
    P_T = np.array([
        lam,        # A_T = lambda
        -1.0,       # B_T = -1
        1/lam,      # C_T = 1/lambda
        0.0,        # D_T = 0
        0.0,        # E_T = 0
        0.0,        # F_T = 0
        0.0,        # K_T = 0
    ])

    # Grille temporelle (on résout de T vers 0)
    t_span = (T, 0.0)
    t_eval = np.linspace(T, 0.0, n_points)

    sol = solve_ivp(
        fun=lambda t, P: riccati_rhs(t, P, params),
        t_span=t_span,
        y0=P_T,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )

    # Remet dans l'ordre chronologique (0 → T)
    t_grid = sol.t[::-1]
    A, B, C, D, E, F, K = [sol.y[i][::-1] for i in range(7)]

    # Coefficients de stratégie
    f = -eps**(-1) * (A + lam * B)
    g = -eps**(-1) * (B + lam * C)
    h = -eps**(-1) * (D + lam * E)

    return {
        't': t_grid,
        'A': A, 'B': B, 'C': C,
        'D': D, 'E': E, 'F': F, 'K': K,
        'f': f, 'g': g, 'h': h,
        'params': params
    }