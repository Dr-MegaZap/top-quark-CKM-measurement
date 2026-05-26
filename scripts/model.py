"""
step3_probabilistic_model.py
==============================
Implements the full probabilistic model from Appendix A of
arXiv:2209.01222, specifically Eq. A31.

This module provides:
  - compute_P_nb_nq()    : P(nb, nq | nj, Rb, theta_i)  [Eq. A31]
  - build_likelihood()   : full log-likelihood L(Rb, theta_i) [Eq. 5]
  - profile_likelihood() : PLR and confidence intervals  [Eq. 10-13]
  - asimov_significance(): Z1 = sqrt(q1_A)              [Eq. 14]

All equation numbers refer to arXiv:2209.01222.
"""

import numpy as np
from scipy.stats import binom as binom_dist
from scipy.special import xlogy

# ─────────────────────────────────────────────────────────────────────────────
# Safe binomial PMF
# ─────────────────────────────────────────────────────────────────────────────

def binom_pmf(k, n, p):
    """
    Binomial probability P(X=k) for X ~ Binom(n, p).
    Returns 0 for invalid inputs.
    """
    if n < 0 or k < 0 or k > n:
        return 0.0
    p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return float(binom_dist.pmf(int(k), int(n), p))

# ─────────────────────────────────────────────────────────────────────────────
# P(nj;t | nj) from Eqs. A8–A10
# ─────────────────────────────────────────────────────────────────────────────

def P_njt_given_nj(njt, nj, ftt, kst, p):
    """
    Probability that njt out of nj jets originate from top decays.
    Implements Eqs. A8-A10.

    Parameters
    ----------
    njt : int   — number of jets from top decays (0, 1, or 2)
    nj  : int   — total number of jets in event
    ftt : float — fraction of tt̄ events in this (ll', nj) category
    kst : float — single-top fraction relative to tt̄
    p   : float — probability of capturing a top decay jet (Eq. A7)
    """
    p = float(np.clip(p, 0.0, 1.0))
    if njt == 0:
        return (1-p)**2 * ftt + (1-p) * ftt * kst + (1 - ftt*(1+kst))
    elif njt == 1:
        return 2*p*(1-p)*ftt + p*ftt*kst
    elif njt == 2:
        return p**2 * ftt
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Nuisance parameter interpolation (Eq. 7-8, following Cranmer et al.)
# ─────────────────────────────────────────────────────────────────────────────

def interp_nuisance(eta, theta0, I_plus, I_minus):
    """
    Implements the polynomial interpolation + exponential extrapolation
    of Eq. 8, following Cranmer et al. (ROOT HistFactory, Ref [13]).

    theta = theta0 * f(eta; 1, I+, I-, 1)

    where I+/I- are the +1sigma/-1sigma fractional variations,
    e.g. I+ = 1.05 for +5% uncertainty.

    Parameters
    ----------
    eta    : float — nuisance parameter value (0 = central value)
    theta0 : float — central value of the parameter
    I_plus : float — value of theta/theta0 at eta=+1 (e.g. 1.05)
    I_minus: float — value of theta/theta0 at eta=-1 (e.g. 0.95)
    """
    if eta >= 1.0:
        f = I_plus ** eta
    elif eta <= -1.0:
        f = I_minus ** (-eta)
    else:
        # 6th order polynomial interpolation
        # Boundary conditions: f(+/-1) = I+/-, f'(+/-1) = ln(I+/-)*I+/-
        # f''(+/-1) = (ln I+/-)^2 * I+/-
        # This gives 6 equations for 6 coefficients a1..a6
        lp = np.log(I_plus)  if I_plus  > 0 else 0.0
        lm = np.log(I_minus) if I_minus > 0 else 0.0

        # Build 6x6 system (eta powers 1..6)
        # [a1..a6] such that f(eta) = 1 + sum_j a_j * eta^j
        A = np.array([
            [ 1,  1,  1,  1,  1,  1],   # f(+1) = I+ → sum = I+ - 1
            [-1,  1, -1,  1, -1,  1],   # f(-1) = I- → sum = I- - 1
            [ 1,  2,  3,  4,  5,  6],   # f'(+1) = lp*I+
            [-1,  2, -3,  4, -5,  6],   # f'(-1) = -lm*I- (sign from chain rule)
            [ 0,  2,  6, 12, 20, 30],   # f''(+1) = lp^2*I+
            [ 0,  2, -6, 12,-20, 30],   # f''(-1) = lm^2*I-
        ], dtype=float)

        b = np.array([
            I_plus  - 1,
            I_minus - 1,
            lp * I_plus,
            lm * I_minus,    # note: derivative of I-^{-eta} at eta=-1
            lp**2 * I_plus,
            lm**2 * I_minus,
        ], dtype=float)

        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to linear
            coeffs = np.zeros(6)
            coeffs[0] = (I_plus - I_minus) / 2

        f = 1.0
        for j, a in enumerate(coeffs):
            f += a * eta**(j+1)

    return float(theta0 * np.clip(f, 1e-10, 10.0))

# ─────────────────────────────────────────────────────────────────────────────
# Core: P(nb, nq | nj, Rb, theta_i)  [Eq. A31]
# ─────────────────────────────────────────────────────────────────────────────

def compute_P_nb_nq(nb, nq, nj, Rb,
                    eps_b_B, eps_b_Q,
                    eps_q_B, eps_q_Q,
                    eps_b_jt, eps_q_jt,
                    ftt, kst, flj_t):
    """
    Compute P(nb, nq | nj, Rb, theta_i) via the full sum in Eq. A31.

    Parameters
    ----------
    nb, nq  : int   — observed b-tagged and q-tagged jets
    nj      : int   — total jets in event
    Rb      : float — B(t→bW) / B(t→jW), the parameter of interest
    eps_b_B : float — b-tagger efficiency for true b quarks from top (ε^b_B)
    eps_b_Q : float — b-tagger mistag for true q quarks from top  (ε^b_Q)
    eps_q_B : float — q-tagger mistag for true b quarks from top  (ε^q_B)
    eps_q_Q : float — q-tagger efficiency for true q quarks from top (ε^q_Q)
    eps_b_jt: float — b-tagger rate for non-top jets              (ε^b_{j;/t})
    eps_q_jt: float — q-tagger rate for non-top jets              (ε^q_{j;/t})
    ftt     : float — fraction of tt̄ events in this category
    kst     : float — single-top fraction relative to tt̄
    flj_t   : float — fraction of lj pairs from top decays (Eq. A6)

    Returns
    -------
    float : probability P(nb, nq | nj, Rb, theta_i)
    """

    # ── Probability p of capturing a top decay jet (Eq. A7) ──────────────
    denom = ftt + ftt * kst / 2.0
    p     = float(np.clip(flj_t * nj / denom, 0.0, 1.0)) if denom > 0 else 0.0

    # ── Conditional q-tagger efficiency given not b-tagged (Eq. A27) ─────
    # r_q_B  = eps_q_B  / (1 - eps_b_B)
    # r_q_Q  = eps_q_Q  / (1 - eps_b_Q)
    # r_q_jt = eps_q_jt / (1 - eps_b_jt)
    def safe_ratio(q, b):
        return float(np.clip(q / (1.0 - b), 0.0, 1.0)) if (1.0 - b) > 1e-9 else 0.0

    r_q_B   = safe_ratio(eps_q_B,   eps_b_B)
    r_q_Q   = safe_ratio(eps_q_Q,   eps_b_Q)
    r_q_jt  = safe_ratio(eps_q_jt,  eps_b_jt)

    total = 0.0

    # ── Sum over njt = 0, 1, 2 (Eq. A21) ─────────────────────────────────
    for njt in range(min(2, nj) + 1):
        P_njt = P_njt_given_nj(njt, nj, ftt, kst, p)
        if P_njt <= 0.0:
            continue

        njnt = nj - njt   # non-top jets

        # ── Sum over nB (true b quarks from top) (Eq. A14) ────────────────
        for nB in range(njt + 1):
            P_nB = binom_pmf(nB, njt, Rb)
            if P_nB == 0.0:
                continue
            nQ = njt - nB   # true q quarks from top

            # ── Sum over nbt (b-tagged top jets) ──────────────────────────
            for nbt in range(min(nb, njt) + 1):
                nbt_nt = nb - nbt   # b-tagged non-top jets
                if nbt_nt > njnt or nbt_nt < 0:
                    continue

                # P(nbt_nt | njnt)  [Eq. A12]
                P_nbt_nt = binom_pmf(nbt_nt, njnt, eps_b_jt)
                if P_nbt_nt == 0.0:
                    continue

                avail_nt = njnt - nbt_nt   # non-top jets available for q-tagging

                # ── Sum over nqt (q-tagged top jets) ──────────────────────
                for nqt in range(min(nq, njt - nbt) + 1):
                    nqt_nt = nq - nqt   # q-tagged non-top jets
                    if nqt_nt > avail_nt or nqt_nt < 0:
                        continue

                    # P(nqt_nt | avail_nt) [Eq. A29, second binomial]
                    P_nqt_nt = binom_pmf(nqt_nt, avail_nt, r_q_jt)
                    if P_nqt_nt == 0.0:
                        continue

                    # ── Sum over nbB (b-tagged true b quarks) ─────────────
                    for nbB in range(min(nbt, nB) + 1):
                        nbQ = nbt - nbB   # b-tagged true q quarks
                        if nbQ > nQ or nbQ < 0:
                            continue

                        # P(nbB | nB) and P(nbQ | nQ) [Eqs. A16, A17]
                        P_nbB = binom_pmf(nbB, nB, eps_b_B)
                        P_nbQ = binom_pmf(nbQ, nQ, eps_b_Q)
                        if P_nbB * P_nbQ == 0.0:
                            continue

                        avail_B = nB - nbB   # true b quarks available for q-tag
                        avail_Q = nQ - nbQ   # true q quarks available for q-tag

                        # ── Sum over nqB (q-tagged true b quarks) ─────────
                        for nqB in range(min(nqt, avail_B) + 1):
                            nqQ = nqt - nqB   # q-tagged true q quarks
                            if nqQ > avail_Q or nqQ < 0:
                                continue

                            # P(nqB | avail_B) and P(nqQ | avail_Q)
                            # [Eq. A30, using r_q_B and r_q_Q]
                            P_nqB = binom_pmf(nqB, avail_B, r_q_B)
                            P_nqQ = binom_pmf(nqQ, avail_Q, r_q_Q)

                            total += (P_njt * P_nB
                                      * P_nbt_nt * P_nqt_nt
                                      * P_nbB * P_nbQ
                                      * P_nqB * P_nqQ)

    return float(total)


def build_P_table(nj, Rb, eps_b_B, eps_b_Q, eps_q_B, eps_q_Q,
                  eps_b_jt, eps_q_jt, ftt, kst, flj_t):
    """
    Pre-compute P(nb, nq | nj) for all valid (nb, nq) for given nj.
    Returns (nj+1) x (nj+1) array.
    """
    mat = np.zeros((nj+1, nj+1))
    for nb in range(nj+1):
        for nq in range(nj - nb + 1):
            mat[nb, nq] = compute_P_nb_nq(
                nb, nq, nj, Rb,
                eps_b_B, eps_b_Q, eps_q_B, eps_q_Q,
                eps_b_jt, eps_q_jt,
                ftt, kst, flj_t
            )
    return mat

# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood [Eq. 5]
# ─────────────────────────────────────────────────────────────────────────────

def compute_neg2lnL(Rb,
                    eta_bB, eta_bQ, eta_qB, eta_qQ,
                    N_obs,        # dict: N_obs[ll][nj] = (nj+1)x(nj+1) array
                    N_total_nj,   # dict: N_total_nj[ll][nj] = float
                    ftt, kst, flj_t,
                    theta0,       # dict of central values
                    sys_unc,      # dict of fractional 1-sigma uncertainties
                    use_nq=True):
    """
    Compute -2 ln L(Rb, eta_i).

    Poisson log-likelihood with Gaussian constraints on nuisance
    parameters (Eqs. 5 and 9).

    Parameters
    ----------
    Rb           : float — parameter of interest
    eta_bB/bQ/qB/qQ : float — nuisance parameter values (0 = central)
    N_obs        : observed counts per (ll, nj, nb, nq) bin
    N_total_nj   : total observed events per (ll, nj) category
    ftt, kst, flj_t : dicts[ll][nj] of nuisance parameters
    theta0       : dict of central tagging efficiencies
    sys_unc      : dict of fractional 1-sigma uncertainties
    use_nq       : bool — if False, marginalise over nq (Section II)
    """

    # ── Decode nuisance parameters (Eq. 7-8) ─────────────────────────────
    eps_b_B = interp_nuisance(eta_bB, theta0['eps_b_B'],
                               1 + sys_unc['eps_b_B'],
                               1 - sys_unc['eps_b_B'])
    eps_b_Q = interp_nuisance(eta_bQ, theta0['eps_b_Q'],
                               1 + sys_unc['eps_b_Q'],
                               1 - sys_unc['eps_b_Q'])
    eps_q_B = interp_nuisance(eta_qB, theta0['eps_q_B'],
                               1 + sys_unc['eps_q_B'],
                               1 - sys_unc['eps_q_B'])
    eps_q_Q = interp_nuisance(eta_qQ, theta0['eps_q_Q'],
                               1 + sys_unc['eps_q_Q'],
                               1 - sys_unc['eps_q_Q'])

    # eps_b_jt and eps_q_jt are fitted from data (paper sets to 0.85 * eps_b_Q)
    eps_b_jt = 0.85 * eps_b_Q
    eps_q_jt = 0.85 * eps_q_Q

    # ── Poisson log-likelihood sum ────────────────────────────────────────
    log_L = 0.0

    for ll in range(3):
        for nj in range(2, 5):
            N_tot = N_total_nj[ll][nj]
            if N_tot <= 0:
                continue

            f   = ftt[ll][nj]
            k   = kst[ll][nj]
            flj = flj_t[ll][nj]

            obs = N_obs[ll][nj]   # (nj+1) x (nj+1) array

            if use_nq:
                # Bin in both nb and nq
                for nb in range(nj + 1):
                    for nq in range(nj - nb + 1):
                        N_ij  = obs[nb, nq]
                        P_ij  = compute_P_nb_nq(
                            nb, nq, nj, Rb,
                            eps_b_B, eps_b_Q, eps_q_B, eps_q_Q,
                            eps_b_jt, eps_q_jt,
                            f, k, flj
                        )
                        N_exp = P_ij * N_tot
                        if N_exp > 0 and N_ij >= 0:
                            # Continuous Poisson: N*ln(lambda) - lambda
                            log_L += xlogy(N_ij, N_exp) - N_exp
            else:
                # Marginalise over nq: use only nb bins
                for nb in range(nj + 1):
                    N_ij = obs[nb, :].sum()   # sum over all nq
                    P_ij = sum(
                        compute_P_nb_nq(
                            nb, nq, nj, Rb,
                            eps_b_B, eps_b_Q, eps_q_B, eps_q_Q,
                            eps_b_jt, eps_q_jt,
                            f, k, flj
                        )
                        for nq in range(nj - nb + 1)
                    )
                    N_exp = P_ij * N_tot
                    if N_exp > 0 and N_ij >= 0:
                        log_L += xlogy(N_ij, N_exp) - N_exp

    # ── Gaussian constraints on nuisance parameters (Eq. 9) ──────────────
    for eta in [eta_bB, eta_bQ, eta_qB, eta_qQ]:
        log_L -= 0.5 * eta**2

    return -2.0 * log_L   # return -2 ln L

# ─────────────────────────────────────────────────────────────────────────────
# Profile likelihood ratio fit [Eqs. 10-12]
# ─────────────────────────────────────────────────────────────────────────────

def profile_likelihood_ratio(N_obs, N_total_nj, ftt, kst, flj_t,
                              theta0, sys_unc,
                              Rb_values=None, use_nq=True):
    """
    Compute the profile likelihood ratio -2 ln lambda(Rb) over a grid
    of Rb values. Also returns the global best-fit Rb_hat.

    Implements Eqs. 10-11 (unconstrained) and 12-13 (constrained, Rb<=1).

    Returns
    -------
    Rb_hat    : float — global MLE of Rb
    q_values  : array — -2 ln lambda(Rb) for each Rb in Rb_values
    q_tilde   : array — constrained test statistic q (Eq. 13)
    Rb_values : array — grid of Rb values
    """
    from iminuit import Minuit

    if Rb_values is None:
        Rb_values = np.linspace(0.5, 1.5, 100)

    def neg2lnL(Rb, eta_bB, eta_bQ, eta_qB, eta_qQ):
        return compute_neg2lnL(
            Rb, eta_bB, eta_bQ, eta_qB, eta_qQ,
            N_obs, N_total_nj, ftt, kst, flj_t,
            theta0, sys_unc, use_nq
        )

    # ── Global minimum ────────────────────────────────────────────────────
    m = Minuit(neg2lnL,
               Rb=0.998, eta_bB=0.0, eta_bQ=0.0, eta_qB=0.0, eta_qQ=0.0)
    m.limits["Rb"]     = (0.0, 2.0)
    m.limits["eta_bB"] = (-5., 5.)
    m.limits["eta_bQ"] = (-5., 5.)
    m.limits["eta_qB"] = (-5., 5.)
    m.limits["eta_qQ"] = (-5., 5.)
    m.errordef = 1.0   # for -2 ln L
    m.migrad()

    Rb_hat    = float(m.values["Rb"])
    L_min     = float(m.fval)

    print(f"  Global best fit: Rb = {Rb_hat:.4f} +/- {m.errors['Rb']:.4f}")
    print(f"  Converged: {m.valid}")

    # ── If Rb_hat > 1, compute L at Rb=1 for constrained PLR (Eq. 12) ───
    if Rb_hat > 1.0:
        def neg2lnL_at1(eta_bB, eta_bQ, eta_qB, eta_qQ):
            return compute_neg2lnL(
                1.0, eta_bB, eta_bQ, eta_qB, eta_qQ,
                N_obs, N_total_nj, ftt, kst, flj_t,
                theta0, sys_unc, use_nq
            )
        m1 = Minuit(neg2lnL_at1,
                    eta_bB=0., eta_bQ=0., eta_qB=0., eta_qQ=0.)
        for par in ["eta_bB","eta_bQ","eta_qB","eta_qQ"]:
            m1.limits[par] = (-5., 5.)
        m1.errordef = 1.0
        m1.migrad()
        L_constrained_denom = float(m1.fval)
    else:
        L_constrained_denom = L_min

    # ── Scan over Rb grid ─────────────────────────────────────────────────
    q_values = []
    q_tilde  = []

    for Rb_fixed in Rb_values:
        def neg2lnL_fixed(eta_bB, eta_bQ, eta_qB, eta_qQ):
            return compute_neg2lnL(
                float(Rb_fixed), eta_bB, eta_bQ, eta_qB, eta_qQ,
                N_obs, N_total_nj, ftt, kst, flj_t,
                theta0, sys_unc, use_nq
            )
        mf = Minuit(neg2lnL_fixed,
                    eta_bB=0., eta_bQ=0., eta_qB=0., eta_qQ=0.)
        for par in ["eta_bB","eta_bQ","eta_qB","eta_qQ"]:
            mf.limits[par] = (-5., 5.)
        mf.errordef = 1.0
        mf.migrad()
        L_fixed = float(mf.fval)

        # Unconstrained PLR (Eq. 10-11)
        q = L_fixed - L_min
        q_values.append(max(q, 0.0))

        # Constrained PLR (Eq. 12-13)
        qt = L_fixed - L_constrained_denom
        q_tilde.append(max(qt, 0.0))

    return (Rb_hat,
            np.array(q_values),
            np.array(q_tilde),
            np.array(Rb_values))


def get_confidence_interval(Rb_values, q_values, cl=0.95):
    """
    Extract confidence interval from PLR scan.
    95% CL: q < 3.841 (chi2 with 1 dof)
    """
    threshold = {0.68: 1.0, 0.95: 3.841, 0.99: 6.635}.get(cl, 3.841)
    inside = Rb_values[q_values < threshold]
    if len(inside) == 0:
        return (np.nan, np.nan)
    return (float(inside.min()), float(inside.max()))

# ─────────────────────────────────────────────────────────────────────────────
# Asimov significance [Eqs. 13-14]
# ─────────────────────────────────────────────────────────────────────────────

def compute_asimov_significance(N_total_nj, ftt, kst, flj_t,
                                 theta0, sys_unc,
                                 Rb_true, use_nq=True):
    """
    Compute the Asimov significance Z1 = sqrt(q1_A) for rejecting
    Rb = 1 when the true value is Rb_true.

    Implements Eqs. 13-14. Uses the Asimov dataset where each bin
    yield equals the expected rate at Rb_true with theta_i = theta0_i.

    Parameters
    ----------
    N_total_nj : dict[ll][nj] — total events per category
    ftt, kst, flj_t : nuisance param dicts
    theta0     : dict of central tagging efficiencies
    sys_unc    : dict of fractional uncertainties
    Rb_true    : float — true value of Rb (e.g. RSM_b = 0.998)
    use_nq     : bool

    Returns
    -------
    Z1 : float — expected significance in sigma
    """
    # ── Build Asimov dataset ──────────────────────────────────────────────
    eps_b_B  = theta0['eps_b_B']
    eps_b_Q  = theta0['eps_b_Q']
    eps_q_B  = theta0['eps_q_B']
    eps_q_Q  = theta0['eps_q_Q']
    eps_b_jt = 0.85 * eps_b_Q
    eps_q_jt = 0.85 * eps_q_Q

    N_asimov = {}
    for ll in range(3):
        N_asimov[ll] = {}
        for nj in range(2, 5):
            N_tot = N_total_nj[ll][nj]
            mat   = np.zeros((nj+1, nj+1))
            for nb in range(nj+1):
                for nq in range(nj - nb + 1):
                    P = compute_P_nb_nq(
                        nb, nq, nj, Rb_true,
                        eps_b_B, eps_b_Q, eps_q_B, eps_q_Q,
                        eps_b_jt, eps_q_jt,
                        ftt[ll][nj], kst[ll][nj], flj_t[ll][nj]
                    )
                    mat[nb, nq] = P * N_tot
            N_asimov[ll][nj] = mat

    # ── Evaluate q1 at Rb=1 using Asimov dataset ─────────────────────────
    _, q_vals, q_tilde, _ = profile_likelihood_ratio(
        N_obs      = N_asimov,
        N_total_nj = N_total_nj,
        ftt        = ftt,
        kst        = kst,
        flj_t      = flj_t,
        theta0     = theta0,
        sys_unc    = sys_unc,
        Rb_values  = np.array([1.0]),
        use_nq     = use_nq
    )

    # Eq. 14: q1 = -2 ln lambda(1) if Rb_hat <= 1, else 0
    # With Asimov dataset and Rb_true < 1, Rb_hat <= 1 always
    q1_A = float(q_tilde[0])
    Z1   = np.sqrt(max(q1_A, 0.0))

    return Z1