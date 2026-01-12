# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:03:29 2025

@author: WONCHAN
"""

# -*- coding: utf-8 -*-
"""
DeepONet policy on belief + trunk(tau, logW) with:
- Checkpoint format v2 (meta + normalizers + model_state) and per-seed best ckpt
- Baselines (RF-only, EW buy&hold, EW daily rebalance) under the SAME evaluation protocol
- Multi-seed runs + aggregation (mean/std/95% CI) and LaTeX tables from aggregated results
- Two "next-step" options implemented as toggles:
  (A) Turnover-penalty warmup schedule (lambda_tc ramp-up)
  (B) Diversification regularizers: anchor-to-EW (L2) and entropy bonus
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import math
import time
import random
import shutil
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETER PRESETS (NEW: Fix training instability)
# =============================================================================

def get_parameter_preset(preset_name: str = "balanced"):
    """Three presets to fix best_it=1, turnover explosion, VAL degradation."""
    
    if preset_name == "conservative":
        return {
            "lr": 1e-5, "batch_size": 256, "early_stop_patience": 10,
            "tc_warmup_steps": 800, "tc_cooldown_steps": 400, "tc_final_mult": 2.0,
            "turnover_clip_start": 0.15, "turnover_clip_end": 0.10,
            "tc_cost_frac_clip": 0.20, "turnover_rescale": 0.5,
            "turnover_clip_warmup_steps": 600,
            "pi_anchor_beta": 0.20, "pi_entropy_beta": 0.10,
            "belief_sigma_floor": 1e-6, "belief_sigma_clip": 5.0,
            "belief_shrink_kappa_kalman": 5.0, "belief_z_clip": 6.0,
            "ckpt_turnover_weight": 0.30, "ckpt_clamp_weight": 1.00, "ckpt_eps": 1e-4,
        }
    elif preset_name == "balanced":
        return {
            "lr": 5e-5, "batch_size": 384, "early_stop_patience": 8,
            "tc_warmup_steps": 600, "tc_cooldown_steps": 300, "tc_final_mult": 1.5,
            "turnover_clip_start": 0.25, "turnover_clip_end": 0.15,
            "tc_cost_frac_clip": 0.30, "turnover_rescale": 0.5,
            "turnover_clip_warmup_steps": 500,
            "pi_anchor_beta": 0.10, "pi_entropy_beta": 0.05,
            "belief_sigma_floor": 1e-6, "belief_sigma_clip": 8.0,
            "belief_shrink_kappa_kalman": 3.0, "belief_z_clip": 7.0,
            "ckpt_turnover_weight": 0.20, "ckpt_clamp_weight": 0.75, "ckpt_eps": 5e-5,
        }
    elif preset_name == "exploration":
        return {
            "lr": 1e-4, "batch_size": 512, "early_stop_patience": 8,
            "tc_warmup_steps": 1000, "tc_cooldown_steps": 300, "tc_final_mult": 2.5,
            "turnover_clip_start": 0.40, "turnover_clip_end": 0.12,
            "tc_cost_frac_clip": 0.35, "turnover_rescale": 0.5,
            "turnover_clip_warmup_steps": 800,
            "pi_anchor_beta": 0.05, "pi_entropy_beta": 0.03,
            "belief_sigma_floor": 1e-6, "belief_sigma_clip": 10.0,
            "belief_shrink_kappa_kalman": 2.0, "belief_z_clip": 8.0,
            "ckpt_turnover_weight": 0.15, "ckpt_clamp_weight": 0.60, "ckpt_eps": 3e-5,
        }
    raise ValueError(f"Unknown preset: {preset_name}. Use: conservative, balanced, exploration")



# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clamp_float(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(x.item())

def huber_abs(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = torch.abs(x)
    if delta <= 0:
        return ax
    return torch.where(ax <= delta, 0.5 * (ax * ax) / delta, ax - 0.5 * delta)

def tc_lambda_schedule(it: int, iters: int, lam_base: float, warmup_steps: int, late_ramp_frac: float = 0.20, late_mult: float = 2.0) -> float:
    iters = max(int(iters), 1)
    warm = max(int(warmup_steps), 0)
    if warm > 0 and it <= warm:
        return lam_base * float(it) / float(warm)
    t = float(it) / float(iters)
    if t >= 1.0 - late_ramp_frac:
        u = (t - (1.0 - late_ramp_frac)) / late_ramp_frac
        return lam_base * (1.0 + u * (late_mult - 1.0))
    return lam_base

def turnover_clip_schedule(it: int, iters: int, clip_start: float = 2.0, clip_end: float = 0.5) -> float:
    iters = max(int(iters), 1)
    t = float(it) / float(iters)
    return float(clip_start + t * (clip_end - clip_start))

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    phat = k / n
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt((phat*(1-phat)/n) + (z*z)/(4*n*n))
    return phat, max(0.0, center-half), min(1.0, center+half)

# =============================================================================
# Normalizers
# =============================================================================

class TorchNormalizer:
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.maximum(self.std, self.eps)

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / (self.std + self.eps)

    def apply_torch(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        mean = torch.tensor(self.mean, device=X.device, dtype=X.dtype).view(1, -1)
        std = torch.tensor(self.std, device=X.device, dtype=X.dtype).view(1, -1)
        return (X - mean) / (std + self.eps)


def _norm_to_dict(norm: Optional[TorchNormalizer]) -> Optional[Dict[str, Any]]:
    if norm is None:
        return None
    if norm.mean is None or norm.std is None:
        return {"mean": None, "std": None, "eps": norm.eps}
    return {
        "mean": norm.mean.astype(float).tolist(),
        "std": norm.std.astype(float).tolist(),
        "eps": float(norm.eps),
    }


def _dict_to_norm(d: Optional[Dict[str, Any]]) -> Optional[TorchNormalizer]:
    if d is None:
        return None
    mean = d.get("mean", None)
    std = d.get("std", None)
    eps = float(d.get("eps", 1e-8))
    norm = TorchNormalizer(eps=eps)
    if mean is None or std is None:
        return norm
    norm.mean = np.array(mean, dtype=np.float64)
    norm.std = np.array(std, dtype=np.float64)
    return norm


# =============================================================================
# Checkpoint (v2) - safe with PyTorch >= 2.6
# =============================================================================

def save_checkpoint_v2(
    path: str,
    model: nn.Module,
    wnorm: Optional[TorchNormalizer],
    bnorm: Optional[TorchNormalizer],
    meta: Dict[str, Any],
):
    ckpt = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "wnorm": _norm_to_dict(wnorm),
        "bnorm": _norm_to_dict(bnorm),
        "meta": meta,
    }
    torch.save(ckpt, path)


def load_checkpoint_v2(path: str, model: nn.Module, device: torch.device):
    """Robust loader for the v2 checkpoint format.

    - Prefers safe loading (weights_only=True) on PyTorch>=2.6.
    - Falls back gracefully for older PyTorch or legacy pickle payloads.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Invalid checkpoint format at: {path}")

    model.load_state_dict(ckpt["model_state"])
    wnorm = _dict_to_norm(ckpt.get("wnorm", None))
    bnorm = _dict_to_norm(ckpt.get("bnorm", None))
    meta = ckpt.get("meta", {})
    return wnorm, bnorm, meta


# =============================================================================
# Simplex Projection
# =============================================================================

def simplex_projection_exact(pi_raw: torch.Tensor) -> torch.Tensor:
    """
    Exact projection onto simplex (row-wise):
        pi_i >= 0, sum_i pi_i = 1
    NOTE: This is used in forward under no_grad (or detached usage).
    """
    x = pi_raw
    B, d = x.shape
    x_sorted, _ = torch.sort(x, dim=1, descending=True)
    cssv = torch.cumsum(x_sorted, dim=1) - 1.0
    ind = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).view(1, -1)
    cond = x_sorted - cssv / ind > 0
    rho = torch.sum(cond, dim=1) - 1
    rho = torch.clamp(rho, min=0)
    theta = cssv[torch.arange(B, device=x.device), rho] / (rho + 1).to(x.dtype)
    w = torch.clamp(x - theta.view(-1, 1), min=0.0)
    # numerical renormalization (rarely needed but safe)
    w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
    return w


def simplex_projection(pi_raw: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Straight-through simplex projection:
      - Forward: exact simplex projection (can produce zeros)
      - Backward: uses softmax gradient (smooth, stable)

    This fixes: RuntimeError: tensor does not require grad.
    """
    # smooth surrogate for gradients
    pi_soft = torch.softmax(pi_raw / float(temperature), dim=1)

    # exact projection for forward pass (no gradient)
    with torch.no_grad():
        pi_proj = simplex_projection_exact(pi_raw)

    # straight-through estimator: forward=pi_proj, backward=pi_soft
    pi = pi_proj + (pi_soft - pi_soft.detach())
    return pi


# =============================================================================
# Utility
# =============================================================================

def utility_crra(W: torch.Tensor, gamma: float) -> torch.Tensor:
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(torch.clamp(W, min=1e-12))
    return torch.pow(torch.clamp(W, min=1e-12), 1.0 - gamma) / (1.0 - gamma)


# =============================================================================
# Belief estimators
# =============================================================================

def compute_rolling_beliefs(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """
    returns: (T, d) daily excess returns (or returns)
    belief: concat(mu_hat, chol_upper_flat or diag of Sigma_hat)
    We'll use (mu_hat, diag_sigma) as a compact belief.
    """
    T, d = returns.shape
    belief_dim = 2 * d
    beliefs = np.zeros((T, belief_dim), dtype=np.float32)

    for t in range(T):
        if t < window:
            beliefs[t] = 0.0
            continue
        X = returns[t - window:t]
        mu = X.mean(axis=0)
        sig = X.std(axis=0)
        beliefs[t, :d] = mu
        beliefs[t, d:] = sig
    return beliefs


def compute_kalman_beliefs(
    returns: np.ndarray,
    q: float = 1e-5,
    r: float = 1e-3,
    alpha: float = 0.02,
    mu_clip: float = 0.50,
    sigma_clip: float = 1.50,
    shrink_kappa: float = 25.0,
) -> np.ndarray:
    T, d = returns.shape
    mu_hat = np.zeros((T, d), dtype=np.float64)
    sigma_hat = np.zeros((T, d), dtype=np.float64)

    m = np.zeros(d, dtype=np.float64)
    P = np.ones(d, dtype=np.float64) * 1e-2
    v = np.ones(d, dtype=np.float64) * 1e-2
    eps = 1e-12

    for t in range(T):
        z = returns[t].astype(np.float64)

        v = (1.0 - alpha) * v + alpha * (z * z)
        meas_var = np.maximum(r * v, eps)

        P = P + q
        K = P / (P + meas_var)
        innov = z - m
        m = m + K * innov
        P = (1.0 - K) * P

        shrink = 1.0 / (1.0 + shrink_kappa * P)
        m_shrunk = m * shrink

        if mu_clip is not None and mu_clip > 0:
            m_shrunk = np.clip(m_shrunk, -mu_clip, mu_clip)

        s = np.sqrt(np.maximum(v, eps))
        if sigma_clip is not None and sigma_clip > 0:
            s = np.clip(s, eps, sigma_clip)

        mu_hat[t] = m_shrunk
        sigma_hat[t] = s

    return np.concatenate([mu_hat, sigma_hat], axis=1).astype(np.float32)


# =============================================================================
# DeepONet policy
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 3, act: str = "silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth

        if act == "relu":
            A = nn.ReLU
        elif act == "tanh":
            A = nn.Tanh
        else:
            A = nn.SiLU

        layers = []
        d0 = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d0, width))
            layers.append(A())
            d0 = width
        layers.append(nn.Linear(d0, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONetPolicy(nn.Module):
    """
    trunk: (tau, logW) -> h_trunk
    branch: belief vector -> h_branch
    combine: elementwise product -> head -> (pi_raw, rho_raw)
    """
    def __init__(
        self,
        trunk_in_dim: int,
        branch_in_dim: int,
        d_assets: int,
        width: int = 256,
        depth: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        self.trunk = MLP(trunk_in_dim, width, width=width, depth=depth, act=act)
        self.branch = MLP(branch_in_dim, width, width=width, depth=depth, act=act)
        self.head_pi = MLP(width, d_assets, width=width, depth=2, act=act)
        self.head_rho = MLP(width, 1, width=width, depth=2, act=act)

    def forward(self, trunk_x: torch.Tensor, branch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ht = self.trunk(trunk_x)
        hb = self.branch(branch_x)
        h = ht * hb
        pi_raw = self.head_pi(h)
        rho_raw = self.head_rho(h)
        return pi_raw, rho_raw


# =============================================================================
# Simulation core (policy / baseline)
# =============================================================================

def simulate_batch_policy(
    model: nn.Module,
    returns_t: torch.Tensor,          # (T, d)
    rf_t: torch.Tensor,               # (T,)
    belief_t: torch.Tensor,           # (T, belief_dim)
    start_idx: torch.Tensor,          # (B,) starting indices
    episode_len: int,
    device: torch.device,
    wnorm: TorchNormalizer,
    bnorm: TorchNormalizer,
    gamma: float,
    lambda_tc: float,
    W0: float,
    W_min: float,
    W_max: float,
    rho_max: float = 0.0,
    pi_anchor_beta: float = 0.0,
    pi_entropy_beta: float = 0.0,
    record_paths: bool = False,
    # --- NEW: turnover penalty rescale/clip + tc cost cap ---
    turnover_rescale: float = 0.5,          # L1(pi-w) in [0,2] -> *0.5 => [0,1]
    turnover_clip: float = 1.0,             # clip on rescaled turnover
    tc_cost_frac_clip: float = 1.0,         # cap tc_cost <= frac * W (1.0 means no cap)
    # --- NEW: belief stabilization (esp. Kalman) ---
    belief_mu_dim: Optional[int] = None,    # if None, infer as belief_dim//2
    belief_sigma_floor: float = 1e-8,
    belief_sigma_clip: float = 1e6,
    belief_shrink_kappa: float = 0.0,       # mu <- mu / (1 + kappa*sigma)
    belief_z_clip: float = 8.0,             # clip AFTER whitening (bnorm) in z-space
) -> Dict[str, torch.Tensor]:
    B = start_idx.shape[0]
    d = returns_t.shape[1]
    eps = 1e-12

    W = torch.full((B,), float(W0), device=device)
    W = torch.clamp(W, min=W_min)

    # Current holdings weights at the beginning of each step (post previous returns).
    w_cur = torch.full((B, d), 1.0 / d, device=device)

    turnovers: List[torch.Tensor] = []
    tc_fracs: List[torch.Tensor] = []
    clamp_hits: List[torch.Tensor] = []
    ent_list: List[torch.Tensor] = []
    anchor_list: List[torch.Tensor] = []

    if record_paths:
        W_path = torch.empty((episode_len + 1, B), device=device)
        pi_path = torch.empty((episode_len, B, d), device=device)
        turn_path = torch.empty((episode_len, B), device=device)
        W_path[0] = W

    # infer belief dims
    belief_dim = belief_t.shape[1]
    mu_dim = int(belief_mu_dim) if belief_mu_dim is not None else int(belief_dim // 2)

    for k in range(episode_len):
        t_idx = start_idx + k

        tau = (episode_len - k) / float(episode_len)
        tau_t = torch.full((B, 1), float(tau), device=device)

        logW = torch.log(torch.clamp(W, min=1e-8)).view(B, 1)
        trunk_x = torch.cat([tau_t, logW], dim=1)
        trunk_x = wnorm.apply_torch(trunk_x)

        # -------- belief stabilization (works for rolling/kalman that are [mu, sigma]) --------
        b_raw = belief_t[t_idx]  # (B, belief_dim)
        if mu_dim > 0 and (mu_dim * 2) <= belief_dim:
            mu = b_raw[:, :mu_dim]
            sig = b_raw[:, mu_dim:mu_dim * 2]
            sig = torch.clamp(sig, min=float(belief_sigma_floor), max=float(belief_sigma_clip))
            if belief_shrink_kappa and float(belief_shrink_kappa) > 0.0:
                mu = mu / (1.0 + float(belief_shrink_kappa) * sig)
            b_raw = torch.cat([mu, sig, b_raw[:, mu_dim * 2:]], dim=1)

        b = bnorm.apply_torch(b_raw)
        if belief_z_clip is not None and float(belief_z_clip) > 0.0:
            b = torch.clamp(b, min=-float(belief_z_clip), max=float(belief_z_clip))

        pi_raw, rho_raw = model(trunk_x, b)

        # Portfolio projection to simplex (long-only, fully invested)
        pi = simplex_projection(pi_raw)

        # Optional consumption ratio (budget-feasible)
        if rho_max > 0.0:
            rho = torch.sigmoid(rho_raw).squeeze(1) * float(rho_max)
        else:
            rho = torch.zeros((B,), device=device)

        # Anchor-to-equal-weight penalty (stability / diversification)
        if pi_anchor_beta > 0.0:
            ew = torch.full((1, d), 1.0 / d, device=device)
            anchor_pen = torch.mean(torch.sum((pi - ew) ** 2, dim=1))
            anchor_list.append(anchor_pen)

        # Entropy regularizer (diversification)
        if pi_entropy_beta > 0.0:
            ent = -torch.sum(pi * torch.log(torch.clamp(pi, min=eps)), dim=1)
            ent_pen = -torch.mean(ent)  # maximize entropy => minimize (-entropy)
            ent_list.append(ent_pen)

        # Turnover vs current holdings (post previous returns)
        turnover_raw = torch.sum(torch.abs(pi - w_cur), dim=1)          # in [0,2]
        turnover = turnover_raw * float(turnover_rescale)              # typically in [0,1]
        if turnover_clip is not None and float(turnover_clip) > 0.0:
            turnover = torch.clamp(turnover, min=0.0, max=float(turnover_clip))

        turnovers.append(turnover)

        # TC cost cap to avoid blow-ups: tc_cost <= frac*W
        tc_cost = float(lambda_tc) * turnover * W
        if tc_cost_frac_clip is not None and float(tc_cost_frac_clip) < 1.0:
            tc_cost = torch.minimum(tc_cost, float(tc_cost_frac_clip) * W)

        tc_frac = tc_cost / torch.clamp(W, min=1e-12)
        tc_fracs.append(tc_frac)

        W_after_tc = torch.clamp(W - tc_cost, min=W_min)

        c = rho * W_after_tc
        W_invest = torch.clamp(W_after_tc - c, min=W_min)

        rf_now = rf_t[t_idx]                  # (B,)
        r_next = returns_t[t_idx + 1]         # (B,d)

        gross = 1.0 + rf_now + torch.sum(pi * r_next, dim=1)
        gross = torch.clamp(gross, min=1e-6)

        W_next = W_invest * gross
        W_next = torch.clamp(W_next, min=W_min, max=W_max)

        # clamp-hit rate (stability proxy)
        hit = (W_next <= (W_min * (1.0 + 1e-12))) | (W_next >= (W_max * (1.0 - 1e-12)))
        clamp_hits.append(hit.to(W_next.dtype))

        # Update holdings weights for the next step (post returns, no immediate re-trade)
        asset_gross = 1.0 + rf_now.view(B, 1) + r_next
        asset_gross = torch.clamp(asset_gross, min=1e-6)

        w_next = (pi * asset_gross) / gross.view(B, 1)
        w_next = torch.clamp(w_next, min=0.0)
        w_next = w_next / torch.clamp(w_next.sum(dim=1, keepdim=True), min=1e-8)

        if record_paths:
            pi_path[k] = pi
            turn_path[k] = turnover
            W_path[k + 1] = W_next

        W = W_next
        w_cur = w_next

    WT = W
    U = utility_crra(WT, gamma=gamma)
    obj = -torch.mean(U)

    if len(anchor_list) > 0:
        obj = obj + float(pi_anchor_beta) * torch.mean(torch.stack(anchor_list))
    if len(ent_list) > 0:
        obj = obj + float(pi_entropy_beta) * torch.mean(torch.stack(ent_list))

    avg_turn = torch.mean(torch.stack(turnovers))
    avg_tc_frac = torch.mean(torch.stack(tc_fracs)) if len(tc_fracs) > 0 else torch.tensor(0.0, device=device)
    clamp_rate = torch.mean(torch.stack(clamp_hits)) if len(clamp_hits) > 0 else torch.tensor(0.0, device=device)

    out: Dict[str, torch.Tensor] = {
        "obj": obj,
        "WT": WT,
        "avg_turnover": avg_turn,
        "avg_tc_frac": avg_tc_frac,
        "clamp_rate": clamp_rate,
    }
    if record_paths:
        out["W_path"] = W_path
        out["pi_path"] = pi_path
        out["turn_path"] = turn_path
    return out



def simulate_batch_baseline(
    returns_t: torch.Tensor,
    rf_t: torch.Tensor,
    start_idx: torch.Tensor,
    episode_len: int,
    device: torch.device,
    gamma: float,
    lambda_tc_for_rebal: float,
    baseline_name: str,
    W0: float,
    W_min: float,
    W_max: float,
) -> Dict[str, torch.Tensor]:
    B = start_idx.shape[0]
    d = returns_t.shape[1]

    W = torch.full((B,), float(W0), device=device)
    W = torch.clamp(W, min=W_min)

    # Current holdings weights at the beginning of each step (post previous returns).
    w_cur = torch.full((B, d), 1.0 / d, device=device)

    turnovers: List[torch.Tensor] = []

    for k in range(episode_len):
        t_idx = start_idx + k
        rf_now = rf_t[t_idx]                        # (B,)
        r_next = returns_t[t_idx + 1]               # (B,d)

        if baseline_name == "rf_only":
            turnover = torch.zeros((B,), device=device)
            W_next = W * (1.0 + rf_now)
            W_next = torch.clamp(W_next, min=W_min, max=W_max)
            w_next = w_cur

        elif baseline_name == "ew_bh":
            turnover = torch.zeros((B,), device=device)
            gross = 1.0 + rf_now + torch.sum(w_cur * r_next, dim=1)
            gross = torch.clamp(gross, min=1e-6)

            W_next = W * gross
            W_next = torch.clamp(W_next, min=W_min, max=W_max)

            asset_gross = 1.0 + rf_now.view(B, 1) + r_next
            asset_gross = torch.clamp(asset_gross, min=1e-6)

            w_next = (w_cur * asset_gross) / gross.view(B, 1)
            w_next = torch.clamp(w_next, min=0.0)
            w_next = w_next / torch.clamp(w_next.sum(dim=1, keepdim=True), min=1e-8)

        elif baseline_name == "ew_rebal":
            pi_target = torch.full((B, d), 1.0 / d, device=device)
            turnover = torch.sum(torch.abs(pi_target - w_cur), dim=1)

            tc_cost = lambda_tc_for_rebal * turnover * W
            W_after_tc = torch.clamp(W - tc_cost, min=W_min)

            gross = 1.0 + rf_now + torch.sum(pi_target * r_next, dim=1)
            gross = torch.clamp(gross, min=1e-6)

            W_next = W_after_tc * gross
            W_next = torch.clamp(W_next, min=W_min, max=W_max)

            asset_gross = 1.0 + rf_now.view(B, 1) + r_next
            asset_gross = torch.clamp(asset_gross, min=1e-6)

            w_next = (pi_target * asset_gross) / gross.view(B, 1)
            w_next = torch.clamp(w_next, min=0.0)
            w_next = w_next / torch.clamp(w_next.sum(dim=1, keepdim=True), min=1e-8)

        else:
            raise ValueError(f"Unknown baseline_name: {baseline_name}")

        turnovers.append(turnover)
        W = W_next
        w_cur = w_next

    WT = W
    U = utility_crra(WT, gamma=gamma)
    obj = -torch.mean(U)

    avg_turn = torch.mean(torch.stack(turnovers))

    out: Dict[str, torch.Tensor] = {
        "obj": obj,
        "WT": WT,
        "avg_turnover": avg_turn
    }
    return out


# =============================================================================
# Data loading
# =============================================================================

def load_timeseries_csv(path_csv: str, assets: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path_csv)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    rf_col_candidates = [c for c in df.columns if c.lower() in ["rf", "r_f", "riskfree", "risk_free", "rf_daily"]]
    if len(rf_col_candidates) == 0:
        raise ValueError("No rf column found. Expected one of: rf, r_f, riskfree, risk_free, rf_daily")
    rf_col = rf_col_candidates[0]

    missing = [a for a in assets if a not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing assets in CSV: {missing}")

    returns = df[assets].values.astype(np.float32)
    rf = df[rf_col].values.astype(np.float32)

    # belief can be computed later; placeholder return
    return df, returns, rf, df["Date"].values if "Date" in df.columns else np.arange(len(df))


# =============================================================================
# Splits
# =============================================================================

@dataclass
class Split:
    name: str
    start: int
    end: int  # exclusive


def make_splits(n: int, train_end: int, val_end: int, test_end: int) -> Dict[str, Split]:
    return {
        "train": Split("train", 0, train_end),
        "val": Split("val", train_end, val_end),
        "test": Split("test", val_end, test_end),
    }


def split_date_range(dates: np.ndarray, split: Split) -> Tuple[str, str]:
    if split.start >= len(dates) or split.end - 1 >= len(dates):
        return ("", "")
    a = str(dates[split.start])[:10]
    b = str(dates[split.end - 1])[:10]
    return (a, b)


# =============================================================================
# Normalizer fit
# =============================================================================

def fit_normalizers(
    returns: np.ndarray,
    rf: np.ndarray,
    beliefs: np.ndarray,
    splits: Dict[str, Split],
    episode_len: int,
    W0: float,
    belief_name: str,
    window_valid_mask: np.ndarray,
) -> Tuple[TorchNormalizer, TorchNormalizer]:
    """
    Fit wnorm on trunk inputs [tau, logW] and bnorm on beliefs.
    We approximate W by simulating a simple equal-weight buy&hold (no TC) on train.
    """
    train = splits["train"]
    T, d = returns.shape

    # Build training indices where belief is valid and episode fits.
    idxs = []
    for t in range(train.start, train.end - episode_len - 1):
        if not window_valid_mask[t]:
            continue
        idxs.append(t)
    if len(idxs) == 0:
        idxs = list(range(train.start, min(train.end - episode_len - 1, train.start + 256)))

    # Approx W path approx by EW buy&hold
    W = W0
    w = np.ones((d,), dtype=np.float64) / d
    logW_list = []
    tau_list = []
    belief_list = []

    # Build a small sample over idxs
    sample = idxs[: min(len(idxs), 2048)]
    for t0 in sample:
        W = W0
        w = np.ones((d,), dtype=np.float64) / d
        for k in range(episode_len):
            t = t0 + k
            tau = (episode_len - k) / float(episode_len)
            logW_list.append([tau, math.log(max(W, 1e-8))])
            belief_list.append(beliefs[t].astype(np.float64).tolist())

            rf_now = float(rf[t])
            r_next = returns[t + 1].astype(np.float64)
            gross = 1.0 + rf_now + float(np.dot(w, r_next))
            gross = max(gross, 1e-6)
            W = W * gross

            # drift weights
            asset_gross = 1.0 + rf_now + r_next
            asset_gross = np.maximum(asset_gross, 1e-6)
            w = (w * asset_gross) / gross
            s = np.sum(w)
            if s <= 0:
                w = np.ones((d,), dtype=np.float64) / d
            else:
                w = w / s

    Xw = np.array(logW_list, dtype=np.float64)
    Xb = np.array(belief_list, dtype=np.float64)

    wnorm = TorchNormalizer()
    bnorm = TorchNormalizer()
    wnorm.fit(Xw)
    bnorm.fit(Xb)
    return wnorm, bnorm


# =============================================================================
# Evaluation for split
# =============================================================================

def eval_split_policy(
    model: nn.Module,
    returns_t: torch.Tensor,
    rf_t: torch.Tensor,
    belief_t: torch.Tensor,
    split: Split,
    valid_mask: np.ndarray,
    episode_len: int,
    device: torch.device,
    wnorm: TorchNormalizer,
    bnorm: TorchNormalizer,
    gamma: float,
    lambda_tc: float,
    W0: float,
    W_min: float,
    W_max: float,
    rho_max: float,
    pi_anchor_beta: float,
    pi_entropy_beta: float,
    n_paths: int = 512,
) -> Dict[str, float]:
    T = returns_t.shape[0]
    start_candidates = []
    for t in range(split.start, split.end - episode_len - 1):
        if valid_mask[t]:
            start_candidates.append(t)
    if len(start_candidates) == 0:
        start_candidates = list(range(split.start, min(split.end - episode_len - 1, split.start + 2048)))

    if len(start_candidates) == 0:
        raise ValueError(f"No valid start indices for split={split.name}")

    pick = np.random.choice(start_candidates, size=min(n_paths, len(start_candidates)), replace=False)
    start_idx = torch.tensor(pick, device=device, dtype=torch.long)

    out = simulate_batch_policy(
        model=model,
        returns_t=returns_t,
        rf_t=rf_t,
        belief_t=belief_t,
        start_idx=start_idx,
        episode_len=episode_len,
        device=device,
        wnorm=wnorm,
        bnorm=bnorm,
        gamma=gamma,
        lambda_tc=lambda_tc,
        W0=W0,
        W_min=W_min,
        W_max=W_max,
        rho_max=rho_max,
        pi_anchor_beta=pi_anchor_beta,
        pi_entropy_beta=pi_entropy_beta,
        record_paths=False,
    )

    WT = out["WT"].detach().cpu().numpy()
    obj = safe_float(out["obj"].detach().cpu())
    avg_turn = safe_float(out["avg_turnover"].detach().cpu())

    # CE_term: certainty equivalent under CRRA for terminal wealth
    if abs(gamma - 1.0) < 1e-12:
        CE = float(np.exp(np.mean(np.log(np.maximum(WT, 1e-12)))))
    else:
        EU = np.mean((np.maximum(WT, 1e-12) ** (1.0 - gamma)) / (1.0 - gamma))
        CE = float(((1.0 - gamma) * EU) ** (1.0 / (1.0 - gamma)))

    W_T_mean = float(np.mean(WT))
    return {
        "obj": obj,
        "CE_term": CE,
        "W_T_mean": W_T_mean,
        "avg_TC": avg_turn,
    }


def eval_split_baseline(
    returns_t: torch.Tensor,
    rf_t: torch.Tensor,
    split: Split,
    valid_mask: np.ndarray,
    episode_len: int,
    device: torch.device,
    gamma: float,
    lambda_tc_for_rebal: float,
    baseline_name: str,
    W0: float,
    W_min: float,
    W_max: float,
    n_paths: int = 512,
) -> Dict[str, float]:
    start_candidates = []
    for t in range(split.start, split.end - episode_len - 1):
        if valid_mask[t]:
            start_candidates.append(t)
    if len(start_candidates) == 0:
        start_candidates = list(range(split.start, min(split.end - episode_len - 1, split.start + 2048)))

    pick = np.random.choice(start_candidates, size=min(n_paths, len(start_candidates)), replace=False)
    start_idx = torch.tensor(pick, device=device, dtype=torch.long)

    out = simulate_batch_baseline(
        returns_t=returns_t,
        rf_t=rf_t,
        start_idx=start_idx,
        episode_len=episode_len,
        device=device,
        gamma=gamma,
        lambda_tc_for_rebal=lambda_tc_for_rebal,
        baseline_name=baseline_name,
        W0=W0,
        W_min=W_min,
        W_max=W_max,
    )

    WT = out["WT"].detach().cpu().numpy()
    obj = safe_float(out["obj"].detach().cpu())
    avg_turn = safe_float(out["avg_turnover"].detach().cpu())

    if abs(gamma - 1.0) < 1e-12:
        CE = float(np.exp(np.mean(np.log(np.maximum(WT, 1e-12)))))
    else:
        EU = np.mean((np.maximum(WT, 1e-12) ** (1.0 - gamma)) / (1.0 - gamma))
        CE = float(((1.0 - gamma) * EU) ** (1.0 / (1.0 - gamma)))

    W_T_mean = float(np.mean(WT))
    return {
        "obj": obj,
        "CE_term": CE,
        "W_T_mean": W_T_mean,
        "avg_TC": avg_turn,
    }


# =============================================================================
# Train loop
# =============================================================================

def train_one_run(
    model: nn.Module,
    returns_t: torch.Tensor,
    rf_t: torch.Tensor,
    belief_t: torch.Tensor,
    splits: Dict[str, Split],
    valid_mask: np.ndarray,
    episode_len: int,
    device: torch.device,
    seed: int,
    out_dir: str,
    belief_name: str,
    lam_tc: float,
    lr: float,
    iters: int,
    batch_size: int,
    gamma: float,
    W0: float,
    W_min: float,
    W_max: float,
    rho_max: float,
    eval_every: int,
    early_stop_patience: int,
    pi_anchor_beta: float,
    pi_entropy_beta: float,
    tc_warmup_steps: int,
    # --- NEW: schedule/clip knobs ---
    tc_cooldown_steps: int = 0,
    tc_final_mult: float = 1.0,                 # cooldown target: lam_tc * tc_final_mult
    turnover_rescale: float = 0.5,
    turnover_clip_start: float = 1.0,
    turnover_clip_end: float = 1.0,
    turnover_clip_warmup_steps: int = 0,
    tc_cost_frac_clip: float = 1.0,
    # --- NEW: belief stabilization (esp. Kalman) ---
    belief_sigma_floor: float = 1e-8,
    belief_sigma_clip: float = 1e6,
    belief_shrink_kappa: float = 0.0,           # recommend >0 for kalman
    belief_z_clip: float = 8.0,
    # --- NEW: checkpoint selection weights ---
    ckpt_turnover_weight: float = 0.05,         # score = CE - w_turn*turn - w_clamp*clamp
    ckpt_clamp_weight: float = 0.50,
    ckpt_eps: float = 1e-12,
) -> Tuple[str, Dict[str, Any]]:
    set_seed(seed)
    model.to(device)
    model.train()

    wnorm, bnorm = fit_normalizers(
        returns=returns_t.detach().cpu().numpy(),
        rf=rf_t.detach().cpu().numpy(),
        beliefs=belief_t.detach().cpu().numpy(),
        splits=splits,
        episode_len=episode_len,
        W0=W0,
        belief_name=belief_name,
        window_valid_mask=valid_mask,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    run_dir = os.path.join(out_dir, "runs", f"belief{belief_name}", f"lam{lam_tc:.6g}", f"seed{seed}")
    ensure_dir(run_dir)

    best_path = os.path.join(run_dir, f"best_seed{seed}_belief{belief_name}_lam{lam_tc:.6g}.pth")

    best_val_score = -1e18
    best_val_CE = -1e18
    best_it = -1
    no_improve = 0

    train_split = splits["train"]
    train_candidates = []
    for t in range(train_split.start, train_split.end - episode_len - 1):
        if valid_mask[t]:
            train_candidates.append(t)
    if len(train_candidates) == 0:
        train_candidates = list(range(train_split.start, min(train_split.end - episode_len - 1, train_split.start + 4096)))

    def _lam_schedule(it: int) -> float:
        if tc_warmup_steps and tc_warmup_steps > 0:
            warm = min(1.0, it / float(tc_warmup_steps))
        else:
            warm = 1.0

        lam = float(lam_tc) * warm

        if tc_cooldown_steps and tc_cooldown_steps > 0 and iters > 1:
            start_cd = max(1, iters - int(tc_cooldown_steps))
            if it >= start_cd:
                t = (it - start_cd) / float(max(1, iters - start_cd))
                lam_end = float(lam_tc) * float(tc_final_mult)
                lam = lam * (1.0 - t) + lam_end * t

        return float(lam)

    def _turn_clip_schedule(it: int) -> float:
        if turnover_clip_warmup_steps and turnover_clip_warmup_steps > 0:
            a = min(1.0, it / float(turnover_clip_warmup_steps))
        else:
            a = 1.0
        return float(turnover_clip_start) * (1.0 - a) + float(turnover_clip_end) * a

    def _eval_split_with_extras(split: Split, n_paths: int = 512) -> Dict[str, float]:
        start_candidates = []
        for t in range(split.start, split.end - episode_len - 1):
            if valid_mask[t]:
                start_candidates.append(t)
        if len(start_candidates) == 0:
            start_candidates = list(range(split.start, min(split.end - episode_len - 1, split.start + 2048)))
        if len(start_candidates) == 0:
            raise ValueError(f"No valid start indices for split={split.name}")

        pick = np.random.choice(start_candidates, size=min(n_paths, len(start_candidates)), replace=False)
        start_idx = torch.tensor(pick, device=device, dtype=torch.long)

        out = simulate_batch_policy(
            model=model,
            returns_t=returns_t,
            rf_t=rf_t,
            belief_t=belief_t,
            start_idx=start_idx,
            episode_len=episode_len,
            device=device,
            wnorm=wnorm,
            bnorm=bnorm,
            gamma=gamma,
            lambda_tc=float(lam_tc),  # eval uses full lambda (not scheduled)
            W0=W0,
            W_min=W_min,
            W_max=W_max,
            rho_max=rho_max,
            pi_anchor_beta=pi_anchor_beta,
            pi_entropy_beta=pi_entropy_beta,
            record_paths=False,
            turnover_rescale=float(turnover_rescale),
            turnover_clip=float(_turn_clip_schedule(it=iters)),  # eval at steady clip
            tc_cost_frac_clip=float(tc_cost_frac_clip),
            belief_mu_dim=None,
            belief_sigma_floor=float(belief_sigma_floor),
            belief_sigma_clip=float(belief_sigma_clip),
            belief_shrink_kappa=float(belief_shrink_kappa),
            belief_z_clip=float(belief_z_clip),
        )

        WT = out["WT"].detach().cpu().numpy()
        obj = safe_float(out["obj"].detach().cpu())
        avg_turn = safe_float(out["avg_turnover"].detach().cpu())
        clamp_rate = safe_float(out["clamp_rate"].detach().cpu())

        if abs(gamma - 1.0) < 1e-12:
            CE = float(np.exp(np.mean(np.log(np.maximum(WT, 1e-12)))))
        else:
            EU = np.mean((np.maximum(WT, 1e-12) ** (1.0 - gamma)) / (1.0 - gamma))
            CE = float(((1.0 - gamma) * EU) ** (1.0 / (1.0 - gamma)))

        W_T_mean = float(np.mean(WT))

        return {
            "obj": obj,
            "CE_term": CE,
            "W_T_mean": W_T_mean,
            "avg_TC": avg_turn,         # keep legacy key name for prints
            "clamp_rate": clamp_rate,
        }

    for it in range(1, iters + 1):
        model.train()
        pick = np.random.choice(train_candidates, size=batch_size, replace=True)
        start_idx = torch.tensor(pick, device=device, dtype=torch.long)

        lam_eff = _lam_schedule(it)
        turn_clip = _turn_clip_schedule(it)

        # recommend: kalman only
        shrink_kappa = float(belief_shrink_kappa) if (belief_name == "kalman") else 0.0

        out = simulate_batch_policy(
            model=model,
            returns_t=returns_t,
            rf_t=rf_t,
            belief_t=belief_t,
            start_idx=start_idx,
            episode_len=episode_len,
            device=device,
            wnorm=wnorm,
            bnorm=bnorm,
            gamma=gamma,
            lambda_tc=lam_eff,
            W0=W0,
            W_min=W_min,
            W_max=W_max,
            rho_max=rho_max,
            pi_anchor_beta=pi_anchor_beta,
            pi_entropy_beta=pi_entropy_beta,
            record_paths=False,
            turnover_rescale=float(turnover_rescale),
            turnover_clip=float(turn_clip),
            tc_cost_frac_clip=float(tc_cost_frac_clip),
            belief_mu_dim=None,
            belief_sigma_floor=float(belief_sigma_floor),
            belief_sigma_clip=float(belief_sigma_clip),
            belief_shrink_kappa=shrink_kappa,
            belief_z_clip=float(belief_z_clip),
        )

        loss = out["obj"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if it % eval_every == 0 or it == 1:
            model.eval()

            tr_metrics = _eval_split_with_extras(splits["train"], n_paths=512)
            val_metrics = _eval_split_with_extras(splits["val"], n_paths=512)

            print(
                f"[TRAIN] it={it:5d} obj={tr_metrics['obj']: .6f} CE={tr_metrics['CE_term']: .6f} "
                f"W_T_mean={tr_metrics['W_T_mean']: .6f} avg_TC={tr_metrics['avg_TC']: .6f} clamp={tr_metrics['clamp_rate']: .4f}"
            )
            print(
                f"[VAL]   it={it:5d} obj={val_metrics['obj']: .6f} CE={val_metrics['CE_term']: .6f} "
                f"W_T_mean={val_metrics['W_T_mean']: .6f} avg_TC={val_metrics['avg_TC']: .6f} clamp={val_metrics['clamp_rate']: .4f}"
            )

            # --- strengthened checkpoint criterion ---
            val_score = (
                float(val_metrics["CE_term"])
                - float(ckpt_turnover_weight) * float(val_metrics["avg_TC"])
                - float(ckpt_clamp_weight) * float(val_metrics["clamp_rate"])
            )

            improved = (val_score > best_val_score + float(ckpt_eps))

            if improved:
                best_val_score = val_score
                best_val_CE = float(val_metrics["CE_term"])
                best_it = it
                no_improve = 0

                meta = {
                    "seed": seed,
                    "belief_name": belief_name,
                    "lambda_tc": float(lam_tc),
                    "best_it": int(best_it),
                    "best_val_CE": float(best_val_CE),
                    "best_val_score": float(best_val_score),
                    "ckpt_turnover_weight": float(ckpt_turnover_weight),
                    "ckpt_clamp_weight": float(ckpt_clamp_weight),
                    "val_avg_turnover": float(val_metrics["avg_TC"]),
                    "val_clamp_rate": float(val_metrics["clamp_rate"]),
                    "timestamp": now_str(),
                    "schedule": {
                        "tc_warmup_steps": int(tc_warmup_steps),
                        "tc_cooldown_steps": int(tc_cooldown_steps),
                        "tc_final_mult": float(tc_final_mult),
                        "turnover_rescale": float(turnover_rescale),
                        "turnover_clip_start": float(turnover_clip_start),
                        "turnover_clip_end": float(turnover_clip_end),
                        "turnover_clip_warmup_steps": int(turnover_clip_warmup_steps),
                        "tc_cost_frac_clip": float(tc_cost_frac_clip),
                    },
                    "belief_stabilization": {
                        "belief_sigma_floor": float(belief_sigma_floor),
                        "belief_sigma_clip": float(belief_sigma_clip),
                        "belief_shrink_kappa": float(belief_shrink_kappa if belief_name == "kalman" else 0.0),
                        "belief_z_clip": float(belief_z_clip),
                    },
                }
                save_checkpoint_v2(best_path, model, wnorm, bnorm, meta)
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(
                        f"[EARLY STOP] it={it} (no VAL improvement for {early_stop_patience} eval steps). "
                        f"best_it={best_it}, best_val_score={best_val_score: .6f}, best_val_CE={best_val_CE: .6f}"
                    )
                    break

    return best_path, {
        "best_val_CE": float(best_val_CE),
        "best_val_score": float(best_val_score),
        "best_it": int(best_it),
        "run_dir": run_dir,
    }

# =============================================================================
# Aggregation / CI / Tables / Figure
# =============================================================================

def mean_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x))
    if len(x) <= 1:
        return (m, m, m)
    s = float(np.std(x, ddof=1))
    z = 1.96  # approx 95%
    half = z * s / math.sqrt(len(x))
    return (m, m - half, m + half)


def write_latex_table_main_ci(df: pd.DataFrame, out_path: str, caption: str):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Train CE (CI) & Val CE (CI) & Test CE (CI) \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        method = row["method"]
        tr = f"{row['train_CE_mean']:.4f} [{row['train_CE_lo']:.4f},{row['train_CE_hi']:.4f}]"
        va = f"{row['val_CE_mean']:.4f} [{row['val_CE_lo']:.4f},{row['val_CE_hi']:.4f}]"
        te = f"{row['test_CE_mean']:.4f} [{row['test_CE_lo']:.4f},{row['test_CE_hi']:.4f}]"
        lines.append(f"{method} & {tr} & {va} & {te} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{table}")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_latex_table_lambda_ci(df: pd.DataFrame, out_path: str, caption: str):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Belief & $\lambda_{tc}$ & Val CE (CI) & Test CE (CI) & Avg Turnover (CI) \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        belief = row["belief"]
        lam = row["lambda_tc"]
        va = f"{row['val_CE_mean']:.4f} [{row['val_CE_lo']:.4f},{row['val_CE_hi']:.4f}]"
        te = f"{row['test_CE_mean']:.4f} [{row['test_CE_lo']:.4f},{row['test_CE_hi']:.4f}]"
        tc = f"{row['test_TC_mean']:.4f} [{row['test_TC_lo']:.4f},{row['test_TC_hi']:.4f}]"
        lines.append(f"{belief} & {lam} & {va} & {te} & {tc} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{table}")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_frontier_ci(df: pd.DataFrame, out_prefix: str, title: str):
    """
    Scatter of (turnover_mean, test_CE_mean) for policy runs, with error bars.
    """
    ensure_dir(os.path.dirname(out_prefix))

    plt.figure(figsize=(7.5, 5.5))
    for _, row in df.iterrows():
        x = row["test_TC_mean"]
        y = row["test_CE_mean"]
        xerr = [[x - row["test_TC_lo"]], [row["test_TC_hi"] - x]]
        yerr = [[y - row["test_CE_lo"]], [row["test_CE_hi"] - y]]
        plt.errorbar([x], [y], xerr=xerr, yerr=yerr, fmt="o", capsize=3)

    plt.xlabel("Average Turnover (mean ± CI)")
    plt.ylabel("Test CE (mean ± CI)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=200)
    plt.savefig(out_prefix + ".pdf")
    plt.close()


# =============================================================================
# Main pipeline
# =============================================================================


def main():
    warnings.filterwarnings('ignore')

    # ========================================================================
    # SELECT PRESET (CRITICAL FIX)
    # ========================================================================
    PRESET = 'balanced'  # Change to: conservative, balanced, or exploration
    preset = get_parameter_preset(PRESET)
    print('='*80)
    print(f'USING PRESET: {PRESET}')
    print('='*80)
    for k, v in preset.items():
        print(f'  {k}: {v}')
    print()

    warnings.filterwarnings("ignore")

    # -----------------------------
    # USER CONFIG (edit as needed)
    # -----------------------------
    csv_path = "./paperA_timeseries (1).csv"

    assets = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'MS', 'TSLA', 'AVGO', 'BRK.B', 'LLY',
              'WMT', 'JPM', 'V', 'ORCL', 'MA', 'JNJ', 'XOM', 'NFLX', 'KO', 'IBM']

    out_dir = "./paperA_runs_v2"

    # dataset split indices (user-provided)
    train_end = 2622
    val_end = 3184
    test_end = 3747

    # beliefs
    belief_names = ["rolling", "kalman"]
    rolling_window = 60

    # seeds and lambda sweep
    # CRITICAL: Multiple seeds for statistical validity
    seeds = list(range(10))  # Changed from [0] to 0-9
    lambda_sweep = [3e-4, 1e-3, 3e-3]
    lambda_main = 1e-3

    # model + training
    episode_len = 60
    iters = 1400
    batch_size = preset['batch_size']  # From preset
    lr = preset['lr']  # From preset (was: 3e-4)
    eval_every = 200
    early_stop_patience = preset['early_stop_patience']  # From preset

    # preference
    gamma = 3.0
    W0 = 1.0
    W_min = 1e-6
    W_max = 1e6

    # consumption disabled by default (set >0 to allow)
    rho_max = 0.0

    # regularizers
    pi_anchor_beta = preset['pi_anchor_beta']  # From preset (was: 1e-3)
    pi_entropy_beta = preset['pi_entropy_beta']  # From preset (was: 1e-4)

    # -----------------------------
    # NEW: turnover/tc schedule + clip + kalman stability + ckpt score
    # -----------------------------
    tc_warmup_steps = preset['tc_warmup_steps']  # From preset (was: 300)
    tc_cooldown_steps = 0
    tc_final_mult = 1.0

    turnover_rescale = 0.5
    turnover_clip_start = preset['turnover_clip_start']  # From preset
    turnover_clip_end = preset['turnover_clip_end']  # From preset
    turnover_clip_warmup_steps = 300
    tc_cost_frac_clip = 0.25

    belief_sigma_floor = 1e-6
    belief_sigma_clip = preset['belief_sigma_clip']  # From preset
    belief_shrink_kappa_kalman = preset['belief_shrink_kappa_kalman']  # From preset
    belief_z_clip = preset['belief_z_clip']  # From preset

    ckpt_turnover_weight = preset['ckpt_turnover_weight']  # From preset (was: 0.05)
    ckpt_clamp_weight = preset['ckpt_clamp_weight']  # From preset (was: 0.50)

    # baselines
    baseline_names = ["rf_only", "ew_bh", "ew_rebal"]
    lambda_tc_for_rebal = lambda_main  # cost for baseline rebal

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================================================================================")
    print("ULTIMATE MASTER PIPELINE (v2)")
    print("================================================================================")
    print(f"Device: {device}")
    print(f"Start time: {now_str()}")
    print("================================================================================\n")

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "tables"))
    ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "runs"))

    # Save run config
    run_config = {
        "csv_path": csv_path,
        "assets": assets,
        "out_dir": out_dir,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "belief_names": belief_names,
        "rolling_window": rolling_window,
        "seeds": seeds,
        "lambda_sweep": lambda_sweep,
        "lambda_main": lambda_main,
        "episode_len": episode_len,
        "iters": iters,
        "batch_size": batch_size,
        "lr": lr,
        "eval_every": eval_every,
        "early_stop_patience": early_stop_patience,
        "gamma": gamma,
        "W0": W0,
        "W_min": W_min,
        "W_max": W_max,
        "rho_max": rho_max,
        "pi_anchor_beta": pi_anchor_beta,
        "pi_entropy_beta": pi_entropy_beta,
        "baseline_names": baseline_names,
        "lambda_tc_for_rebal": float(lambda_tc_for_rebal),
        "timestamp": now_str(),
        "new": {
            "tc_warmup_steps": int(tc_warmup_steps),
            "tc_cooldown_steps": int(tc_cooldown_steps),
            "tc_final_mult": float(tc_final_mult),
            "turnover_rescale": float(turnover_rescale),
            "turnover_clip_start": float(turnover_clip_start),
            "turnover_clip_end": float(turnover_clip_end),
            "turnover_clip_warmup_steps": int(turnover_clip_warmup_steps),
            "tc_cost_frac_clip": float(tc_cost_frac_clip),
            "belief_sigma_floor": float(belief_sigma_floor),
            "belief_sigma_clip": float(belief_sigma_clip),
            "belief_shrink_kappa_kalman": float(belief_shrink_kappa_kalman),
            "belief_z_clip": float(belief_z_clip),
            "ckpt_turnover_weight": float(ckpt_turnover_weight),
            "ckpt_clamp_weight": float(ckpt_clamp_weight),
        }
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # Load data
    df, returns_np, rf_np, dates = load_timeseries_csv(csv_path, assets)
    T, d = returns_np.shape
    splits = make_splits(T, train_end=train_end, val_end=val_end, test_end=test_end)

    # Print date ranges
    tr0, tr1 = split_date_range(dates, splits["train"])
    va0, va1 = split_date_range(dates, splits["val"])
    te0, te1 = split_date_range(dates, splits["test"])
    print(f"N={T} | split train=({splits['train'].start}, {splits['train'].end}) "
          f"val=({splits['val'].start}, {splits['val'].end}) test=({splits['test'].start}, {splits['test'].end})")
    print(f"Date ranges | train={tr0}~{tr1} | val={va0}~{va1} | test={te0}~{te1}\n")

    # Torch tensors
    returns_t = torch.tensor(returns_np, device=device, dtype=torch.float32)
    rf_t = torch.tensor(rf_np, device=device, dtype=torch.float32)

    results_rows = []

    # -----------------------------
    # Baselines first
    # -----------------------------
    for seed in seeds:
        set_seed(seed)
        print(f"[BASELINES] seed={seed} lambda_tc_for_rebal={lambda_tc_for_rebal}")
        for belief_name in belief_names:
            if belief_name == "rolling":
                beliefs_np = compute_rolling_beliefs(returns_np, window=rolling_window)
                valid_mask = np.zeros((T,), dtype=bool)
                valid_mask[rolling_window:] = True
            elif belief_name == "kalman":
                beliefs_np = compute_kalman_beliefs(returns_np)
                valid_mask = np.ones((T,), dtype=bool)
            else:
                raise ValueError(belief_name)

            belief_t = torch.tensor(beliefs_np, device=device, dtype=torch.float32)

            for bname in baseline_names:
                m_train = eval_split_baseline(
                    returns_t=returns_t,
                    rf_t=rf_t,
                    split=splits["train"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    gamma=gamma,
                    lambda_tc_for_rebal=(lambda_tc_for_rebal if bname == "ew_rebal" else 0.0),
                    baseline_name=bname,
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    n_paths=512,
                )
                m_val = eval_split_baseline(
                    returns_t=returns_t,
                    rf_t=rf_t,
                    split=splits["val"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    gamma=gamma,
                    lambda_tc_for_rebal=(lambda_tc_for_rebal if bname == "ew_rebal" else 0.0),
                    baseline_name=bname,
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    n_paths=512,
                )
                m_test = eval_split_baseline(
                    returns_t=returns_t,
                    rf_t=rf_t,
                    split=splits["test"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    gamma=gamma,
                    lambda_tc_for_rebal=(lambda_tc_for_rebal if bname == "ew_rebal" else 0.0),
                    baseline_name=bname,
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    n_paths=512,
                )

                print(f"[BASELINE] {bname} belief={belief_name} seed={seed} "
                      f"lambda_tc_for_rebal={(lambda_tc_for_rebal if bname == 'ew_rebal' else 0.0)}")
                print(f"[TRAIN] obj={m_train['obj']: .6f} CE={m_train['CE_term']: .6f} "
                      f"W_T_mean={m_train['W_T_mean']: .6f} avg_TC={m_train['avg_TC']: .6f}")
                print(f"[VAL]   obj={m_val['obj']: .6f} CE={m_val['CE_term']: .6f} "
                      f"W_T_mean={m_val['W_T_mean']: .6f} avg_TC={m_val['avg_TC']: .6f}")
                print(f"[TEST]  obj={m_test['obj']: .6f} CE={m_test['CE_term']: .6f} "
                      f"W_T_mean={m_test['W_T_mean']: .6f} avg_TC={m_test['avg_TC']: .6f}\n")

                results_rows.append({
                    "d": d,
                    "method": f"baseline_{bname}",
                    "belief": belief_name,
                    "lambda_tc": 0.0 if bname != "ew_rebal" else float(lambda_tc_for_rebal),
                    "seed": seed,
                    "split": "train",
                    "obj": m_train["obj"],
                    "CE_term": m_train["CE_term"],
                    "W_T_mean": m_train["W_T_mean"],
                    "avg_TC": m_train["avg_TC"],
                })
                results_rows.append({
                    "d": d,
                    "method": f"baseline_{bname}",
                    "belief": belief_name,
                    "lambda_tc": 0.0 if bname != "ew_rebal" else float(lambda_tc_for_rebal),
                    "seed": seed,
                    "split": "val",
                    "obj": m_val["obj"],
                    "CE_term": m_val["CE_term"],
                    "W_T_mean": m_val["W_T_mean"],
                    "avg_TC": m_val["avg_TC"],
                })
                results_rows.append({
                    "d": d,
                    "method": f"baseline_{bname}",
                    "belief": belief_name,
                    "lambda_tc": 0.0 if bname != "ew_rebal" else float(lambda_tc_for_rebal),
                    "seed": seed,
                    "split": "test",
                    "obj": m_test["obj"],
                    "CE_term": m_test["CE_term"],
                    "W_T_mean": m_test["W_T_mean"],
                    "avg_TC": m_test["avg_TC"],
                })

    # -----------------------------
    # Policy runs: belief x lambda sweep x seed
    # -----------------------------
    for belief_name in belief_names:
        if belief_name == "rolling":
            beliefs_np = compute_rolling_beliefs(returns_np, window=rolling_window)
            valid_mask = np.zeros((T,), dtype=bool)
            valid_mask[rolling_window:] = True
            belief_shrink_kappa = 0.0
        elif belief_name == "kalman":
            beliefs_np = compute_kalman_beliefs(returns_np)
            valid_mask = np.ones((T,), dtype=bool)
            belief_shrink_kappa = float(belief_shrink_kappa_kalman)
        else:
            raise ValueError(belief_name)

        belief_t = torch.tensor(beliefs_np, device=device, dtype=torch.float32)

        for lam_tc in lambda_sweep:
            for seed in seeds:
                print(f"[RUN] belief={belief_name} lambda_tc={lam_tc} seed={seed} device={device}")

                model = DeepONetPolicy(
                    trunk_in_dim=2,
                    branch_in_dim=beliefs_np.shape[1],
                    d_assets=d,
                    width=256,
                    depth=3,
                    act="silu",
                )

                best_path, run_meta = train_one_run(
                    model=model,
                    returns_t=returns_t,
                    rf_t=rf_t,
                    belief_t=belief_t,
                    splits=splits,
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    seed=seed,
                    out_dir=out_dir,
                    belief_name=belief_name,
                    lam_tc=float(lam_tc),
                    lr=lr,
                    iters=iters,
                    batch_size=batch_size,
                    gamma=gamma,
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    rho_max=rho_max,
                    eval_every=eval_every,
                    early_stop_patience=early_stop_patience,
                    pi_anchor_beta=pi_anchor_beta,
                    pi_entropy_beta=pi_entropy_beta,
                    tc_warmup_steps=int(tc_warmup_steps),
                    tc_cooldown_steps=int(tc_cooldown_steps),
                    tc_final_mult=float(tc_final_mult),
                    turnover_rescale=float(turnover_rescale),
                    turnover_clip_start=float(turnover_clip_start),
                    turnover_clip_end=float(turnover_clip_end),
                    turnover_clip_warmup_steps=int(turnover_clip_warmup_steps),
                    tc_cost_frac_clip=float(tc_cost_frac_clip),
                    belief_sigma_floor=float(belief_sigma_floor),
                    belief_sigma_clip=float(belief_sigma_clip),
                    belief_shrink_kappa=float(belief_shrink_kappa),
                    belief_z_clip=float(belief_z_clip),
                    ckpt_turnover_weight=float(ckpt_turnover_weight),
                    ckpt_clamp_weight=float(ckpt_clamp_weight),
                )
                print(f"[CKPT] {best_path}")

                # Final eval from best checkpoint
                model2 = DeepONetPolicy(
                    trunk_in_dim=2,
                    branch_in_dim=beliefs_np.shape[1],
                    d_assets=d,
                    width=256,
                    depth=3,
                    act="silu",
                ).to(device)
                wnorm, bnorm, ckpt_meta = load_checkpoint_v2(best_path, model2, device)

                print("---- Final Evaluation (best-val checkpoint) ----")
                m_train = eval_split_policy(
                    model=model2,
                    returns_t=returns_t,
                    rf_t=rf_t,
                    belief_t=belief_t,
                    split=splits["train"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    wnorm=wnorm,
                    bnorm=bnorm,
                    gamma=gamma,
                    lambda_tc=float(lam_tc),
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    rho_max=rho_max,
                    pi_anchor_beta=pi_anchor_beta,
                    pi_entropy_beta=pi_entropy_beta,
                    n_paths=512,
                )
                m_val = eval_split_policy(
                    model=model2,
                    returns_t=returns_t,
                    rf_t=rf_t,
                    belief_t=belief_t,
                    split=splits["val"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    wnorm=wnorm,
                    bnorm=bnorm,
                    gamma=gamma,
                    lambda_tc=float(lam_tc),
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    rho_max=rho_max,
                    pi_anchor_beta=pi_anchor_beta,
                    pi_entropy_beta=pi_entropy_beta,
                    n_paths=512,
                )
                m_test = eval_split_policy(
                    model=model2,
                    returns_t=returns_t,
                    rf_t=rf_t,
                    belief_t=belief_t,
                    split=splits["test"],
                    valid_mask=valid_mask,
                    episode_len=episode_len,
                    device=device,
                    wnorm=wnorm,
                    bnorm=bnorm,
                    gamma=gamma,
                    lambda_tc=float(lam_tc),
                    W0=W0,
                    W_min=W_min,
                    W_max=W_max,
                    rho_max=rho_max,
                    pi_anchor_beta=pi_anchor_beta,
                    pi_entropy_beta=pi_entropy_beta,
                    n_paths=512,
                )

                print(f"[TRAIN] obj={m_train['obj']: .6f} CE={m_train['CE_term']: .6f} "
                      f"W_T_mean={m_train['W_T_mean']: .6f} avg_TC={m_train['avg_TC']: .6f}")
                print(f"[VAL]   obj={m_val['obj']: .6f} CE={m_val['CE_term']: .6f} "
                      f"W_T_mean={m_val['W_T_mean']: .6f} avg_TC={m_val['avg_TC']: .6f}")
                print(f"[TEST]  obj={m_test['obj']: .6f} CE={m_test['CE_term']: .6f} "
                      f"W_T_mean={m_test['W_T_mean']: .6f} avg_TC={m_test['avg_TC']: .6f}\n")

                results_rows.append({
                    "d": d,
                    "method": "policy_deeponet",
                    "belief": belief_name,
                    "lambda_tc": float(lam_tc),
                    "seed": seed,
                    "split": "train",
                    "obj": m_train["obj"],
                    "CE_term": m_train["CE_term"],
                    "W_T_mean": m_train["W_T_mean"],
                    "avg_TC": m_train["avg_TC"],
                    "best_it": run_meta.get("best_it", -1),
                    "best_val_CE": run_meta.get("best_val_CE", np.nan),
                    "best_val_score": run_meta.get("best_val_score", np.nan),
                    "ckpt_path": best_path,
                })
                results_rows.append({
                    "d": d,
                    "method": "policy_deeponet",
                    "belief": belief_name,
                    "lambda_tc": float(lam_tc),
                    "seed": seed,
                    "split": "val",
                    "obj": m_val["obj"],
                    "CE_term": m_val["CE_term"],
                    "W_T_mean": m_val["W_T_mean"],
                    "avg_TC": m_val["avg_TC"],
                    "best_it": run_meta.get("best_it", -1),
                    "best_val_CE": run_meta.get("best_val_CE", np.nan),
                    "best_val_score": run_meta.get("best_val_score", np.nan),
                    "ckpt_path": best_path,
                })
                results_rows.append({
                    "d": d,
                    "method": "policy_deeponet",
                    "belief": belief_name,
                    "lambda_tc": float(lam_tc),
                    "seed": seed,
                    "split": "test",
                    "obj": m_test["obj"],
                    "CE_term": m_test["CE_term"],
                    "W_T_mean": m_test["W_T_mean"],
                    "avg_TC": m_test["avg_TC"],
                    "best_it": run_meta.get("best_it", -1),
                    "best_val_CE": run_meta.get("best_val_CE", np.nan),
                    "best_val_score": run_meta.get("best_val_score", np.nan),
                    "ckpt_path": best_path,
                })

    # Save per-split results by seed
    df_res = pd.DataFrame(results_rows)
    by_seed_path = os.path.join(out_dir, "results_by_seed.csv")
    df_res.to_csv(by_seed_path, index=False, encoding="utf-8-sig")
    print(f"[FILES] wrote: {by_seed_path}")

    # -----------------------------
    # NEW (B): Win-rate / CI table generation (after results_by_seed.csv)
    # -----------------------------
    def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
        if n <= 0:
            return (0.0, 0.0, 0.0)
        phat = k / n
        denom = 1.0 + (z ** 2) / n
        center = (phat + (z ** 2) / (2 * n)) / denom
        half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z ** 2) / (4 * n ** 2))
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        return (phat, lo, hi)

    def write_latex_table_winrate(df_win: pd.DataFrame, out_path: str, caption: str):
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lllc}")
        lines.append(r"\toprule")
        lines.append(r"Belief & $\lambda_{tc}$ & Baseline & Win-rate (95\% CI) \\")
        lines.append(r"\midrule")
        for _, row in df_win.iterrows():
            belief = row["belief"]
            lam = row["lambda_tc"]
            base = row["baseline"]
            s = f"{row['win_rate']:.3f} [{row['win_lo']:.3f},{row['win_hi']:.3f}]"
            lines.append(f"{belief} & {lam} & {base} & {s} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(r"\end{table}")
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    df_test = df_res[df_res["split"] == "test"].copy()

    win_rows = []
    policy_test = df_test[df_test["method"] == "policy_deeponet"][["d", "belief", "lambda_tc", "seed", "CE_term"]].rename(
        columns={"CE_term": "CE_policy"}
    )

    for belief_name in belief_names:
        for lam_tc in lambda_sweep:
            pol = policy_test[(policy_test["belief"] == belief_name) & (policy_test["lambda_tc"] == float(lam_tc))].copy()
            if pol.empty:
                continue

            for bname in baseline_names:
                base_method = f"baseline_{bname}"
                base = df_test[(df_test["method"] == base_method) & (df_test["belief"] == belief_name)][
                    ["d", "belief", "seed", "CE_term"]
                ].rename(columns={"CE_term": "CE_base"})

                merged = pd.merge(pol, base, on=["d", "belief", "seed"], how="inner")
                if merged.empty:
                    continue

                wins = (merged["CE_policy"].values > merged["CE_base"].values + 1e-12).astype(np.int64)
                k = int(wins.sum())
                n = int(len(wins))
                wr, lo, hi = wilson_ci(k, n, z=1.96)

                win_rows.append({
                    "belief": belief_name,
                    "lambda_tc": float(lam_tc),
                    "baseline": base_method,
                    "n": n,
                    "wins": k,
                    "win_rate": wr,
                    "win_lo": lo,
                    "win_hi": hi,
                })

    df_win = pd.DataFrame(win_rows).sort_values(["belief", "lambda_tc", "baseline"])
    win_csv = os.path.join(out_dir, "winrate_ci.csv")
    df_win.to_csv(win_csv, index=False, encoding="utf-8-sig")
    print(f"[WIN] wrote: {win_csv}")

    win_tex = os.path.join(out_dir, "tables", "table_winrate_ci.tex")
    write_latex_table_winrate(df_win, win_tex, caption="Win-rate of policy vs baselines on Test CE (Wilson 95\\% CI across seeds).")
    print(f"[TABLES] wrote: {win_tex}")

    # -----------------------------
    # Aggregate CI tables (existing)
    # -----------------------------
    agg_rows = []
    for (method, belief, lam, split), g in df_res.groupby(["method", "belief", "lambda_tc", "split"]):
        ce = g["CE_term"].values.astype(np.float64)
        tc = g["avg_TC"].values.astype(np.float64)
        ce_m, ce_lo, ce_hi = mean_ci(ce)
        tc_m, tc_lo, tc_hi = mean_ci(tc)

        agg_rows.append({
            "method": method,
            "belief": belief,
            "lambda_tc": lam,
            "split": split,
            "CE_mean": ce_m,
            "CE_lo": ce_lo,
            "CE_hi": ce_hi,
            "TC_mean": tc_m,
            "TC_lo": tc_lo,
            "TC_hi": tc_hi,
            "n": len(g),
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_path = os.path.join(out_dir, "results_aggregated_ci.csv")
    df_agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    print(f"[AGG] wrote: {agg_path}")

    # Table main
    main_rows = []
    for method in sorted(df_res["method"].unique()):
        if method == "policy_deeponet":
            sub = df_agg[(df_agg["method"] == method) & (df_agg["lambda_tc"] == float(lambda_main))]
        else:
            sub = df_agg[df_agg["method"] == method]

        for belief in sorted(df_res["belief"].unique()):
            sub_b = sub[sub["belief"] == belief]
            if sub_b.empty:
                continue

            def row_for(split_name: str):
                x = sub_b[sub_b["split"] == split_name]
                if x.empty:
                    return (np.nan, np.nan, np.nan)
                return (float(x["CE_mean"].values[0]), float(x["CE_lo"].values[0]), float(x["CE_hi"].values[0]))

            tr = row_for("train")
            va = row_for("val")
            te = row_for("test")

            main_rows.append({
                "method": f"{method} ({belief})",
                "train_CE_mean": tr[0], "train_CE_lo": tr[1], "train_CE_hi": tr[2],
                "val_CE_mean": va[0], "val_CE_lo": va[1], "val_CE_hi": va[2],
                "test_CE_mean": te[0], "test_CE_lo": te[1], "test_CE_hi": te[2],
            })

    df_main = pd.DataFrame(main_rows)
    table_main_path = os.path.join(out_dir, "tables", "table_main_ci.tex")
    write_latex_table_main_ci(df_main, table_main_path, caption="Terminal-wealth CE with 95\\% CI (by seed).")
    print(f"[TABLES] wrote: {table_main_path}")

    # Lambda table: for policy only
    lam_rows = []
    sub = df_agg[df_agg["method"] == "policy_deeponet"]
    for (belief, lam), g in sub[sub["split"].isin(["val", "test"])].groupby(["belief", "lambda_tc"]):
        val = g[g["split"] == "val"]
        test = g[g["split"] == "test"]
        if val.empty or test.empty:
            continue
        lam_rows.append({
            "belief": belief,
            "lambda_tc": lam,
            "val_CE_mean": float(val["CE_mean"].values[0]),
            "val_CE_lo": float(val["CE_lo"].values[0]),
            "val_CE_hi": float(val["CE_hi"].values[0]),
            "test_CE_mean": float(test["CE_mean"].values[0]),
            "test_CE_lo": float(test["CE_lo"].values[0]),
            "test_CE_hi": float(test["CE_hi"].values[0]),
            "test_TC_mean": float(test["TC_mean"].values[0]),
            "test_TC_lo": float(test["TC_lo"].values[0]),
            "test_TC_hi": float(test["TC_hi"].values[0]),
        })

    df_lam = pd.DataFrame(lam_rows).sort_values(["belief", "lambda_tc"])
    table_lam_path = os.path.join(out_dir, "tables", "table_lambda_ci.tex")
    write_latex_table_lambda_ci(df_lam, table_lam_path, caption="Policy CE/turnover by $\\lambda_{tc}$ with 95\\% CI.")
    print(f"[TABLES] wrote: {table_lam_path}")

    # Frontier fig (policy only)
    df_front = df_lam.copy()
    fig_prefix = os.path.join(out_dir, "figures", "fig_frontier_ci")
    plot_frontier_ci(df_front, fig_prefix, title="Policy Frontier: Test CE vs Turnover (CI)")
    print(f"[FIG] wrote: {fig_prefix}.*")

    print("\n[DONE] out_dir=", out_dir)
    print("[FILES] results_by_seed.csv, winrate_ci.csv, results_aggregated_ci.csv, tables/*.tex, figures/*.png/*.pdf, run_config.json")


if __name__ == "__main__":
    main()
