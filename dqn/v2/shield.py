"""Action masks built from the per-action lookahead features.

Two levels, both computed only from the feature matrix (no extra simulation):

1. Death mask (always on in v2): never pick a move that dies immediately. The env already tells us this
   (feature `would_die`), so letting the agent choose it only wastes episodes and adds noise to the targets.

2. Safety shield (OPTIONAL, --shield): among the non-lethal moves, keep only those after which the snake can
   still escape (time-aware BFS reaches a body cell after it frees up, i.e. the head can chase its tail).
   If no move is safe, fall back to the non-lethal moves with the largest reachable area (the "least bad"
   ones). The Q-network still chooses *which* safe move to take; the shield only removes moves that lead to a
   provable trap. This is a form of "shielded RL" (Alshiekh et al., 2018) and is reported as an ablation.
"""
import numpy as np

from .state import F


def death_mask(features):
    """bool (3,): True = allowed. If every move dies, allow all (the episode ends anyway)."""
    mask = features[:, F["would_die"]] < 0.5
    if not mask.any():
        mask[:] = True
    return mask


def shield_mask(features):
    """bool (3,): safety-shield mask (subset of death_mask)."""
    alive = death_mask(features)
    if features[:, F["would_die"]].min() > 0.5:
        return alive
    safe = alive & (features[:, F["can_escape"]] > 0.5)
    if safe.any():
        return safe
    # No escape route anywhere: keep the moves with the most room.
    space = np.where(alive, features[:, F["reachable_fraction"]], -1.0)
    return space >= space.max() - 1e-6


def action_mask(features, use_shield):
    return shield_mask(features) if use_shield else death_mask(features)
