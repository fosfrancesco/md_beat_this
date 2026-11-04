"""
Masked diffusion inference: iterative unmasking of the beat and downbeat sequences.
"""

import math

import torch
import torch.nn.functional as F

from md_beat_this.model.beat_tracker import EVENT, MASK, NO_EVENT, PAD


@torch.no_grad()
def unmask(models, spect, steps=8, tolerance=3, prefix=None):
    """
    Predict the beat and downbeat sequences of a spectrogram excerpt by iterative unmasking.

    All tokens start masked. At each step, the logits (averaged over the ensemble) are
    peak-picked into event predictions, and a share of the still-masked frames is committed:
    the most confident ones, with separate budgets for predicted events and non-events so
    that both are drained at the same rate. The neighbours of each committed event within
    `tolerance` frames are committed as non-events. Remaining masked frames take the
    predictions of the last step. Trailing all-zero frames are treated as padding.

    Args:
        models (list): The models of the ensemble.
        spect (torch.Tensor): Spectrogram of shape (1, time, bins).
        steps (int): The number of unmasking steps.
        tolerance (int): Minimum distance in frames between two events.
        prefix (tuple, optional): Boolean beat and downbeat sequences that fix the
            first frames of the excerpt, which are then never masked.

    Returns:
        tuple: Boolean beat and downbeat sequences of shape (time,).
    """
    length = spect.shape[1]
    tokens = {
        ch: torch.full((1, length), MASK, dtype=torch.long, device=spect.device)
        for ch in ("beat", "downbeat")
    }
    masked = {ch: torch.ones(length, dtype=torch.bool, device=spect.device) for ch in tokens}
    if prefix is not None:
        for ch, fixed in zip(tokens, prefix):
            tokens[ch][0, : len(fixed)] = torch.where(fixed, EVENT, NO_EVENT)
            masked[ch][: len(fixed)] = False
    nonzero = (spect[0] != 0).any(-1)
    if nonzero.any():
        n_valid = nonzero.nonzero().max().item() + 1
        for ch in tokens:
            tokens[ch][:, n_valid:] = PAD
            masked[ch][n_valid:] = False

    for step in range(steps):
        logits = ensemble_logits(models, spect, tokens)
        predictions = predict_tokens(logits, tolerance)

        selected = {}
        for ch in tokens:
            selected[ch] = select_frames(logits[ch], masked[ch], steps - step)
            selected[ch] = spare_suppressed_frames(
                selected[ch], predictions[ch], masked[ch], logits[ch], tolerance
            )
        for ch in tokens:
            tokens[ch][0, selected[ch]] = predictions[ch][selected[ch]]
            masked[ch] &= ~selected[ch]

        # downbeats are beats: a committed downbeat event forces the beat event,
        # which also wins over beat events committed nearby in the same step
        new_downbeats = selected["downbeat"] & (tokens["downbeat"][0] == EVENT)
        forced = new_downbeats & masked["beat"]
        if forced.any():
            tokens["beat"][0, forced] = EVENT
            masked["beat"] &= ~forced
            selected["beat"] |= forced
            other_beats = selected["beat"] & ~forced & (tokens["beat"][0] == EVENT)
            conflict = near_events(forced, other_beats, tolerance)
            tokens["beat"][0, conflict] = NO_EVENT
            selected["beat"] &= ~conflict
        # commit the neighbours of new events as non-events
        for ch in tokens:
            new_events = selected[ch] & (tokens[ch][0] == EVENT)
            suppressed = near_events(new_events, masked[ch], tolerance)
            tokens[ch][0, suppressed] = NO_EVENT
            masked[ch] &= ~suppressed

        all_committed = not any(masked[ch].any() for ch in tokens)
        nothing_new = not any(selected[ch].any() for ch in tokens)
        if all_committed or nothing_new:
            break

    for ch in tokens:
        tokens[ch][0, masked[ch]] = predictions[ch][masked[ch]]
    downbeat = tokens["downbeat"][0] == EVENT
    beat = (tokens["beat"][0] == EVENT) | downbeat
    return beat, downbeat


def ensemble_logits(models, spect, tokens):
    """Average the beat and downbeat logits of all models, returned with shape (time,)."""
    outputs = [model(spect, tokens["beat"], tokens["downbeat"]) for model in models]
    return {
        ch: torch.stack([output[ch] for output in outputs]).mean(0)[0]
        for ch in ("beat", "downbeat")
    }


def predict_tokens(logits, tolerance):
    """Beat events are the positive local maxima of the beat logits, downbeat events
    are beat events with positive downbeat logits."""
    beat_logits = logits["beat"][None]
    local_max = beat_logits == F.max_pool1d(beat_logits, 2 * tolerance + 1, 1, tolerance)
    beat = (local_max & (beat_logits > 0))[0]
    downbeat = beat & (logits["downbeat"] > 0)
    return {
        "beat": torch.where(beat, EVENT, NO_EVENT),
        "downbeat": torch.where(downbeat, EVENT, NO_EVENT),
    }


def select_frames(logits, masked, remaining_steps):
    """
    Select the masked frames to commit: a budget of ceil(masked / remaining_steps) frames,
    split between positive and negative logits in proportion to their counts, each filled
    with the frames of highest absolute logit.
    """
    positive = masked & (logits > 0)
    negative = masked & (logits <= 0)
    n_pos = positive.sum().item()
    n_neg = negative.sum().item()
    neg_pos_ratio = n_neg / n_pos if n_pos and n_neg else 1.0
    budget = math.ceil((n_pos + n_neg) / remaining_steps)
    k_pos = min(max(1, round(budget / (1 + neg_pos_ratio))), n_pos)
    confidence = logits.abs()
    return top_k(confidence, positive, k_pos) | top_k(confidence, negative, budget - k_pos)


def spare_suppressed_frames(selected, predictions, masked, logits, tolerance):
    """
    Committing new events will also commit their masked neighbours as non-events.
    Compensate by unselecting as many selected non-events (away from the new events)
    as there will be suppressed neighbours, starting from the least confident ones.
    """
    new_events = selected & (predictions == EVENT)
    n_suppressed = near_events(new_events, masked & ~selected, tolerance).sum().item()
    removable = (
        selected
        & (predictions != EVENT)
        & ~near_events(new_events, masked, tolerance)
    )
    n_remove = min(n_suppressed, removable.sum().item())
    if n_remove == 0:
        return selected
    confidence = logits.abs().masked_fill(~removable, torch.inf)
    selected = selected.clone()
    selected[confidence.topk(n_remove, largest=False).indices] = False
    return selected


def top_k(values, candidates, k):
    """Boolean mask of the (at most) `k` candidates with the highest values."""
    selected = torch.zeros_like(candidates)
    k = min(k, candidates.sum().item())
    if k > 0:
        values = values.masked_fill(~candidates, -torch.inf)
        selected[values.topk(k).indices] = True
    return selected


def near_events(events, candidates, tolerance):
    """Candidates within `tolerance` frames of an event, excluding the events themselves."""
    near = F.max_pool1d(events[None].float(), 2 * tolerance + 1, 1, tolerance)[0] > 0
    return near & ~events & candidates
