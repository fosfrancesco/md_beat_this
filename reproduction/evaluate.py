#!/usr/bin/env python3
"""Evaluate predictions/ against beat_this_annotations and print result tables.

Requires:
    pip install mir_eval numpy

Annotations (release v1.1):
    git clone --branch v1.1 --depth 1 \\
        https://github.com/CPJKU/beat_this_annotations.git

Usage (from this repo root):
    python reproduction/evaluate.py --ann-root /path/to/beat_this_annotations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mir_eval.beat
import numpy as np

SKIP_FIRST = 5.0  # seconds; MIREX / Beat This! convention

# Second column of both annotations and predictions (beat number within the bar)
# marks a downbeat with 1.
DOWNBEAT = 1.0

METRIC_KEYS = ("F-measure", "CMLc", "CMLt", "AMLc", "AMLt")

# Dataset folder names, shared between this repo and beat_this_annotations.
DATASETS = (
    "asap",
    "ballroom",
    "beatles",
    "candombe",
    "filosax",
    "groove_midi",
    "guitarset",
    "hainsworth",
    "harmonix",
    "hjdb",
    "jaah",
    "rwc",
    "simac",
    "smc",
    "tapcorrect",
)

RWC_SUBSETS = [
    ("rwc-popular", "rwc_popular"),
    ("rwc-classical", "rwc_classical"),
    ("rwc-jazz", "rwc_jazz"),
    ("rwc-royalty-free", "rwc_royalty-free"),
]

# GTZAN: three independent training runs; report mean ± std across runs.
GTZAN_RUNS = ("run1", "run2", "run3")


def load_beats(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read beat times and, when present, the beat position of each beat.

    Files hold one beat per line as whitespace-separated columns: the time in
    seconds, then optionally the beat position. Files without a second column
    (simac, smc and six GTZAN tracks) carry no downbeat information and yield
    ``None`` positions. Harmonix additionally has a third (bar number) column,
    which is ignored.
    """
    # Not mir_eval.io.load_labeled_events: it splits with maxsplit=1, so the eight
    # three-column Harmonix files would yield beat positions like "2\t1".
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    times = np.array([float(r[0]) for r in rows], dtype=np.float64)
    if not rows or len(rows[0]) == 1:
        return times, None
    return times, np.array([float(r[1]) for r in rows], dtype=np.float64)


def score(ref_t: np.ndarray, est_t: np.ndarray) -> dict[str, float]:
    ref_t = mir_eval.beat.trim_beats(ref_t, min_beat_time=SKIP_FIRST)
    est_t = mir_eval.beat.trim_beats(est_t, min_beat_time=SKIP_FIRST)
    f = mir_eval.beat.f_measure(ref_t, est_t)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref_t, est_t)
    return {
        "F-measure": float(f),
        "CMLc": float(cmlc),
        "CMLt": float(cmlt),
        "AMLc": float(amlc),
        "AMLt": float(amlt),
    }


def evaluate_dataset(
    pred_dir: Path,
    ann_dir: Path,
    name_prefix: str | None = None,
    *,
    require_beat_positions: bool = False,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    beats, downs = [], []
    for pred in sorted(pred_dir.glob("*.beats.txt")):
        stem = pred.name[: -len(".beats.txt")]
        if name_prefix is not None and not stem.startswith(name_prefix):
            continue
        ann = ann_dir / f"{stem}.beats"
        ref_t, ref_p = load_beats(ann)
        est_t, est_p = load_beats(pred)
        if require_beat_positions and ref_p is None:
            continue
        if ref_t.size == 0:
            print(f"[skip] {ann}: no beats annotated", file=sys.stderr)
            continue
        beats.append(score(ref_t, est_t))
        if ref_p is not None and est_p is not None:
            downs.append(score(ref_t[ref_p == DOWNBEAT], est_t[est_p == DOWNBEAT]))
    return beats, downs


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}


def mean_std_across_runs(
    aggregates: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for k in METRIC_KEYS:
        values = [m[k] for m in aggregates]
        out[k] = (float(np.mean(values)), float(np.std(values)))
    return out


def fmt_cv(n: int, m: dict[str, float] | None) -> str:
    if m is None or n == 0:
        return "— | — | — | —"
    return f"{n} | {m['F-measure']:.3f} | {m['CMLt']:.3f} | {m['AMLt']:.3f}"


def print_mean_std_table(
    beat: dict[str, tuple[float, float]],
    downbeat: dict[str, tuple[float, float]],
) -> None:
    print("| Metric | Beat (mean ± std) | Downbeat (mean ± std) |")
    print("| --- | --- | --- |")
    for k in METRIC_KEYS:
        cells = " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in (beat[k], downbeat[k]))
        print(f"| {k} | {cells} |")
    print()


def print_cv_table(pred_root: Path, ann_root: Path) -> None:
    header = (
        "| Dataset | Beat N | Beat F | Beat CMLt | Beat AMLt | "
        "Downbeat N | Downbeat F | Downbeat CMLt | Downbeat AMLt |"
    )
    print(header)
    print("|" + "|".join(["---"] * 9) + "|")

    macro_b, macro_d = [], []
    for ds in DATASETS:
        pred_dir = pred_root / ds / "predictions"
        ann_dir = ann_root / ds / "annotations" / "beats"
        if not pred_dir.is_dir() or not ann_dir.is_dir():
            print(f"[skip] {ds}: missing predictions or annotations", file=sys.stderr)
            continue

        beats, downs = evaluate_dataset(pred_dir, ann_dir)
        mb, md = mean_metrics(beats), mean_metrics(downs)
        label = f"{ds} (all)" if ds == "rwc" else ds
        print(f"| {label} | {fmt_cv(len(beats), mb)} | {fmt_cv(len(downs), md)} |")
        if mb is not None:
            macro_b.append(mb)
        if md is not None:
            macro_d.append(md)

        if ds == "rwc":
            for sub_label, prefix in RWC_SUBSETS:
                sb, sd = evaluate_dataset(pred_dir, ann_dir, name_prefix=prefix)
                print(
                    f"| &nbsp;&nbsp;↳ {sub_label} | "
                    f"{fmt_cv(len(sb), mean_metrics(sb))} | {fmt_cv(len(sd), mean_metrics(sd))} |"
                )

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    print(
        f"| **Mean (macro)** |  | "
        f"**{mean(macro_b, 'F-measure'):.3f}** | **{mean(macro_b, 'CMLt'):.3f}** | "
        f"**{mean(macro_b, 'AMLt'):.3f}** |  | "
        f"**{mean(macro_d, 'F-measure'):.3f}** | **{mean(macro_d, 'CMLt'):.3f}** | "
        f"**{mean(macro_d, 'AMLt'):.3f}** |"
    )
    print()


def print_gtzan_tables(pred_root: Path, ann_root: Path) -> None:
    ann_dir = ann_root / "gtzan" / "annotations" / "beats"
    if not ann_dir.is_dir():
        print("[skip] gtzan: missing annotations", file=sys.stderr)
        return

    beat_aggs, down_aggs = [], []
    for run in GTZAN_RUNS:
        pred_dir = pred_root / "gtzan" / run / "predictions"
        if not pred_dir.is_dir():
            print(f"[skip] gtzan/{run}: missing predictions", file=sys.stderr)
            return
        beats, downs = evaluate_dataset(pred_dir, ann_dir, require_beat_positions=True)
        mb, md = mean_metrics(beats), mean_metrics(downs)
        if mb is None or md is None:
            print(f"[skip] gtzan/{run}: no scored tracks", file=sys.stderr)
            return
        print(f"[gtzan/{run}] N_beat={len(beats)} N_downbeat={len(downs)}", file=sys.stderr)
        beat_aggs.append(mb)
        down_aggs.append(md)

    print("### GTZAN (mean ± std over 3 runs)\n")
    print_mean_std_table(
        mean_std_across_runs(beat_aggs), mean_std_across_runs(down_aggs)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ann-root",
        type=Path,
        required=True,
        help="Clone of beat_this_annotations (use release v1.1)",
    )
    ap.add_argument(
        "--pred-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Root folder holding the per-dataset prediction folders (default: ./results)",
    )
    ap.add_argument(
        "--only-gtzan",
        action="store_true",
        help="Only print the GTZAN mean±std tables (skip the CV table).",
    )
    args = ap.parse_args()

    if not args.only_gtzan:
        print("### Cross-validation\n")
        print_cv_table(args.pred_root, args.ann_root)

    print_gtzan_tables(args.pred_root, args.ann_root)


if __name__ == "__main__":
    main()
