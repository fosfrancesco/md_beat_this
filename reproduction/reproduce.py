#!/usr/bin/env python3
"""Recompute the released predictions, to be evaluated with evaluate.py.

    python reproduction/reproduce.py gtzan --audio-root AUDIO -o PREDICTIONS
    python reproduction/reproduce.py cv --audio-root AUDIO -o PREDICTIONS [--datasets smc ...]
    python reproduction/evaluate.py --pred-root PREDICTIONS --ann-root beat_this_annotations

Audio files are expected at AUDIO/<dataset>/<track>.<ext>, named after the annotations
of beat_this_annotations. Predictions are written with the layout of results/.

gtzan: each of the three ensembles (run1, run2, run3) processes every GTZAN track in a
single chunk. cv: each track is processed in snippets by the model of the fold that
held it out (fold0 to fold7, as listed in results/cv_folds.tsv).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import tqdm

from md_beat_this.inference import File2File

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GTZAN_RUNS = ("run1", "run2", "run3")


def gtzan_jobs():
    """Map each checkpoint name to its (dataset, track, output folder) jobs."""
    predictions = RESULTS_DIR / "gtzan" / GTZAN_RUNS[0] / "predictions"
    tracks = sorted(path.name.removesuffix(".beats.txt") for path in predictions.iterdir())
    return {
        run: [("gtzan", track, Path("gtzan", run)) for track in tracks]
        for run in GTZAN_RUNS
    }


def cv_jobs(datasets=None):
    """Map each checkpoint name to its (dataset, track, output folder) jobs."""
    jobs = defaultdict(list)
    with open(RESULTS_DIR / "cv_folds.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dataset, track = row["dataset"], row["track"]
            if datasets is None or dataset in datasets:
                jobs[f"fold{row['fold']}"].append((dataset, track, Path(dataset)))
    return dict(sorted(jobs.items()))


def find_audio(audio_files, audio_root, dataset, track):
    """Look up the audio file of a track, listing each dataset folder once."""
    if dataset not in audio_files:
        audio_dir = audio_root / dataset
        if audio_dir.is_dir():
            audio_files[dataset] = {path.stem: path for path in audio_dir.iterdir()}
        else:
            print(f"No audio folder {audio_dir}, skipping {dataset}.")
            audio_files[dataset] = None
    if audio_files[dataset] is None:
        return None
    if track not in audio_files[dataset]:
        print(f"No audio for {dataset}/{track}, skipping it.")
    return audio_files[dataset].get(track)


def run(jobs, snippets, audio_root, output_dir, checkpoint_dir, skip_existing, gpu):
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    audio_files = {}
    for model, tracks in jobs.items():
        checkpoint = model if checkpoint_dir is None else checkpoint_dir / f"{model}.ckpt"
        file2file = File2File(checkpoint, device, snippets=snippets)
        for dataset, track, folder in tqdm.tqdm(tracks, desc=model):
            output = output_dir / folder / "predictions" / f"{track}.beats.txt"
            if skip_existing and output.exists():
                continue
            audio = find_audio(audio_files, audio_root, dataset, track)
            if audio is not None:
                file2file(audio, output)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Folder holding the audio files at <dataset>/<track>.<ext>.",
    )
    common.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to write the predictions to, with the layout of results/.",
    )
    common.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Folder holding the run*.ckpt and fold*.ckpt checkpoints. If omitted, "
        "the released checkpoints are downloaded on first use.",
    )
    common.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tracks whose output file already exists.",
    )
    common.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use, or -1 for CPU (default: %(default)s).",
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="results", required=True)
    subparsers.add_parser(
        "gtzan", parents=[common], help="The GTZAN predictions of the three ensembles."
    )
    cv = subparsers.add_parser(
        "cv", parents=[common], help="The cross-validation predictions."
    )
    cv.add_argument(
        "--datasets", nargs="+", help="Only process these datasets (default: all)."
    )
    args = vars(parser.parse_args())

    if args.pop("results") == "gtzan":
        jobs, snippets = gtzan_jobs(), False
    else:
        jobs, snippets = cv_jobs(args.pop("datasets")), True
    run(jobs, snippets, **args)


if __name__ == "__main__":
    main()
