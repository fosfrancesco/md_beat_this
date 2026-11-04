import argparse
import sys
from pathlib import Path

import torch
import tqdm
from beat_this.cli import derive_output_path

from md_beat_this.inference import File2File


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detects beats and downbeats in given audio files with an "
        "ensemble of masked diffusion Beat This! models."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="Audio file(s) or directories of audio files to process.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="run1",
        help="The model to use: run1, run2 or run3 (the GTZAN ensembles), "
        "fold0 to fold7 (the cross-validation models), a local checkpoint file, "
        "or a URL. Released models are downloaded on first use "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file name for a single input file, or output directory "
        "for multiple input files. If omitted, outputs are saved next to each "
        "input file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".beats.txt",
        help="Suffix to replace the input file suffix with when deriving "
        "output file names (default: %(default)s).",
    )
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append the suffix instead of replacing the input file suffix.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing files whose output file already exists.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of unmasking steps (default: %(default)s).",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        help="Only use the first N models of the ensemble: 1 to 3 for run1, run2 "
        "and run3 (default: all of them). The fold models are single models.",
    )
    parser.add_argument(
        "--snippets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process the audio in 30 seconds snippets, each conditioned on the "
        "predictions of the previous one, as for the cross-validation results "
        "(default). With --no-snippets, process the whole audio in a single chunk, "
        "as for the GTZAN results; meant for excerpts of about 30 seconds.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use, or -1 for CPU (default: %(default)s).",
    )
    return parser


def run(
    inputs,
    model,
    output,
    suffix,
    append,
    skip_existing,
    steps,
    num_models,
    snippets,
    gpu,
):
    # determine device
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # prepare model
    file2file = File2File(model, device, steps, snippets, num_models)

    # process inputs
    inputs = [Path(item) for item in inputs]
    if output is not None:
        output = Path(output)
    if len(inputs) == 1 and not inputs[0].is_dir():
        # special case: single input file
        if output is None or output.is_dir():
            output = derive_output_path(inputs[0], suffix, append, output)
        file2file(inputs[0], output)
        return
    # multiple inputs: first collect tasks so we can have a progress bar
    tasks = []
    for item in inputs:
        if item.is_dir():
            for fn in sorted(item.rglob("*")):
                if not fn.name.endswith(suffix) and not fn.is_dir():
                    tasks.append(
                        (fn, derive_output_path(fn, suffix, append, output, parent=item))
                    )
        else:
            tasks.append((item, derive_output_path(item, suffix, append, output)))
    if skip_existing:
        tasks = [(item, out) for item, out in tasks if not out.exists()]
    # then process all of them
    for item, out in tqdm.tqdm(tasks):
        try:
            file2file(item, out)
        except Exception:
            print(
                f'Could not process "{item}". Rerun with this file alone for details.',
                file=sys.stderr,
            )


def main():
    run(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    main()
