# Masked Diffusion Enables Coherent Beat Tracking

Official inference code and models for the ISMIR 2026 paper "[Masked Diffusion Enables Coherent Beat Tracking](https://arxiv.org/abs/2608.04624)" by Francesco Foscarin, Filip Korzeniowski and Richard Vogl. Please [cite the paper](#citation) if you use any of this work.

You can use the released models to detect beats and downbeats in your own audio files. We do not release training code or audio, but we provide our predictions for all evaluation datasets (and the activations for GTZAN), so results can be explored and metrics recomputed. We also report 8-fold cross-validation results (not in the paper) and an expanded version of Figure 4.

* [Inference](#inference)
* [Available models](#available-models)
* [Differences from Beat This!](#differences-from-beat-this)
* [Released predictions and activations](#released-predictions-and-activations)
* [Results](#results)
* [Reproducing the results](#reproducing-the-results)
* [License](#license)
* [Citation](#citation)

## Inference

To predict beats for audio files, you can use the command line tool or call the beat tracker from Python.

### Requirements

1. [Install PyTorch](https://pytorch.org/get-started/locally/) following the instructions for your platform.
2. Install [ffmpeg](https://ffmpeg.org/), which is used to decode audio files.

Then install the package with:
```bash
pip install https://github.com/fosfrancesco/md_beat_this/archive/main.zip
```
or, from a clone of this repository, with `pip install .`.

### Command line

The package installs a command line tool called `md_beat_this`. To see all options, run:
```bash
md_beat_this --help
```
The basic usage is:
```bash
md_beat_this path/to/audio.file -o path/to/output.beats.txt
```
To process multiple files, pass multiple input files or directories, and an output directory:
```bash
md_beat_this path/to/*.mp3 path/to/whole_directory/ -o path/to/output_directory --skip-existing
```
Without `-o`, outputs are saved next to each input file. Each output line holds a beat time in seconds, a tab, and the beat number within the bar (`1` for downbeats), as in [Beat This!](https://github.com/CPJKU/beat_this).

The tool uses the first GPU by default and falls back to CPU if none is available; pass `--gpu=-1` to force the CPU. Use `-m` to pick another [model](#available-models), and `--steps` to change the number of unmasking steps (default 8, as in the paper). The `run*` models are ensembles of 3 models; pass `--num-models 1` or `--num-models 2` to only use the first one or two, which is faster but slightly less accurate (see the [step study](#step-study)).

### Python

```python
from md_beat_this.inference import File2Beats

file2beats = File2Beats(checkpoint_path="run1", device="cuda", steps=8, snippets=True)
beats, downbeats = file2beats("path/to/audio.file")
```
If you already have an audio signal loaded, use `Audio2Beats` and pass the signal and its sample rate. `File2File` writes the output file directly. Pass `num_models=1` or `num_models=2` to only use the first models of the ensemble.

### Long audio files

The models were trained on 30 second excerpts. By default (`--snippets`), the audio is processed in 30 second snippets with 10 seconds of overlap, block-autoregressively: each snippet is conditioned on the predictions of the previous one in the overlap, except for its last 2 seconds. Shorter audio is padded to 30 seconds. Our [cross-validation](#cross-validation) predictions were computed this way.

With `--no-snippets` (`snippets=False` in Python), the whole audio is processed in a single chunk. This is meant for excerpts of about 30 seconds, and a warning is printed for audio longer than 35 seconds. The [GTZAN](#gtzan) predictions were computed this way.

## Available models

Models are downloaded automatically on first use from the [GitHub release](https://github.com/fosfrancesco/md_beat_this/releases/tag/v1.0) and cached by `torch.hub`. Select one with `-m` on the command line or `checkpoint_path` in Python:

* `run1` (default), `run2`, `run3`: three ensembles of 3 models each, trained from independent seeds with GTZAN held out as the test set. These produce the [GTZAN results](#gtzan).
* `fold0` to `fold7`: single models from the 8-fold cross-validation. These produce the [cross-validation results](#cross-validation). `results/cv_folds.tsv` lists the fold of each track.

You can also pass the path of a local checkpoint file or a URL.

Results may be unfairly good on files the models were trained on. Use the `run*` models for GTZAN or your own data; for the other datasets we used, use the fold model that held each track out.

All models use the inference settings of the paper: 8 unmasking steps, balanced frame-based unmasking, peak picking, and no DBN.

## Released predictions and activations

The `results/` folder holds our predictions for every evaluation dataset, and the activations for GTZAN:

```
results/
  gtzan/run{1,2,3}/          # GTZAN: one ensemble per run
    predictions/             # *.beats.txt: time [s] <tab> beat number (1 = downbeat)
    activations/             # *.npy: float32 array of shape (2, T); row 0 = beat logits,
                             #        row 1 = downbeat logits, at 50 fps (22050 Hz / 441)
  <dataset>/                 # cross-validation: one fold model per track
    predictions/
  cv_folds.tsv               # cross-validation fold of each track
```

Filenames match the stems of the beat files in [beat_this_annotations v1.1](https://github.com/CPJKU/beat_this_annotations/releases/tag/v1.1) (e.g. `ballroom_Albums-AnaBelen_Veneo-01.beats.txt`), so they can be evaluated directly against it.

The activations are the outputs of the beat and downbeat heads at the last masked diffusion step. The predictions cannot be obtained exactly by postprocessing them, since they are built iteratively over all inference steps (see the paper), but the two should roughly correspond.

## Results

### GTZAN

Each of `run1`, `run2` and `run3` is an ensemble of 3 models. Metrics are the mean ± std across the three runs. Tracks without downbeat annotations are dropped (6 tracks). The F-measure, CMLt and AMLt values match the "Ours (8 inference steps)" row of Table 1 in the paper.

| Metric | Beat (mean ± std) | Downbeat (mean ± std) |
| --- | --- | --- |
| F-measure | 0.897 ± 0.002 | 0.795 ± 0.003 |
| CMLc | 0.804 ± 0.004 | 0.761 ± 0.006 |
| CMLt | 0.829 ± 0.004 | 0.764 ± 0.006 |
| AMLc | 0.889 ± 0.001 | 0.880 ± 0.003 |
| AMLt | 0.925 ± 0.002 | 0.885 ± 0.003 |

### Cross-validation

Each track is predicted by the single fold model that did not train on it (no ensembling across folds), [in snippets](#long-audio-files). Note that the folds of `asap`, `harmonix` and `rwc` [differ from the Beat This! ones](#by-mistake).

Datasets without downbeat annotations (`simac`, `smc`) only report beat metrics. Tracks whose annotation file contains no beat are excluded, which only affects the Beatles track `Revolution 9`. Three annotated tracks are not part of our evaluation: `Schumann_Toccata_WangY07M` and `Schumann_Toccata_no_repeat_DupreeF13M` (`asap`), and `hainsworth_179` (`hainsworth`).

| Dataset | Beat N | Beat F | Beat CMLt | Beat AMLt | Downbeat N | Downbeat F | Downbeat CMLt | Downbeat AMLt |
|---|---|---|---|---|---|---|---|---|
| asap | 471 | 0.898 | 0.806 | 0.842 | 471 | 0.855 | 0.733 | 0.849 |
| ballroom | 685 | 0.978 | 0.971 | 0.980 | 685 | 0.966 | 0.963 | 0.975 |
| beatles | 179 | 0.940 | 0.855 | 0.948 | 179 | 0.894 | 0.769 | 0.878 |
| candombe | 35 | 0.994 | 0.995 | 0.995 | 35 | 0.994 | 0.998 | 0.998 |
| filosax | 48 | 0.995 | 0.987 | 0.987 | 48 | 0.990 | 0.987 | 0.987 |
| groove_midi | 336 | 0.920 | 0.862 | 0.913 | 336 | 0.800 | 0.761 | 0.884 |
| guitarset | 180 | 0.980 | 0.971 | 0.972 | 180 | 0.962 | 0.951 | 0.964 |
| hainsworth | 221 | 0.929 | 0.863 | 0.951 | 221 | 0.797 | 0.698 | 0.863 |
| harmonix | 911 | 0.960 | 0.907 | 0.961 | 911 | 0.920 | 0.859 | 0.926 |
| hjdb | 235 | 0.988 | 0.985 | 0.990 | 235 | 0.973 | 0.971 | 0.984 |
| jaah | 113 | 0.954 | 0.901 | 0.933 | 113 | 0.898 | 0.852 | 0.895 |
| rwc (all) | 226 | 0.890 | 0.780 | 0.889 | 226 | 0.858 | 0.757 | 0.859 |
| &nbsp;&nbsp;↳ rwc-popular | 100 | 0.966 | 0.910 | 0.973 | 100 | 0.948 | 0.902 | 0.944 |
| &nbsp;&nbsp;↳ rwc-classical | 61 | 0.798 | 0.583 | 0.775 | 61 | 0.709 | 0.452 | 0.700 |
| &nbsp;&nbsp;↳ rwc-jazz | 50 | 0.830 | 0.722 | 0.837 | 50 | 0.833 | 0.801 | 0.856 |
| &nbsp;&nbsp;↳ rwc-royalty-free | 15 | 0.961 | 0.904 | 0.970 | 15 | 0.946 | 0.890 | 0.957 |
| simac | 595 | 0.760 | 0.483 | 0.870 | — | — | — | — |
| smc | 217 | 0.611 | 0.489 | 0.643 | — | — | — | — |
| tapcorrect | 101 | 0.923 | 0.810 | 0.918 | 101 | 0.870 | 0.700 | 0.836 |
| **Mean (macro)** |  | **0.915** | **0.844** | **0.919** |  | **0.906** | **0.846** | **0.915** |

### Step study

GTZAN metrics against the number of diffusion steps, expanding Figure 4 of the paper.

Non-diffusion baseline vs. diffusion (single model):

![Non-diffusion vs. diffusion](figures/step_study_nondiffusion_vs_diffusion.png)

Single model vs. ensemble of 3 models:

![Single vs. ensemble](figures/step_study_single_vs_ensemble.png)

## Reproducing the results

### Recomputing the metrics

`reproduction/evaluate.py` recomputes the [GTZAN](#gtzan) and [cross-validation](#cross-validation) tables from the predictions in `results/` and the reference annotations. Metrics use [`mir_eval`](https://github.com/craffel/mir_eval) defaults (F-measure window of 70 ms, continuity thresholds of 0.175), skipping the first 5 seconds of each track.

```bash
git clone --branch v1.1 --depth 1 https://github.com/CPJKU/beat_this_annotations.git
pip install mir_eval numpy
python reproduction/evaluate.py --ann-root ./beat_this_annotations
```
Pass `--only-gtzan` to print only the GTZAN table, and `--pred-root` to read predictions from another folder. Datasets without predictions are skipped.

### Recomputing the predictions

If you have the audio, `reproduction/reproduce.py` recomputes our predictions with the released models. Put the audio files at `<audio-root>/<dataset>/<track>.<ext>`, named after the stems of the beat_this_annotations beat files (e.g. `gtzan/gtzan_blues_00000.wav`), and run:
```bash
pip install ".[eval]"
python reproduction/reproduce.py gtzan --audio-root path/to/audio -o my_predictions
python reproduction/reproduce.py cv --audio-root path/to/audio -o my_predictions
python reproduction/evaluate.py --pred-root my_predictions --ann-root ./beat_this_annotations
```
* `gtzan` processes each GTZAN track with the three ensembles, in a single chunk.
* `cv` processes each track in snippets, with the fold model that held it out (listed in `results/cv_folds.tsv`). Pass `--datasets smc hainsworth` to only process some datasets.

Predictions are written with the layout of `results/`. Pass `--checkpoint-dir` to read the models from a local folder instead of downloading them. Small differences from our predictions are expected:

* On GPU and with the same audio files, the GTZAN predictions match ours exactly. Different audio files (e.g. other encodings) or running on CPU can move some beats by a frame.
* Our cross-validation predictions were computed with `torch.compile`, which slightly changes floating-point computations. The recomputed predictions differ on about 12% of the tracks, which changes the per-dataset metrics by at most 0.003 and leaves the macro means unchanged.

## Differences from Beat This!

Our model is based on [Beat This!](https://github.com/CPJKU/beat_this), with the same data, augmentations, loss and most hyperparameters. The differences are described below.

### Intended

On top of the masked diffusion formulation (see the paper), we changed:

* SwiGLU instead of GELU in the transformer feed-forward layers, with the hidden dimension reduced to 2/3 (rounded up to a multiple of 32);
* learning rate 0.0004 instead of 0.0008;
* weight decay 0.1 instead of 0.01 (excluding embedding layers);
* AdamW β2 = 0.95 instead of 0.999;
* gradient clipping at 1;
* 300 training epochs instead of 150;
* WSD (warmup-stable-decay) learning rate schedule instead of cosine, with 1000 warm-up steps and linear decay over the last 15% of steps;
* leading and trailing silence is trimmed before inference (100 ms windows with an RMS below 1.5 × 10⁻³).
### By mistake

Some differences were introduced by mistake when reimplementing our training code, 
we expect them to not significantly influence the results:

* the input spectrogram differs slightly (the inference code keeps these differences, since the models were trained with them):
  * the mel filters go up to 10 kHz instead of 11 kHz;
  * the mel filters are Slaney-normalized (`norm="slaney"`) instead of unnormalized;
  * the magnitude spectrogram is not normalized by the frame length;
  * the logarithmic compression is computed as `log(1000 x + 1)` instead of `log1p(1000 x)`, which only differs numerically;
* the cross-validation folds of `asap`, `harmonix` and `rwc` differ from the `8-folds.split` files of [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations/releases/tag/v1.1), so our cross-validation results on these three datasets are not computed on the same folds as the Beat This! ones. The other datasets use the same folds, and `results/cv_folds.tsv` lists the folds we used.

## License

The code, the model weights, the predictions and the activations are licensed under [CC BY-NC 4.0](LICENSE) ([Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)): you can use and adapt them for non-commercial purposes, with attribution.

## Citation

```bibtex
@inproceedings{foscarin2026maskedbeat,
    author = {Francesco Foscarin and Filip Korzeniowski and Richard Vogl},
    title = {Masked Diffusion Enables Coherent Beat Tracking},
    booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
    year = 2026,
}
```
