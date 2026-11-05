import warnings
from pathlib import Path

import numpy as np
import torch
from beat_this.inference import zeropad
from beat_this.utils import save_beat_tsv

from md_beat_this.diffusion import unmask
from md_beat_this.model.beat_tracker import MDBeatThis
from md_beat_this.preprocessing import (
    SAMPLE_RATE,
    LogMelSpect,
    load_audio,
    prepare_signal,
    rms_trim_bounds,
)

CHECKPOINT_URL = "https://github.com/fosfrancesco/md_beat_this/releases/download/v1.0"
CHECKPOINT_NAMES = [f"run{i}" for i in range(1, 4)] + [f"fold{i}" for i in range(8)]
FPS = 50
BORDER_SIZE = 6
# with snippets, as for the cross-validation results: 30 seconds chunks (the training
# length) with 10 seconds of overlap, each conditioned on the predictions of the previous
# chunk in the overlap, except for its last 2 seconds; shorter pieces are padded to 30 seconds
CHUNK_SIZE = 1500
OVERLAP = 500
DISCARD = 100
# without snippets, as for the GTZAN results: the whole piece in a single chunk
MIN_SINGLE_CHUNK_SIZE = 100
MAX_SINGLE_PIECE_SIZE = 1750  # 35 seconds; longer pieces are processed with a warning


def load_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> dict:
    """
    Load a checkpoint given as a local path, a URL, or the name of a released checkpoint
    (run1 to run3, fold0 to fold7). Downloaded checkpoints are cached by torch.hub.
    """
    if Path(checkpoint_path).is_file():
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    if str(checkpoint_path).startswith(("https://", "http://")):
        url, file_name = str(checkpoint_path), None
    elif checkpoint_path not in CHECKPOINT_NAMES:
        raise ValueError(
            f"{checkpoint_path!r} is not a file, a URL, "
            f"or one of the released checkpoints {CHECKPOINT_NAMES}"
        )
    else:
        url = f"{CHECKPOINT_URL}/{checkpoint_path}.ckpt"
        file_name = f"md_beat_this-{checkpoint_path}.ckpt"
    return torch.hub.load_state_dict_from_url(
        url, file_name=file_name, map_location=device, weights_only=True
    )


def load_models(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    num_models: int | None = None,
) -> list[MDBeatThis]:
    """
    Load the models of the ensemble stored in a checkpoint (see `load_checkpoint`),
    holding their state dicts under "state_dicts". If `num_models` is given, only the
    first `num_models` models of the ensemble are loaded.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dicts = checkpoint["state_dicts"]
    if num_models is not None and not 1 <= num_models <= len(state_dicts):
        raise ValueError(
            f"num_models must be between 1 and {len(state_dicts)} "
            f"for {checkpoint_path!r}, got {num_models}"
        )
    models = []
    for state_dict in state_dicts[:num_models]:
        model = MDBeatThis()
        model.load_state_dict(state_dict)
        models.append(model.to(device).eval())
    return models


def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int,
    overlap: int,
    min_size: int,
):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size`
    frames, consecutive chunks overlapping by `overlap` frames, and return the chunks and
    their starting positions (counted after padding).
    The piece is first padded with `border_size` frames in the beginning, and the last chunk is
    shifted left to end at the end of the piece. If the padded piece is not longer than
    `chunk_size`, it is returned as a single chunk, zero-padded at the end to at least
    `min_size` frames.
    """
    spect = zeropad(spect, left=border_size)
    if len(spect) <= chunk_size:
        return [zeropad(spect, right=max(0, min_size - len(spect)))], [0]
    starts = list(range(0, len(spect) - chunk_size, chunk_size - overlap))
    starts.append(len(spect) - chunk_size)
    return [spect[start : start + chunk_size] for start in starts], starts


def aggregate_prediction(
    pred_chunks: list,
    starts: list,
    full_size: int,
    chunk_size: int,
    border_size: int,
    overlap: int,
) -> torch.Tensor:
    """
    Aggregate the framewise predictions of the chunks produced by `split_piece`.
    Each chunk provides the central `chunk_size - overlap` frames of its regular
    (not shifted) position, the first and last chunk extend to the start and end of the piece.
    """
    hop = chunk_size - overlap
    context = (chunk_size - 1) // 2 - (hop - 1) // 2 - border_size
    piece_prediction = pred_chunks[0].new_zeros(full_size)
    for i, (start, pchunk) in enumerate(zip(starts, pred_chunks)):
        begin = 0 if i == 0 else i * hop + context
        end = full_size if i == len(starts) - 1 else (i + 1) * hop + context
        shift = border_size - start
        piece_prediction[begin:end] = pchunk[begin + shift : end + shift]
    return piece_prediction


class Spect2Frames:
    """
    Class for extracting framewise beat and downbeat predictions (booleans) from a spectrogram.

    Args:
        checkpoint_path (str): Name of a released checkpoint (run1 to run3, fold0 to fold7),
            local path or URL. Default is "run1".
        device (str): Device to use for inference. Default is "cpu".
        steps (int): Number of unmasking steps. Default is 8.
        snippets (bool): If True, process the piece in 30 seconds snippets, each conditioned
            on the predictions of the previous one (block-autoregressive), as for the
            cross-validation results. If False, process the whole piece in a single chunk,
            as for the GTZAN results; meant for excerpts of about 30 seconds.
            Default is True.
        num_models (int, optional): Only use the first `num_models` models of the
            ensemble stored in the checkpoint. Default is None, using all of them.
    """

    def __init__(
        self,
        checkpoint_path="run1",
        device="cpu",
        steps=8,
        snippets=True,
        num_models=None,
    ):
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        # the released predictions were computed with TF32 matrix multiplications on GPU
        torch.set_float32_matmul_precision("high")
        self.device = torch.device(device)
        self.models = load_models(checkpoint_path, self.device, num_models)
        self.steps = steps
        self.snippets = snippets

    def spect2frames(self, spect):
        if self.snippets:
            chunk_size, overlap, min_size = CHUNK_SIZE, OVERLAP, CHUNK_SIZE
        else:
            if len(spect) > MAX_SINGLE_PIECE_SIZE:
                warnings.warn(
                    f"Processing {len(spect) / FPS:.0f} seconds in a single chunk, "
                    "while the models were trained on 30 seconds excerpts. "
                    "Consider using snippets."
                )
            chunk_size, overlap = len(spect) + BORDER_SIZE, 0
            min_size = MIN_SINGLE_CHUNK_SIZE
        with torch.inference_mode():
            chunks, starts = split_piece(
                spect, chunk_size, BORDER_SIZE, overlap, min_size
            )
            pred_chunks = []
            for i, chunk in enumerate(chunks):
                prefix = None
                if i > 0:
                    # the previous chunk's predictions where it overlaps this chunk
                    overlap_start = starts[i] - starts[i - 1]
                    prefix = tuple(
                        pred[overlap_start : chunk_size - DISCARD]
                        for pred in pred_chunks[-1]
                    )
                pred_chunks.append(
                    unmask(self.models, chunk.unsqueeze(0), self.steps, prefix=prefix)
                )
            beat, downbeat = (
                aggregate_prediction(
                    [p[i] for p in pred_chunks],
                    starts,
                    len(spect),
                    chunk_size,
                    BORDER_SIZE,
                    overlap,
                )
                for i in range(2)
            )
        return beat, downbeat

    def __call__(self, spect):
        return self.spect2frames(spect)


class Audio2Beats(Spect2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio signal.
    The signal is trimmed to its non-silent region before processing.
    """

    def __init__(
        self,
        checkpoint_path="run1",
        device="cpu",
        steps=8,
        snippets=True,
        num_models=None,
    ):
        super().__init__(checkpoint_path, device, steps, snippets, num_models)
        self.spect = LogMelSpect(device=self.device)

    def __call__(self, signal, sr):
        signal = prepare_signal(signal, sr)
        start, end = rms_trim_bounds(signal)
        signal = torch.tensor(signal[start:end], device=self.device)
        beat, downbeat = self.spect2frames(self.spect(signal))
        frames = beat.nonzero()[:, 0].cpu()
        beats = frames.float() / FPS + torch.tensor(start / SAMPLE_RATE)
        downbeats = beats[downbeat.cpu()[frames]]
        return beats.numpy(), downbeats.numpy()


class File2Beats(Audio2Beats):
    def __call__(self, audio_path):
        signal, sr = load_audio(audio_path)
        return super().__call__(signal, sr)


class File2File(File2Beats):
    def __call__(self, audio_path, output_path):
        # millisecond precision, as in the released predictions
        beats, downbeats = (
            times.astype(np.float64).round(3) for times in super().__call__(audio_path)
        )
        save_beat_tsv(beats, downbeats, output_path)
