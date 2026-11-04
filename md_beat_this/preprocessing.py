import subprocess

import numpy as np
import soxr
import torch
import torchaudio

SAMPLE_RATE = 22050


def load_audio(path, sr=SAMPLE_RATE):
    """Decode an audio file with ffmpeg to a mono float32 signal at sample rate `sr`."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-ar", str(sr), "-ac", "1"]
    cmd += ["-f", "f32le", "pipe:1"]
    signal_bytes = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(signal_bytes, dtype=np.float32), sr


def prepare_signal(signal, sr):
    """Convert a signal of shape (samples,) or (samples, channels) to mono float32 at 22050 Hz."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(1)
    elif signal.ndim != 1:
        raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
    if sr != SAMPLE_RATE:
        signal = soxr.resample(signal, in_rate=sr, out_rate=SAMPLE_RATE)
    return signal


def rms_trim_bounds(signal, sr=SAMPLE_RATE, threshold=1.5e-3, window=0.1, hop=0.05):
    """
    Find the start and end sample of the region spanning from the first to the last
    window (of `window` seconds, every `hop` seconds) with an RMS of at least `threshold`.
    Returns the full signal bounds if the signal is shorter than a window or entirely silent.
    """
    window_size = int(window * sr)
    hop_size = int(hop * sr)
    if len(signal) < window_size:
        return 0, len(signal)

    def is_active(i):
        return np.mean(signal[i * hop_size : i * hop_size + window_size] ** 2) >= threshold**2

    num_windows = (len(signal) - window_size) // hop_size + 1
    first = next((i for i in range(num_windows) if is_active(i)), None)
    if first is None:
        return 0, len(signal)
    last = next((i for i in range(num_windows - 1, first, -1) if is_active(i)), first)
    return first * hop_size, min(last * hop_size + window_size, len(signal))


class LogMelSpect(torch.nn.Module):
    """
    Like beat_this.preprocessing.LogMelSpect, but with the features of the masked
    diffusion models: slaney-normalized filters up to 10 kHz, no frame length
    normalization, and log(1 + 1000 x) computed without log1p.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=10000,
        n_mels=128,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Input is a waveform as a monodimensional array of shape T,
        output is a 2D log mel spectrogram of shape (F,128)."""
        return torch.log(self.spect_class(x) * self.log_multiplier + 1).T
