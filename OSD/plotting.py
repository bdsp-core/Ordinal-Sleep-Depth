from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

try:
    from .inference import EEG_CHANNELS, read_h5_contents
except ImportError:
    from inference import EEG_CHANNELS, read_h5_contents

if TYPE_CHECKING:
    try:
        from .inference import OSDResult
    except ImportError:
        from inference import OSDResult


STAGE_LABELS = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake"}


def _choose_plot_channel(signals: dict[str, np.ndarray], preferred_channel: str) -> str:
    preferred_channel = preferred_channel.lower()
    if preferred_channel in signals:
        return preferred_channel
    for channel in ("c4-m1", "c3-m2", *EEG_CHANNELS):
        if channel in signals:
            return channel
    return next(iter(signals))


def _downsample_track(values: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return values
    return values[::step]


def _time_ticks(total_hours: float) -> np.ndarray:
    if total_hours <= 1.0:
        step = 0.1
    elif total_hours <= 2.0:
        step = 0.25
    elif total_hours <= 6.0:
        step = 0.5
    else:
        step = 1.0
    return np.arange(0, total_hours + step * 0.5, step)


def _smooth_epoch_scores(epoch_scores: np.ndarray, window: int = 10) -> np.ndarray:
    if epoch_scores.size == 0:
        return epoch_scores
    kernel = np.ones(window, dtype=np.float32)
    weights = np.convolve(np.ones_like(epoch_scores, dtype=np.float32), kernel, mode="same")
    smoothed = np.convolve(epoch_scores.astype(np.float32), kernel, mode="same") / weights
    return smoothed


def plot_summary(
    input_path: Path | str,
    result: "OSDResult",
    output_path: Path | str,
    preferred_channel: str = "c4-m1",
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() != ".h5":
        raise ValueError("Summary plotting currently expects an H5 input with signals/annotations.")

    signals, annotations, sampling_rate = read_h5_contents(input_path)
    stage = annotations.get("stage")
    plot_channel = _choose_plot_channel(signals, preferred_channel)
    eeg = signals[plot_channel]

    spec_freqs, spec_times, spec_power = spectrogram(
        eeg * 1e6,
        fs=sampling_rate,
        window="hann",
        nperseg=800,
        noverlap=600,
        scaling="density",
        mode="psd",
    )
    freq_mask = spec_freqs <= 20
    spec_freqs = spec_freqs[freq_mask]
    spec_power = spec_power[freq_mask]
    spec_db = 10 * np.log10(spec_power + 1e-12)

    total_hours = eeg.shape[0] / sampling_rate / 3600
    osd_axis = np.arange(result.sample_scores.shape[0]) / result.sampling_rate / 3600

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]},
        constrained_layout=True,
    )

    if stage is not None:
        stage_plot = _downsample_track(stage, sampling_rate)
        stage_time = np.arange(stage_plot.shape[0]) / 3600
        rem_mask = stage_plot.copy()
        rem_mask[rem_mask != 4] = np.nan
        axes[0].step(stage_time, stage_plot, where="post", color="black", linewidth=0.9)
        axes[0].step(stage_time, rem_mask, where="post", color="red", linewidth=1.0)
        axes[0].set_yticks(list(STAGE_LABELS))
        axes[0].set_yticklabels([STAGE_LABELS[idx] for idx in STAGE_LABELS])
        axes[0].set_ylim(5.25, 0.75)
        axes[0].set_ylabel("Stage")
    else:
        axes[0].text(0.5, 0.5, "No stage annotations", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_yticks([])
    axes[0].set_title("Stage Hypnogram")
    axes[0].grid(alpha=0.2)

    mesh = axes[1].pcolormesh(
        spec_times / 3600,
        spec_freqs,
        spec_db,
        shading="auto",
        cmap="magma",
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 95),
    )
    axes[1].set_title(f"EEG Spectrogram ({plot_channel})")
    axes[1].set_ylabel("Hz")
    axes[1].set_ylim(0, 20)
    colorbar = figure.colorbar(mesh, ax=axes, pad=0.01, fraction=0.025)
    colorbar.set_label("dB")

    smoothed_epoch_scores = _smooth_epoch_scores(result.epoch_scores, window=10)
    osd_axis = np.arange(smoothed_epoch_scores.shape[0]) * result.samples_per_epoch / result.sampling_rate / 3600
    axes[2].plot(osd_axis, smoothed_epoch_scores, color="#0b7285", linewidth=1.2)
    axes[2].set_title("Ordinal Sleep Depth")
    axes[2].set_ylabel("OSD")
    axes[2].set_xlabel("Hours")
    axes[2].grid(alpha=0.2)

    x_max = max(total_hours, osd_axis[-1] if len(osd_axis) else 0)
    xticks = _time_ticks(x_max)
    for axis in axes:
        axis.set_xlim(0, x_max)
        axis.set_xticks(xticks)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
