from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


EEG_CHANNELS = ["f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1"]
OUTPUT_COLUMN = "OSD"
SAMPLES_PER_EPOCH = 600
DEFAULT_SAMPLING_RATE = 200


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for OSD inference. Use an environment with the "
            "trained-model dependencies installed."
        ) from exc
    return tf


def _load_mne():
    try:
        import mne
        from mne.preprocessing import EOGRegression
    except ImportError as exc:
        raise RuntimeError(
            "MNE is required for OSD preprocessing. Install `mne` in the runtime environment."
        ) from exc
    return mne, EOGRegression


def _default_weights_path() -> Path:
    return MODULE_DIR / "utils/models/weights_scaler/OSD_weigths.h5"


def _default_scaler_path() -> Path:
    return MODULE_DIR / "utils/scaling_osd/scaling_values.pkl"


def _flatten_dataset(dataset: h5py.Dataset) -> np.ndarray:
    return np.squeeze(np.asarray(dataset)).astype(np.float32)


def _normalize_name(name: str) -> str:
    return name.lower().strip()


def _fallback_sources() -> Dict[str, Tuple[str, ...]]:
    return {
        "f3-m2": ("f3-m2", "c3-m2", "c4-m1"),
        "f4-m1": ("f4-m1", "c4-m1", "c3-m2"),
        "c3-m2": ("c3-m2", "f3-m2", "o1-m2", "c4-m1"),
        "c4-m1": ("c4-m1", "f4-m1", "o2-m1", "c3-m2"),
        "o1-m2": ("o1-m2", "c3-m2", "f3-m2"),
        "o2-m1": ("o2-m1", "c4-m1", "f4-m1"),
    }


def _match_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[0] == target_length:
        return signal
    if signal.shape[0] > target_length:
        return signal[:target_length]
    return np.pad(signal, (0, target_length - signal.shape[0]), mode="edge")


def _epoch_windows(eeg_uV: np.ndarray, samples_per_epoch: int) -> np.ndarray:
    n_channels, n_samples = eeg_uV.shape
    n_epochs = n_samples // samples_per_epoch
    if n_epochs == 0:
        raise ValueError(
            f"Input signal is too short for OSD inference: {n_samples} samples < {samples_per_epoch}."
        )
    fit_samples = n_epochs * samples_per_epoch
    trimmed = eeg_uV[:, :fit_samples]
    return trimmed.reshape(n_channels, samples_per_epoch, n_epochs, order="F").transpose(2, 1, 0)


def _expand_epoch_scores(epoch_scores: np.ndarray, target_length: int, samples_per_epoch: int) -> np.ndarray:
    expanded = np.repeat(epoch_scores.astype(np.float32), samples_per_epoch)
    if expanded.shape[0] >= target_length:
        return expanded[:target_length]
    if expanded.shape[0] == 0:
        return np.zeros(target_length, dtype=np.float32)
    return np.pad(expanded, (0, target_length - expanded.shape[0]), mode="edge")


@dataclass
class OSDResult:
    sample_scores: np.ndarray
    epoch_scores: np.ndarray
    sampling_rate: int
    samples_per_epoch: int
    channels_used: Dict[str, str]
    source_path: Optional[Path] = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({OUTPUT_COLUMN: self.sample_scores})


class OSDScorer:
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else _default_weights_path()
        self.scaler_path = Path(scaler_path) if scaler_path else _default_scaler_path()
        self.sampling_rate = int(sampling_rate)
        self.samples_per_epoch = int(samples_per_epoch)
        self._model = None
        self._scale_params = None

    def score_file(self, path: Path | str) -> OSDResult:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".h5":
            return self.score_h5(path)
        if suffix == ".edf":
            return self.score_edf(path)
        raise ValueError(f"Unsupported input format: {path.suffix}. Expected .h5 or .edf.")

    def score_h5(self, path: Path | str) -> OSDResult:
        path = Path(path)
        eeg_v, channels_used, sampling_rate = load_h5_eeg(path)
        return self._score_eeg(eeg_v=eeg_v, source_path=path, channels_used=channels_used, sampling_rate=sampling_rate)

    def score_edf(self, path: Path | str) -> OSDResult:
        path = Path(path)
        eeg_v, channels_used, sampling_rate = self._load_edf_eeg(path)
        return self._score_eeg(eeg_v=eeg_v, source_path=path, channels_used=channels_used, sampling_rate=sampling_rate)

    def write_csv(self, result: OSDResult, output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_frame().to_csv(output_path, index=False)
        return output_path

    def _score_eeg(
        self,
        eeg_v: np.ndarray,
        source_path: Optional[Path],
        channels_used: Dict[str, str],
        sampling_rate: int,
    ) -> OSDResult:
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"OSD expects {self.sampling_rate} Hz data after preparation, received {sampling_rate} Hz."
            )

        eeg_v = self._preprocess_eeg(eeg_v)
        windows = _epoch_windows(eeg_v * 1e6, samples_per_epoch=self.samples_per_epoch)

        model = self._get_model()
        ordinal_outputs = model.predict(windows, verbose=0)[1]
        epoch_scores = ordinal_outputs[:, 3].astype(np.float32)

        scale_params = self._get_scale_params()
        epoch_scores = (epoch_scores - scale_params["min"]) / scale_params["max"]
        sample_scores = _expand_epoch_scores(
            epoch_scores,
            target_length=eeg_v.shape[1],
            samples_per_epoch=self.samples_per_epoch,
        )

        return OSDResult(
            sample_scores=sample_scores,
            epoch_scores=epoch_scores.astype(np.float32),
            sampling_rate=sampling_rate,
            samples_per_epoch=self.samples_per_epoch,
            channels_used=channels_used,
            source_path=source_path,
        )

    def _get_model(self):
        if self._model is None:
            _load_tensorflow()
            from utils.models.OSD_architecture import OSD_architecture

            self._model = OSD_architecture()
            self._model.load_weights(self.weights_path)
        return self._model

    def _get_scale_params(self) -> Dict[str, float]:
        if self._scale_params is None:
            params = pd.read_pickle(self.scaler_path)
            self._scale_params = {"min": float(params["min"]), "max": float(params["max"])}
        return self._scale_params

    def _load_edf_eeg(self, path: Path) -> Tuple[np.ndarray, Dict[str, str], int]:
        mne, _ = _load_mne()
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        available = {
            _normalize_name(name): raw.get_data(picks=[name])[0].astype(np.float32)
            for name in raw.ch_names
        }
        eeg_v, channels_used = build_eeg_matrix(available, raw.n_times)
        sampling_rate = int(round(raw.info["sfreq"]))
        return eeg_v, channels_used, sampling_rate

    def _preprocess_eeg(self, eeg_v: np.ndarray) -> np.ndarray:
        mne, EOGRegression = _load_mne()

        if eeg_v.shape[0] != len(EEG_CHANNELS):
            raise ValueError(f"Expected {len(EEG_CHANNELS)} EEG channels, received {eeg_v.shape[0]}.")

        ecg = np.zeros((1, eeg_v.shape[1]), dtype=np.float32)
        mastoid = np.zeros((1, eeg_v.shape[1]), dtype=np.float32)
        raw_data = np.vstack([eeg_v, ecg, mastoid])

        info = mne.create_info(
            ch_names=["F3", "F4", "C3", "C4", "O1", "O2", "ECG", "M1"],
            sfreq=self.sampling_rate,
            ch_types=["eeg"] * 6 + ["ecg", "eeg"],
        )
        raw = mne.io.RawArray(raw_data, info, verbose="ERROR")
        raw.set_eeg_reference(ref_channels=["M1"], verbose="ERROR")
        raw.filter(l_freq=0.3, h_freq=30.0, verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="ERROR")

        try:
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, verbose="ERROR")
            if len(ecg_epochs) > 0:
                ecg_evoked = ecg_epochs.average()
                artifact_model = EOGRegression(picks="eeg", picks_artifact="ecg").fit(ecg_evoked)
                raw = artifact_model.apply(raw.copy())
        except Exception:
            pass

        return raw.get_data()[:6, :].astype(np.float32)


def build_eeg_matrix(
    available_signals: Dict[str, np.ndarray],
    target_length: int,
) -> Tuple[np.ndarray, Dict[str, str]]:
    eeg = []
    channels_used: Dict[str, str] = {}

    for target in EEG_CHANNELS:
        source = next((candidate for candidate in _fallback_sources()[target] if candidate in available_signals), None)
        if source is None:
            present = ", ".join(sorted(available_signals))
            raise ValueError(
                "OSD requires at least one left and right EEG source. "
                f"Could not synthesize `{target}` from available channels: {present}."
            )
        eeg.append(_match_length(available_signals[source], target_length))
        channels_used[target] = source

    return np.vstack(eeg).astype(np.float32), channels_used


def read_h5_contents(path: Path | str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "signals" not in handle:
            raise ValueError(f"{path} does not contain a `signals` group.")

        signals = {
            _normalize_name(name): _flatten_dataset(handle["signals"][name])
            for name in handle["signals"].keys()
        }
        annotations = {}
        if "annotations" in handle:
            annotations = {
                _normalize_name(name): _flatten_dataset(handle["annotations"][name])
                for name in handle["annotations"].keys()
            }
        sampling_rate = int(handle.attrs.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    return signals, annotations, sampling_rate


def load_h5_eeg(path: Path | str) -> Tuple[np.ndarray, Dict[str, str], int]:
    signals, _, sampling_rate = read_h5_contents(path)
    if not signals:
        raise ValueError(f"{path} does not contain any signal datasets.")
    target_length = max(signal.shape[0] for signal in signals.values())
    eeg_v, channels_used = build_eeg_matrix(signals, target_length)
    return eeg_v, channels_used, sampling_rate


def score_file(path: Path | str) -> OSDResult:
    return OSDScorer().score_file(path)


def score_samples(path: Path | str) -> np.ndarray:
    return score_file(path).sample_scores


def _default_plot_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.summary.png")


def _default_csv_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.osd.csv")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ordinal Sleep Depth inference on a .h5 or .edf file.")
    parser.add_argument("input_path", type=Path, help="Prepared .h5 or source .edf input file.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--weights", type=Path, default=_default_weights_path(), help="Path to model weights.")
    parser.add_argument("--scaler", type=Path, default=_default_scaler_path(), help="Path to scaling values.")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        default=None,
        help="Write a summary PNG. Without a value, uses <input_stem>.summary.png.",
    )
    parser.add_argument(
        "--plot-channel",
        type=str,
        default="c4-m1",
        help="Preferred channel for the EEG spectrogram in H5 inputs.",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print channel mapping and output sizes.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    scorer = OSDScorer(weights_path=args.weights, scaler_path=args.scaler)
    result = scorer.score_file(args.input_path)

    csv_path = args.output if args.output else _default_csv_path(args.input_path)
    scorer.write_csv(result, csv_path)

    plot_path = None
    if args.plot is not None:
        plot_path = _default_plot_path(args.input_path) if args.plot == "auto" else Path(args.plot)
        try:
            from .plotting import plot_summary
        except ImportError:
            from plotting import plot_summary

        plot_summary(args.input_path, result, plot_path, preferred_channel=args.plot_channel)

    if args.print_summary:
        print(f"input={args.input_path}")
        print(f"csv={csv_path}")
        if plot_path is not None:
            print(f"plot={plot_path}")
        print(f"samples={result.sample_scores.shape[0]}")
        print(f"epochs={result.epoch_scores.shape[0]}")
        print(f"sampling_rate={result.sampling_rate}")
        print(f"channels_used={result.channels_used}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
