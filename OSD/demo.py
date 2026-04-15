from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

try:
    from .inference import OSDScorer
    from .plotting import plot_summary
except ImportError:
    from inference import OSDScorer
    from plotting import plot_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sample_file() -> Path:
    return _repo_root() / "sample-file.h5"


def _default_csv() -> Path:
    return _repo_root() / "sample-file.osd.csv"


def _default_plot() -> Path:
    return _repo_root() / "sample-file.summary.png"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the packaged OSD demo on sample-file.h5.")
    parser.add_argument("--input", type=Path, default=_sample_file(), help="Sample or custom H5 file.")
    parser.add_argument("--csv", type=Path, default=_default_csv(), help="Output CSV path.")
    parser.add_argument("--plot", type=Path, default=_default_plot(), help="Output PNG path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    scorer = OSDScorer()
    result = scorer.score_h5(args.input)
    scorer.write_csv(result, args.csv)
    plot_summary(args.input, result, args.plot)
    print(f"input={args.input}")
    print(f"csv={args.csv}")
    print(f"plot={args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
