from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.common.names import ManuscriptNames


def export_chain_size_pdf(*, dataset: str, outdir: Path) -> Path:
    base_dir = Path(__file__).resolve().parent
    src = base_dir / dataset / "sgd_mice" / "0" / "sgd_mice_chain_size.pdf"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing seed-0 chain-size plot for {dataset!r}: {src}\n"
            f"Generate it with: python -m experiments.logistic_regression.run_logreg --dataset {dataset} --seed 0"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / ManuscriptNames(dataset).logistic_chain_size_pdf
    shutil.copyfile(src, dst)
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description="Export logistic-regression chain-size PDFs with manuscript filenames.")
    p.add_argument("--dataset", type=str, required=True, help="mushrooms|gisette|higgs")
    default_outdir = Path(__file__).resolve().parents[1] / "output"
    p.add_argument("--outdir", type=str, default=str(default_outdir), help="Destination directory")
    args = p.parse_args()

    out = Path(args.outdir).expanduser().resolve()
    dst = export_chain_size_pdf(dataset=args.dataset, outdir=out)
    print(f"Wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
