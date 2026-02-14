from __future__ import annotations

import bz2
import lzma
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen

import numpy as np


def _urls_for_dataset(dataset: str) -> list[str]:
    # Primary: official LIBSVM datasets site
    if dataset.lower() == "mushrooms":
        return [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
            "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
        ]
    if dataset.lower() == "gisette":
        # 6000 x 5000 training set
        return [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
            "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        ]
    if dataset.lower() == "higgs":
        # 11M x 28, UCI HIGGS (LibSVM binary page: HIGGS.xz)
        return [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz",
            "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz",
        ]
    raise ValueError(f"Unknown dataset for download: {dataset!r}")


def download_raw_libsvm(dataset: str, *, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / dataset
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_err: Optional[Exception] = None
    compressed_path = out_dir / f"{dataset}.xz"
    # If a local .xz exists (e.g. manually downloaded) and dest is missing, decompress it
    if compressed_path.exists() and compressed_path.stat().st_size > 0 and (not dest.exists() or dest.stat().st_size == 0):
        try:
            chunk_size = 1 << 22
            with open(compressed_path, "rb") as src:
                with lzma.open(src, "rb") as dec:
                    with open(dest, "wb") as out:
                        while True:
                            chunk = dec.read(chunk_size)
                            if not chunk:
                                break
                            out.write(chunk)
            return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Remove only partial dest; keep .xz in case it was a manual download
            try:
                if dest.exists():
                    dest.unlink(missing_ok=True)
            except OSError:
                pass
    for url in _urls_for_dataset(dataset):
        try:
            # Large files (e.g. HIGGS.xz): stream to temp then decompress to dest
            timeout = 3600 if ".xz" in url or dataset.lower() == "higgs" else 60
            with urlopen(url, timeout=timeout) as r:
                if url.endswith(".xz"):
                    chunk_size = 1 << 22  # 4 MiB
                    try:
                        with open(compressed_path, "wb") as f:
                            while True:
                                chunk = r.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                        with open(compressed_path, "rb") as src:
                            with lzma.open(src, "rb") as dec:
                                with open(dest, "wb") as out:
                                    while True:
                                        chunk = dec.read(chunk_size)
                                        if not chunk:
                                            break
                                        out.write(chunk)
                        # Success: remove archive so a failed unlink doesn't trigger retry
                        try:
                            compressed_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return dest
                    except Exception as xz_err:
                        # Remove partial/corrupt files so retry or next run gets a clean slate
                        for p in (compressed_path, dest):
                            try:
                                if p.exists():
                                    p.unlink(missing_ok=True)
                            except OSError:
                                pass
                        last_err = xz_err
                        continue
                if url.endswith(".bz2"):
                    data = r.read()
                    if not data:
                        raise RuntimeError(f"Empty download from {url}")
                    data = bz2.decompress(data)
                    dest.write_bytes(data)
                    return dest
                # Uncompressed
                data = r.read()
                if not data:
                    raise RuntimeError(f"Empty download from {url}")
                dest.write_bytes(data)
                return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    hint = ""
    if isinstance(last_err, EOFError):
        hint = (
            " The compressed file was truncated (incomplete download). "
            "Try again with a stable connection, or download the .xz file manually (e.g. with a browser or wget) "
            "and place it in the dataset folder, then re-run. For Higgs: "
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz → "
            f"{compressed_path!s}"
        )
    raise RuntimeError(f"Failed to download dataset {dataset!r}.{hint}") from last_err


def parse_libsvm_dense(path: Path, *, n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a LIBSVM-format file into dense (X, y).
    Labels are converted to y in {-1, +1} when needed.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty dataset file: {path}")

    # Determine label encoding from the file (important: some datasets are {-1,+1}, others {1,2}, etc.)
    label_set = set()
    max_j = 0
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        label_set.add(float(parts[0]))
        for item in parts[1:]:
            j, _ = item.split(":")
            max_j = max(max_j, int(j))

    if n_features is None:
        n_features = max_j

    if label_set.issubset({-1.0, 1.0}):
        def map_y(v: float) -> float:
            return float(v)
    elif label_set.issubset({0.0, 1.0}):
        def map_y(v: float) -> float:
            return float(2.0 * v - 1.0)  # {0,1}->{-1,+1}
    elif label_set.issubset({1.0, 2.0}):
        def map_y(v: float) -> float:
            return float(2.0 * v - 3.0)  # {1,2}->{-1,+1}
    else:
        raise ValueError(f"Unsupported label set in {path}: {sorted(label_set)}")

    n = len(lines)
    X = np.zeros((n, int(n_features)), dtype=float)
    y = np.zeros((n,), dtype=float)

    for i, line in enumerate(lines):
        parts = line.split()
        y[i] = map_y(float(parts[0]))
        for item in parts[1:]:
            j, v = item.split(":")
            X[i, int(j) - 1] = float(v)

    return X, y


def ensure_dataset_npy(dataset: str, *, base_dir: Path) -> Path:
    """
    Ensure `<base_dir>/<dataset>/<dataset>.npy` exists, downloading & converting if needed.
    Saves an object array of shape (n,2) where row is (x_i, y_i).
    """
    ds_dir = base_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    npy_path = ds_dir / f"{dataset}.npy"
    # If it exists, we still validate labels are in {-1,+1} AND both classes exist.
    # If not, rebuild.
    if npy_path.exists():
        try:
            thetas = np.load(str(npy_path), allow_pickle=True)
            y = np.asarray(thetas[:, 1], dtype=float)
            u = set(np.unique(y).tolist())
            if u == {-1.0, 1.0}:
                return npy_path
        except Exception:
            pass

    raw_path = download_raw_libsvm(dataset, out_dir=ds_dir)
    X, y = parse_libsvm_dense(raw_path)

    thetas = np.empty((X.shape[0], 2), dtype=object)
    thetas[:, 0] = list(X)
    thetas[:, 1] = list(y)
    # Ensure pickle format; this mirrors how v1 expects to load these datasets
    np.save(str(npy_path), thetas, allow_pickle=True)
    return npy_path

