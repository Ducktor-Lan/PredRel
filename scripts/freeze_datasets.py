"""Seed the unified benchmark cache and lock it (frozen nine + 11 bootstrap).

The nine frozen datasets are never downloaded: they are byte-copied from a
verified seed cache and checked against the unified manifest pins. The
eleven extension datasets are bootstrapped explicitly with --allow-download
BEFORE any Teacher run; afterwards they are pinned like the frozen nine
and the runner refuses any download. Any mismatch refuses instead of
mutating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(_project_root() / "provenance" / "dataset_manifest_benchmark.json"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seed-cache-dir", required=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _copy_verified(spec: dict[str, object], *, seed_dir: Path, cache_dir: Path) -> None:
    dataset_id = str(spec["dataset_id"])
    expected = str(spec.get("expected_content_sha256", ""))
    if not expected:
        raise ValueError(dataset_id + ": frozen manifest lacks an expected content hash")
    for suffix in (".npz", ".metadata.json"):
        source = seed_dir / (dataset_id + suffix)
        destination = cache_dir / (dataset_id + suffix)
        if not source.is_file():
            raise ValueError(dataset_id + ": seed cache entry is missing: " + str(source))
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(dataset_id + ": cache entry differs from the verified seed")
            continue
        shutil.copyfile(source, destination)


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(_project_root() / "src"))
    from predrel.data import DatasetError, ensure_dataset_lock, load_manifest, materialize_manifest

    try:
        manifest, manifest_hash = load_manifest(args.manifest)
        seed_dir = Path(args.seed_cache_dir).resolve()
        cache_dir = Path(args.cache_dir).resolve()
        if seed_dir == cache_dir:
            raise ValueError("seed cache and benchmark cache must be different directories")
        # The eleven extension datasets are exactly the manifest IDs beyond
        # the first nine (frozen order preserved).
        extension_ids = set(str(item["dataset_id"]) for item in manifest["datasets"][9:])
        if len(extension_ids) != 11:
            raise ValueError("frozen manifest must carry exactly eleven extension datasets")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for spec in manifest["datasets"]:
            dataset_id = str(spec["dataset_id"])
            if dataset_id in extension_ids:
                continue
            _copy_verified(spec, seed_dir=seed_dir, cache_dir=cache_dir)
        datasets = materialize_manifest(
            manifest, cache_dir=cache_dir, allow_download=bool(args.allow_download)
        )
        for dataset in datasets:
            pinned = next(
                str(item.get("expected_content_sha256", ""))
                for item in manifest["datasets"]
                if str(item["dataset_id"]) == dataset.dataset_id
            )
            observed = str(dataset.metadata.get("content_sha256", ""))
            if not pinned or observed != pinned:
                raise ValueError(dataset.dataset_id + ": content differs from the frozen pin")
        lock = ensure_dataset_lock(manifest, manifest_hash, datasets, cache_dir=cache_dir)
    except (DatasetError, ValueError, OSError) as error:
        print("DATASET_SEED_ERROR: " + str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "locked",
                "dataset_manifest_sha256": manifest_hash,
                "cache_dir": str(cache_dir),
                "seed_cache_dir": str(seed_dir),
                "extension_dataset_ids": sorted(extension_ids),
                "dataset_ids": [d.dataset_id for d in datasets],
                "dataset_lock": lock,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
