from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libero_a2b.config import load_config
from libero_a2b.export_simplevla import export_simplevla_manifest, write_variant_readme


def _copy_tree(src: Path, dst: Path) -> None:
    marker = dst / "verl" / "utils" / "dataset" / "libero_offline_dataset.py"
    if marker.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SimpleVLA-RL binary and dense sibling variants.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "libero_a2b_v1.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    source_root = (ROOT / "SimpleVLA-RL").resolve()
    binary_root = (ROOT / cfg.export.binary_variant_dir).resolve()
    dense_root = (ROOT / cfg.export.dense_variant_dir).resolve()
    _copy_tree(source_root, binary_root)
    _copy_tree(source_root, dense_root)

    binary_manifest = export_simplevla_manifest(
        cfg, "binary", cfg.resolve_path(cfg.paths.binary_dir), binary_root
    )
    dense_manifest = export_simplevla_manifest(
        cfg, "dense", cfg.resolve_path(cfg.paths.shaped_dir), dense_root
    )
    write_variant_readme(binary_root, "binary")
    write_variant_readme(dense_root, "dense")

    print(
        json.dumps(
            {
                "binary_root": str(binary_root),
                "dense_root": str(dense_root),
                "binary_manifest": str(binary_manifest),
                "dense_manifest": str(dense_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
