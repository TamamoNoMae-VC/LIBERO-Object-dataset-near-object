from __future__ import annotations

import argparse
from pathlib import Path


def patch_file(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_robosuite(robosuite_root: Path) -> list[str]:
    changed = []
    targets = [
        robosuite_root / "robosuite" / "models" / "arenas" / "multi_table_arena.py",
    ]
    for path in targets:
        if not path.exists():
            continue
        if patch_file(path, "from collections import Iterable", "from collections.abc import Iterable"):
            changed.append(str(path))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch known Python 3.12 compatibility issues in Colab dependencies.")
    parser.add_argument("--robosuite-root", required=True, help="Path to the cloned robosuite repository root.")
    args = parser.parse_args()

    changed = patch_robosuite(Path(args.robosuite_root).resolve())
    if changed:
        print("Patched files:")
        for item in changed:
            print(item)
    else:
        print("No robosuite compatibility patches were applied.")


if __name__ == "__main__":
    main()
