# LIBERO A-to-B Dataset Pipeline

This workspace now includes a Colab-first pipeline for collecting **successful-only** LIBERO trajectories where one fixed object is moved from randomized point `A` to randomized point `B` near one fixed reference object.

## What It Produces

- `data/raw/`: canonical successful trajectories saved once
- `data/dataset_binary_reward/`: same trajectories with binary rewards
- `data/dataset_shaped_reward/`: same trajectories with shaped rewards and RTG fields
- `results/metrics/`: collection and validation summaries
- `results/reports/`: resolved task and validation reports
- `results/plots/`: reserved for analysis outputs
- `results/rollouts/`: rollout videos
- `SimpleVLA-RL-binary/`: binary reward trainer variant
- `SimpleVLA-RL-dense/`: dense reward trainer variant

## Colab Installation

Run the following in a fresh Colab runtime after enabling a GPU:

```bash
!git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
!git clone https://github.com/moojink/openvla-oft.git
!git clone https://github.com/ARISE-Initiative/robosuite.git
!cd robosuite && git checkout v1.4.1_libero
!cd robosuite && pip install -r requirements.txt
!cd robosuite && pip install -r requirements-extra.txt
!python /content/datasetlibero2/scripts/patch_colab_compat.py --robosuite-root /content/datasetlibero2/robosuite
!cd robosuite && pip install -e .
!pip install -e LIBERO
!pip install -e openvla-oft
!pip install torch torchvision h5py imageio pyyaml tensorflow
```

Do not use `pip install robosuite` latest here. Newer robosuite releases removed the `SingleArmEnv` import path that LIBERO currently expects.
On Colab Python 3.12, the LIBERO-compatible robosuite branch also needs a small patch replacing `from collections import Iterable` with `from collections.abc import Iterable`. The script above applies that patch automatically.

## Execution Order

1. `python scripts/collect_raw.py --preflight-only`
2. `python scripts/collect_raw.py --config configs/libero_a2b_v1.yaml`
3. `python scripts/validate_raw.py --config configs/libero_a2b_v1.yaml`
4. `python scripts/build_binary_dataset.py --config configs/libero_a2b_v1.yaml`
5. `python scripts/build_dense_dataset.py --config configs/libero_a2b_v1.yaml`
6. `python scripts/validate_exports.py --config configs/libero_a2b_v1.yaml`
7. `python scripts/export_rollouts.py --config configs/libero_a2b_v1.yaml`
8. `python scripts/prepare_simplevla_variants.py --config configs/libero_a2b_v1.yaml`

## Important Defaults

- One global `master_seed` controls the full collection.
- Saved episodes are **successful only**.
- `A/B` relations are limited to `left`, `right`, `front`, `back`.
- Episode diversity comes from deterministic per-attempt RNG under the one master seed.
- Reward annotations differ between derived datasets, but trajectory content and ordering remain identical.

## Main Reports

- `results/metrics/collection_summary.json`
- `results/reports/raw_validation_report.json`
- `results/reports/export_validation_report.json`
- `results/reports/resolved_task.json`

## Notes

- The collector resolves a task from `libero_object` dynamically at runtime.
- The preferred scene is one containing `alphabet_soup` and `basket`.
- If the installed LIBERO checkout does not expose a compatible task, the resolver fails instead of silently switching task family.
