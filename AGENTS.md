# Repository Guidelines

## Project Structure

- `starVLA/`: models (`model/`), data pipelines (`dataloader/`), trainers (`training/`), and configs (`config/`).
- `examples/<benchmark>/`: training/evaluation scripts, configs, and dataset registrations.
- `deployment/model_server/`: inference services. `scripts/data/`: data conversion. `scripts/test/`: regression tests.
- `docs/`: guides.

## Coding Guidelines

- Keep model changes within model components when possible; modify shared trainers, datasets, or dataloaders only when necessary.
- Write concise, clear, and elegant code. Avoid excessive defensive checks and redundant validation; check for concrete failure modes.
- Use four-space indentation, `snake_case` for functions/variables, and `PascalCase` for classes. Preserve existing framework filenames such as `QwenOFT.py`.
- Target Python 3.10 with a 121-character line limit. Keep formatting changes scoped.

## Validation

Run commands from the repository root:

- Check changed Python files with `black --check <files>` and `ruff check <files>`.
- Use `unittest` with `test_*.py` files and `test_*` methods; add regression coverage for behavior changes.
- Preserve the `scripts/test/*` section in `.gitignore`. You may add local test scripts, but do not commit them.
- Run relevant tests, e.g. `python -m unittest scripts.test.test_infersystem_protocol -v`. Record prerequisites and skips for GPU/parity tests.

## Commits & Pull Requests

- Use `[type] 中文说明` commit subjects with a lowercase type, e.g. `[fix] 修复动作归一化`; technical identifiers may remain in English.
- When committing changes, use the current branch unless the user explicitly requests a new branch.
- When a new branch is requested, branch from `starVLA_dev` and use names such as `feat/my-feature` or `fix/action-shape`; target `starVLA_dev` for pull requests.
- Include motivation, a linked issue, changes, validation, and compatibility impacts in pull requests.
- Framework, benchmark, and core training/dataloader changes require benchmark results, a public checkpoint, an example config, and reproduction instructions.

## Configuration & Artifacts

- Configure dataset/model paths locally. Keep secrets, datasets, checkpoints, and logs out of commits; use `playground/` and `results/` for local artifacts.
- Clone external source dependencies beside this repository; do not commit local reference checkouts.
