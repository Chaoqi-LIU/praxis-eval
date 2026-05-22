# Contributing

## Development Setup

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval.git
cd praxis-eval
uv sync --extra dev
uv run --extra dev pre-commit install
uv run --extra dev pytest --strict-markers -m "not manual"
```

## Project Boundary

`praxis-eval` owns benchmark setup, rollout execution, metrics, artifacts,
setup/verification tools, and observation/action contracts. Model-specific
policy preprocessing and checkpoint loading should stay in the caller.

Keep benchmark-specific behavior in the benchmark driver that owns it. Shared
helpers should only move into shared modules when at least two benchmark
families use the same behavior for the same reason.

## License Headers

Add explicit SPDX headers to new manually maintained Python files under `src/`:

```text
SPDX-FileCopyrightText: 2027 Your Name
SPDX-License-Identifier: Apache-2.0
```

Use the current year when creating a new file. Do not update every file just
because the calendar year changed. If an existing file receives copyrightable
changes in a later year, update only that file's year range or copyright holder
when appropriate, for example:

```text
SPDX-FileCopyrightText: 2026-2027 Chaoqi Liu and contributors
SPDX-License-Identifier: Apache-2.0
```

Tests, documentation, workflow files, project config, generated files, lockfiles,
and files that should not be edited directly are licensed through the root
`LICENSE` and `NOTICE` files, but they should not carry visible SPDX headers.

## Checks

Run the same checks as CI before sending changes:

```bash
uv run --extra dev pre-commit run check-license-headers --all-files
uv run --extra dev pre-commit run --all-files
uv run --extra dev pytest --strict-markers -m "not manual"
uv build --sdist --wheel
```
