# Contributing

Contributor setup:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval.git
cd praxis-eval
uv sync --extra dev
uv run --extra dev pre-commit install
uv run --extra dev pytest --strict-markers -m "not manual"
```

Documentation setup uses VitePress and Node 22:

```bash
cd docs
npm ci
npm run docs:check
npm run docs:dev
npm run docs:build
```

## Project Boundary

`praxis-eval` owns benchmark setup, rollout execution, metrics, artifacts, setup and verification tools, and observation/action contracts. Model-specific policy preprocessing, checkpoint loading, and training code should stay in the caller.

Keep benchmark-specific behavior inside the benchmark driver that owns it. Move helpers into shared modules only when at least two benchmark families use the same behavior for the same reason.

## License Headers

Add SPDX headers to new manually maintained Python files under `src/`:

```text
SPDX-FileCopyrightText: 2027 Your Name
SPDX-License-Identifier: Apache-2.0
```

Tests, documentation, workflow files, project config, generated files, and lockfiles are licensed through the root `LICENSE` and `NOTICE` files and should not carry visible SPDX headers.

## Checks

Run the same checks as CI before sending changes:

```bash
uv run --extra dev pre-commit run check-license-headers --all-files
uv run --extra dev pre-commit run --all-files
uv run --extra dev pytest --strict-markers -m "not manual"
uv build --sdist --wheel
(cd docs && npm ci && npm run docs:check && npm run docs:build)
```

See [Adding Benchmarks](development/adding-benchmarks.md) for the benchmark-driver development workflow.
