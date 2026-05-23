# Installation

`praxis-eval` supports Python 3.10 and newer.

Install the core package:

```bash
pip install praxis-eval
```

The core install includes the public API, contracts, registry, result types, setup/verify dispatchers, and local policy adapter. It does not install heavy simulator stacks.

## Benchmark Extras

Install extras for the benchmark families you plan to run:

```bash
pip install "praxis-eval[libero]"
pip install "praxis-eval[robocasa]"
pip install "praxis-eval[robomimic]"
pip install "praxis-eval[metaworld]"
pip install "praxis-eval[simpler]"
pip install "praxis-eval[mshab]"
pip install "praxis-eval[remote]"
```

| Extra | Adds | Notes |
| --- | --- | --- |
| `remote` | `praxis-remote>=0.1.0,<0.2.0` | Optional policy transport. |
| `libero` | `praxis-libero`, `praxis-robosuite`, LeRobot support | Runs in the current environment. |
| `robocasa` | `praxis-robocasa`, `praxis-robosuite`, MuJoCo | Requires RoboCasa asset setup. |
| `robomimic` | RoboMimic and `praxis-robosuite` | Runs robosuite tasks in the current environment. |
| `metaworld` | LeRobot MetaWorld dependencies | Runs MetaWorld tasks in the current environment. |
| `simpler` | `praxis-simpler`, `praxis-remote` | Usually evaluates through a dedicated runtime. |
| `mshab` | `praxis-mshab`, `praxis-remote` | Usually evaluates through a dedicated runtime. |

The `all` extra exists for broad integration environments. Prefer narrower extras for policy development because simulator dependencies are heavy and sometimes mutually constraining.

## Setup Commands

Setup commands prepare assets or dedicated simulator runtimes:

```bash
praxis-eval-setup --help
praxis-eval-setup robocasa
praxis-eval-setup simpler
praxis-eval-setup mshab
```

Each setup command also supports `--help`:

```bash
praxis-eval-setup robocasa --help
praxis-eval-setup simpler --help
praxis-eval-setup mshab --help
```

## Verify Commands

Verifier commands run short random-action rollouts. They should be run only on machines that can run the corresponding simulator:

```bash
praxis-eval-verify --help
praxis-eval-verify libero
praxis-eval-verify robocasa
praxis-eval-verify robomimic
praxis-eval-verify metaworld
praxis-eval-verify simpler
praxis-eval-verify mshab
```

For SimplerEnv and MS-HAB, pass either a managed env name or an explicit Python interpreter:

```bash
praxis-eval-verify simpler --env-name simpler-praxis --num-episodes 1 --num-envs 1
praxis-eval-verify mshab --env-name mshab-praxis --num-episodes 1 --num-envs 1

praxis-eval-verify simpler --env-python-bin /path/to/simpler-praxis/bin/python
praxis-eval-verify mshab --env-python-bin /path/to/mshab-praxis/bin/python
```

## Source Install

Use a source install when you need the current branch:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-eval.git
cd praxis-eval
uv sync --extra dev
uv run pytest --strict-markers -m "not manual"
```

Documentation uses VitePress and Node 22:

```bash
cd docs
npm ci
npm run docs:check
npm run docs:build
```
