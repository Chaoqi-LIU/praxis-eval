# Local Vs Remote Policy

`praxis-eval` supports two policy execution modes.

## Local Policy

Local mode runs the evaluator and policy adapter in one process:

```python
from praxis_eval import LocalPolicy

policy = LocalPolicy(my_policy)
```

`my_policy` may be a callable or an object with an `act(...)` method. If it defines `reset(...)`, the adapter calls it before rollout waves.

Local mode is the simplest option when the simulator stack and policy stack can coexist in one Python environment.

## Remote Policy

Remote mode uses `praxis-remote` as an optional transport layer:

```python
from praxis_eval import RemotePolicy

policy = RemotePolicy("127.0.0.1:50051", timeout=30.0)
```

The remote server receives the same observation mappings and returns the same batched numpy action contract. The transport package is separate from `praxis-eval`: <https://github.com/Chaoqi-LIU/praxis-remote>.

Remote mode is useful when:

- the policy runtime and simulator runtime need incompatible dependencies;
- the policy should run on a different host or accelerator;
- a dedicated simulator runtime calls back into a policy server, as with SimplerEnv and MS-HAB.

Install the remote extra only when needed:

```bash
pip install "praxis-eval[remote]"
```

`praxis-remote` is optional transport, not a required part of the core evaluator.
