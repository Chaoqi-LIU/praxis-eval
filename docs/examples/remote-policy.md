# Remote Policy

Remote evaluation uses `praxis-remote` to keep the evaluator and policy runtime separate.

Install the optional transport:

```bash
pip install "praxis-eval[remote]==0.1.1"
```

On the evaluator side:

```python
from praxis_eval import EvalConfig, RemotePolicy, evaluate

result = evaluate(
    "robocasa",
    policy=RemotePolicy("127.0.0.1:50051", timeout=30.0),
    config=EvalConfig(
        task="CloseToasterOvenDoor",
        num_eval_per_task=5,
        num_parallel_env=1,
        output_dir="eval/robocasa_remote",
    ),
)

print(result.overall)
```

The policy server should expose the `praxis-remote` policy server contract and return batched numpy actions for the received observation mappings. `praxis-remote` is maintained separately: <https://github.com/Chaoqi-LIU/praxis-remote>.

Remote mode is the preferred path when simulator and policy dependencies cannot be installed in the same environment. It is also the internal transport used by dedicated SimplerEnv and MS-HAB subprocess evaluation.
