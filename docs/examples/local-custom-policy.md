# Local Custom Policy

Wrap an in-process policy with `LocalPolicy`. The wrapped object can be callable or expose an `act(...)` method.

```python
import numpy as np

from praxis_eval import EvalConfig, LocalPolicy, evaluate


class MyPolicy:
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        # Load your model here. praxis-eval does not own checkpoints.
        return object()

    def reset(self, episode_ids=None) -> None:
        # Reset recurrent state, caches, or per-episode bookkeeping here.
        pass

    def act(self, observations, *, action_spec=None, policy_kwargs=None, episode_ids=None):
        actions = []
        for obs in observations:
            task = obs["task"]
            image = obs.get("observation.images.image")
            state = obs.get("observation.state")
            action = self._predict(task=task, image=image, state=state)
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def _predict(self, *, task, image, state):
        # Convert praxis-eval observations into your model's input format.
        del task, image, state
        return np.zeros((7,), dtype=np.float32)


result = evaluate(
    "libero",
    policy=LocalPolicy(MyPolicy("checkpoint.pt")),
    config=EvalConfig(
        task="libero_10",
        task_ids=(0,),
        num_eval_per_task=5,
        num_parallel_env=1,
        output_dir="eval/libero_custom",
        policy_kwargs={"decode_temperature": 0.0},
    ),
)

print(result.overall)
```

The adapter receives `policy_kwargs` exactly as passed in `EvalConfig`. Use this for inference-time options such as decoding parameters. Do not use it for benchmark configuration; benchmark settings belong in `env_kwargs`.
