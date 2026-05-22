from __future__ import annotations

import numpy as np
import pytest

from praxis_eval.contracts import ActionSpec


def test_action_spec_validates_shape_dtype_and_bounds() -> None:
    spec = ActionSpec(shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0)
    action = np.asarray([0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(spec.validate(action), action)


def test_action_spec_rejects_invalid_action() -> None:
    spec = ActionSpec(shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0)

    with pytest.raises(ValueError):
        spec.validate(np.asarray([2.0, 0.0], dtype=np.float32))


def test_action_spec_rejects_non_finite_action() -> None:
    spec = ActionSpec(shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0)

    with pytest.raises(ValueError):
        spec.validate(np.asarray([np.nan, 0.0], dtype=np.float32))


def test_action_spec_shape_none_still_validates_dtype_and_range() -> None:
    spec = ActionSpec(shape=None, dtype="float32", minimum=-1.0, maximum=1.0)

    with pytest.raises(TypeError):
        spec.validate_batch(np.asarray([[0, 1]], dtype=np.int32), batch_size=1)

    with pytest.raises(ValueError):
        spec.validate_batch(np.asarray([[2.0]], dtype=np.float32), batch_size=1)
