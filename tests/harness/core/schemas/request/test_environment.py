"""Tests for velari_harness.core.schemas.request.environment."""

import random
from dataclasses import FrozenInstanceError
from typing import Any

import pytest


def test_seed_for_returns_pinned_values_so_the_derivation_cannot_change_silently():
    from velari_harness.core.schemas.request.environment import Environment

    environment = Environment(seed=42)

    assert [environment.seed_for(index) for index in range(3)] == [247386905, 2345647074, 368107479]
    assert environment.seed_for(0, trial=1) == 4052739279
    assert environment.seed_for("fold-3") == 3729625937


def test_seed_for_same_environment_and_arguments_give_the_same_seed():
    from velari_harness.core.schemas.request.environment import Environment

    assert Environment(seed=7).seed_for(3, trial=2) == Environment(seed=7).seed_for(3, trial=2)


@pytest.mark.parametrize("other", [
    dict(seed=8, key=3, trial=2),
    dict(seed=7, key=4, trial=2),
    dict(seed=7, key=3, trial=3),
])
def test_seed_for_any_changed_part_gives_a_different_seed(other):
    from velari_harness.core.schemas.request.environment import Environment

    base = Environment(seed=7).seed_for(3, trial=2)

    assert Environment(seed=other["seed"]).seed_for(other["key"], trial=other["trial"]) != base


def test_seed_for_fits_in_32_bits():
    from velari_harness.core.schemas.request.environment import Environment

    environment = Environment()

    assert all(0 <= environment.seed_for(index, trial) < 2**32 for index in range(50) for trial in range(3))


def test_seed_for_leaves_global_random_state_untouched():
    from velari_harness.core.schemas.request.environment import Environment

    random.seed(1)
    expected = random.random()
    random.seed(1)

    Environment().seed_for(3)

    assert random.random() == expected


@pytest.mark.parametrize("seed", ["42", 4.2, None, True])
def test_environment_non_int_seed_raises_typeerror(seed):
    from velari_harness.core.schemas.request.environment import Environment

    not_a_seed: Any = seed

    with pytest.raises(TypeError):
        Environment(seed=not_a_seed)


def test_environment_is_immutable():
    from velari_harness.core.schemas.request.environment import Environment

    environment = Environment()

    with pytest.raises(FrozenInstanceError):
        setattr(environment, "seed", 1)
