"""Tests for velari_harness.core.schemas.request.task."""

from dataclasses import FrozenInstanceError

import pytest


def test_task_keeps_examples_in_order():
    from velari_harness.core.schemas.response.record import Example
    from velari_harness.core.schemas.request.task import Task

    examples = [Example("card charged twice", "billing"), Example("login fails", "general")]

    assert Task("ticket-routing", examples).examples == tuple(examples)


def test_task_stores_examples_as_tuple_so_later_changes_to_the_source_do_not_leak_in():
    from velari_harness.core.schemas.response.record import Example
    from velari_harness.core.schemas.request.task import Task

    source = [Example("card charged twice")]
    task = Task("ticket-routing", source)
    source.append(Example("login fails"))

    assert task.examples == (Example("card charged twice"),)


def test_task_empty_name_raises_valueerror():
    from velari_harness.core.schemas.response.record import Example
    from velari_harness.core.schemas.request.task import Task

    with pytest.raises(ValueError):
        Task("", [Example("card charged twice")])


def test_task_is_immutable():
    from velari_harness.core.schemas.response.record import Example
    from velari_harness.core.schemas.request.task import Task

    task = Task("ticket-routing", [Example("card charged twice")])

    with pytest.raises(FrozenInstanceError):
        setattr(task, "version", "2")


def test_task_runs_against_two_candidates_without_knowing_either():
    from velari_harness.core.schemas.request.task import Task
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.response.record import Example, Output

    task = Task("ticket-routing", [Example("Card charged twice for invoice INV-2291"), Example("Login fails with a 500 error")])
    baseline = CallableCandidate(lambda text: "general")
    variant = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")

    baseline_outputs = [r.outcome.output for r in Runner().run(baseline, task.examples)]
    variant_outputs = [r.outcome.output for r in Runner().run(variant, task.examples)]

    assert baseline_outputs == [Output(value="general"), Output(value="general")]
    assert variant_outputs == [Output(value="billing"), Output(value="general")]
