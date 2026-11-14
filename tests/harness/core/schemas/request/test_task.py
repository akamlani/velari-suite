"""Tests for velari_harness.core.schemas.request.task."""

from dataclasses import FrozenInstanceError

import pytest


def test_task_keeps_inputs_in_order():
    from velari_harness.core.schemas.request.task import Task

    assert Task("ticket-routing", ["card charged twice", "login fails"]).inputs == ("card charged twice", "login fails")


def test_task_stores_inputs_as_tuple_so_later_changes_to_the_source_do_not_leak_in():
    from velari_harness.core.schemas.request.task import Task

    source = ["card charged twice"]
    task = Task("ticket-routing", source)
    source.append("login fails")

    assert task.inputs == ("card charged twice",)


def test_task_empty_name_raises_valueerror():
    from velari_harness.core.schemas.request.task import Task

    with pytest.raises(ValueError):
        Task("", ["card charged twice"])


def test_task_is_immutable():
    from velari_harness.core.schemas.request.task import Task

    task = Task("ticket-routing", ["card charged twice"])

    with pytest.raises(FrozenInstanceError):
        setattr(task, "version", "2")


def test_task_runs_against_two_candidates_without_knowing_either():
    from velari_harness.core.schemas.request.task import Task
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = Task("ticket-routing", ["Card charged twice for invoice INV-2291", "Login fails with a 500 error"])
    baseline = CallableCandidate(lambda text: "general")
    variant = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")

    baseline_outputs = [r.outcome.output for r in Runner().run(baseline, task.inputs)]
    variant_outputs = [r.outcome.output for r in Runner().run(variant, task.inputs)]

    assert baseline_outputs == ["general", "general"]
    assert variant_outputs == ["billing", "general"]


def test_task_references_are_stored_as_a_tuple():
    from velari_harness.core.schemas.request.task import Task

    task = Task("ticket-routing", ["Card charged twice", "Login fails"], references=["billing", "general"])

    assert task.references == ("billing", "general")


def test_task_references_with_a_different_length_raises_valueerror():
    from velari_harness.core.schemas.request.task import Task

    with pytest.raises(ValueError):
        Task("ticket-routing", ["Card charged twice", "Login fails"], references=["billing"])
