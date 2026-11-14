"""Tests for velari_harness.core.schemas.request.task."""

from dataclasses import FrozenInstanceError

import pytest


def test_task_keeps_cases_in_order():
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.request.task import Task

    cases = [Case("card charged twice", "billing"), Case("login fails", "general")]

    assert Task("ticket-routing", cases).cases == tuple(cases)


def test_task_stores_cases_as_tuple_so_later_changes_to_the_source_do_not_leak_in():
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.request.task import Task

    source = [Case("card charged twice")]
    task = Task("ticket-routing", source)
    source.append(Case("login fails"))

    assert task.cases == (Case("card charged twice"),)


def test_task_empty_name_raises_valueerror():
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.request.task import Task

    with pytest.raises(ValueError):
        Task("", [Case("card charged twice")])


def test_task_is_immutable():
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.request.task import Task

    task = Task("ticket-routing", [Case("card charged twice")])

    with pytest.raises(FrozenInstanceError):
        setattr(task, "version", "2")


def test_task_runs_against_two_candidates_without_knowing_either():
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.request.task import Task
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = Task("ticket-routing", [Case("Card charged twice for invoice INV-2291"), Case("Login fails with a 500 error")])
    baseline = CallableCandidate(lambda text: "general")
    variant = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")

    baseline_outputs = [r.outcome.output for r in Runner().run(baseline, task.cases)]
    variant_outputs = [r.outcome.output for r in Runner().run(variant, task.cases)]

    assert baseline_outputs == ["general", "general"]
    assert variant_outputs == ["billing", "general"]
