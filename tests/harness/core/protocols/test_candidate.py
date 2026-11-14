"""Tests for velari_harness.core.protocols.candidate."""

from typing import Any

import pytest


@pytest.mark.parametrize("fn, value, expected", [
    (str.upper,                          "card charged twice", "CARD CHARGED TWICE"),
    (lambda ticket: len(ticket.split()), "login fails again",  3),
    (sorted,                             [3, 1, 2],            [1, 2, 3]),
])
def test_callable_candidate_run_returns_function_result(fn, value, expected):
    from velari_harness.core.protocols.candidate import CallableCandidate

    assert CallableCandidate(fn).run(value) == expected


def test_callable_candidate_run_propagates_exceptions():
    from velari_harness.core.protocols.candidate import CallableCandidate

    def failing(ticket):
        raise ValueError("unreadable ticket")

    with pytest.raises(ValueError):
        CallableCandidate(failing).run("any")


def test_subclass_with_only_run_inherits_noop_lifecycle():
    from velari_harness.core.protocols.candidate import Candidate

    class EchoCandidate(Candidate):
        def run(self, input):
            return input

    candidate = EchoCandidate()

    assert candidate.setup() is None
    assert candidate.run("ticket") == "ticket"
    assert candidate.teardown() is None


def test_subclass_overriding_lifecycle_runs_hooks_in_order():
    from velari_harness.core.protocols.candidate import Candidate

    class RecordingCandidate(Candidate):
        def __init__(self):
            self.calls = []

        def setup(self):
            self.calls.append("setup")

        def run(self, input):
            self.calls.append("run")
            return input

        def teardown(self):
            self.calls.append("teardown")

    candidate = RecordingCandidate()
    candidate.setup()
    candidate.run("ticket")
    candidate.teardown()

    assert candidate.calls == ["setup", "run", "teardown"]


def test_subclass_without_run_raises_typeerror():
    from velari_harness.core.protocols.candidate import Candidate

    class Incomplete(Candidate):
        pass

    incomplete_cls: Any = Incomplete

    with pytest.raises(TypeError):
        incomplete_cls()
