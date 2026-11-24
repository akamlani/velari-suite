"""Tests for velari_harness.core.protocols.candidate."""

from typing import Any

import pytest


@pytest.fixture
def ctx(tmp_path):
    from velari_harness.core.protocols.candidate import TrialContext

    return TrialContext(seed=0, workdir=tmp_path)


class TestCallableCandidate:
    @pytest.mark.parametrize("fn, value, expected", [
        (str.upper,                          "card charged twice", "CARD CHARGED TWICE"),
        (lambda ticket: len(ticket.split()), "login fails again",  3),
        (sorted,                             [3, 1, 2],            [1, 2, 3]),
    ])
    def test_run_returns_function_result(self, fn, value, expected, ctx):
        from velari_harness.core.protocols.candidate import CallableCandidate

        assert CallableCandidate(fn).run(value, ctx) == expected

    def test_run_propagates_exceptions(self, ctx):
        from velari_harness.core.protocols.candidate import CallableCandidate

        def failing(ticket):
            raise ValueError("unreadable ticket")

        with pytest.raises(ValueError):
            CallableCandidate(failing).run("any", ctx)


class TestCandidate:
    def test_subclass_with_only_run_inherits_noop_lifecycle(self, ctx):
        from velari_harness.core.protocols.candidate import Candidate

        class EchoCandidate(Candidate):
            def run(self, input, ctx):
                return input

        candidate = EchoCandidate()

        assert candidate.setup(ctx) is None
        assert candidate.run("ticket", ctx) == "ticket"
        assert candidate.teardown(ctx) is None

    def test_subclass_overriding_lifecycle_runs_hooks_in_order(self, ctx):
        from velari_harness.core.protocols.candidate import Candidate

        class RecordingCandidate(Candidate):
            def __init__(self):
                self.calls = []

            def setup(self, ctx):
                self.calls.append("setup")

            def run(self, input, ctx):
                self.calls.append("run")
                return input

            def teardown(self, ctx):
                self.calls.append("teardown")

        candidate = RecordingCandidate()
        candidate.setup(ctx)
        candidate.run("ticket", ctx)
        candidate.teardown(ctx)

        assert candidate.calls == ["setup", "run", "teardown"]

    def test_subclass_without_run_raises_typeerror(self):
        from velari_harness.core.protocols.candidate import Candidate

        class Incomplete(Candidate):
            pass

        incomplete_cls: Any = Incomplete

        with pytest.raises(TypeError):
            incomplete_cls()


class TestTrialContext:
    def test_emit_records_events_in_order_with_nondecreasing_offsets(self, tmp_path):
        from velari_harness.core.protocols.candidate import TrialContext

        ctx = TrialContext(seed=7, workdir=tmp_path)
        ctx.emit("skill_triggered", {"skill": "write-invoice"})
        ctx.emit("answered")

        first, second = ctx.events
        assert (first.name, first.data, second.name) == ("skill_triggered", {"skill": "write-invoice"}, "answered")
        assert 0 <= first.elapsed_sec <= second.elapsed_sec

    def test_events_returns_a_copy(self, tmp_path):
        from velari_harness.core.protocols.candidate import TrialContext

        ctx = TrialContext(seed=7, workdir=tmp_path)
        ctx.emit("answered")
        ctx.events.clear()

        assert [event.name for event in ctx.events] == ["answered"]

    def test_emit_copies_details_so_later_changes_do_not_alter_the_event(self, ctx):
        tools = ["lookup"]
        ctx.emit("step", {"tools": tools})
        tools.append("refund")

        assert ctx.events[0].data == {"tools": ["lookup"]}

    def test_events_returns_independent_events(self, ctx):
        ctx.emit("step", {"tools": ["lookup"]})
        ctx.events[0].data["tools"].append("refund")

        assert ctx.events[0].data == {"tools": ["lookup"]}
