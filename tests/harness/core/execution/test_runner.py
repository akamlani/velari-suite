"""Tests for velari_harness.core.execution.runner."""

import pytest


def _error(record):
    return failure.error if (failure := record.outcome.failure) else None


def _reason(record):
    return failure.reason_code if (failure := record.outcome.failure) else None


def _cases(*inputs):
    from velari_harness.core.schemas.request.case import Case

    return [Case(value) for value in inputs]


def _recording_candidate(fail_run_on=None, fail_setup=False, fail_teardown=False, run_exc=None):
    from velari_harness.core.protocols.candidate import Candidate

    class RecordingCandidate(Candidate):
        def __init__(self):
            self.calls = []

        def setup(self):
            self.calls.append("setup")
            if fail_setup:
                raise RuntimeError("setup failed")

        def run(self, input):
            self.calls.append("run")
            if input == fail_run_on:
                raise (run_exc or ValueError)(f"cannot route {input!r}")
            return input.upper()

        def teardown(self):
            self.calls.append("teardown")
            if fail_teardown:
                raise OSError("teardown failed")

    return RecordingCandidate()


def test_run_returns_one_record_per_input_in_order():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.schemas.types import Status

    records = Runner().run(_recording_candidate(), _cases("card charged twice", "login fails"))

    assert [(r.index, r.case.input, r.outcome.output, r.outcome.status, _error(r)) for r in records] == [
        (0, "card charged twice", "CARD CHARGED TWICE", Status.SUCCESS, None),
        (1, "login fails",        "LOGIN FAILS",        Status.SUCCESS, None),
    ]
    assert all(r.outcome.duration_sec >= 0 for r in records)


def test_run_runs_candidate_lifecycle_once_per_input():
    from velari_harness.core.execution.runner import Runner

    candidate = _recording_candidate()

    Runner().run(candidate, _cases("a", "b"))

    assert candidate.calls == ["setup", "run", "teardown"] * 2


def test_run_without_inputs_returns_empty_list():
    from velari_harness.core.execution.runner import Runner

    assert Runner().run(_recording_candidate(), []) == []


def test_run_failing_input_is_recorded_with_its_error_and_no_output():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.schemas.types import Status

    records = Runner().run(_recording_candidate(fail_run_on="bad"), _cases("ok", "bad", "fine"))

    assert [r.outcome.status for r in records] == [Status.SUCCESS, Status.ERROR, Status.SUCCESS]
    assert (_error(records[1]), records[1].outcome.output) == ("ValueError: cannot route 'bad'", None)


def test_run_failing_input_does_not_stop_later_inputs():
    from velari_harness.core.execution.runner import Runner

    candidate = _recording_candidate(fail_run_on="bad")

    records = Runner().run(candidate, _cases("ok", "bad", "fine"))

    assert records[2].outcome.output == "FINE"
    assert candidate.calls == ["setup", "run", "teardown"] * 3


def test_run_failing_setup_skips_run_and_still_tears_down():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.schemas.types import Status

    candidate = _recording_candidate(fail_setup=True)

    records = Runner().run(candidate, _cases("a"))

    assert candidate.calls == ["setup", "teardown"]
    assert (records[0].outcome.status, _error(records[0])) == (Status.ERROR, "RuntimeError: setup failed")


def test_run_teardown_failure_after_success_is_an_error_and_keeps_output():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.schemas.types import Status

    records = Runner().run(_recording_candidate(fail_teardown=True), _cases("a"))

    assert (records[0].outcome.status, records[0].outcome.output, _error(records[0])) == (Status.ERROR, "A", "OSError: teardown failed")


@pytest.mark.parametrize("candidate_kwargs, expected_reason", [
    ({"fail_setup": True},                        "unexpected"),
    ({"fail_run_on": "a"},                        "invalid_input"),
    ({"fail_teardown": True},                     "unavailable"),
    ({"fail_run_on": "a", "fail_teardown": True}, "invalid_input"),
    ({},                                          None),
])
def test_run_reason_code_is_the_one_of_the_first_failure(candidate_kwargs, expected_reason):
    from velari_harness.core.execution.runner import Runner

    record = Runner().run(_recording_candidate(**candidate_kwargs), _cases("a"))[0]

    assert _reason(record) == expected_reason


@pytest.mark.parametrize("exc, expected_reason", [
    (KeyError("ACC-10293"),            "not_found"),
    (FileNotFoundError("report.csv"),  "not_found"),
    (PermissionError("read denied"),   "permission_denied"),
    (TimeoutError("no reply in 30s"),  "timeout"),
    (ConnectionError("refused"),       "unavailable"),
    (ZeroDivisionError("division"),    "unexpected"),
])
def test_run_classifies_the_exception_type_into_a_reason_code(exc, expected_reason):
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate

    def failing(text):
        raise exc

    record = Runner().run(CallableCandidate(failing), _cases("a"))[0]

    assert _reason(record) == expected_reason


def test_run_trial_error_declares_its_own_reason_code():
    from velari_harness.core.errors import TrialError
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.types import ReasonCode

    def failing(text):
        raise TrialError(ReasonCode.TIMEOUT, "billing API did not answer")

    record = Runner().run(CallableCandidate(failing), _cases("a"))[0]

    assert (_reason(record), _error(record)) == (ReasonCode.TIMEOUT, "TrialError: billing API did not answer")


def test_run_teardown_failure_does_not_mask_earlier_error():
    from velari_harness.core.execution.runner import Runner

    records = Runner().run(_recording_candidate(fail_run_on="bad", fail_teardown=True), _cases("bad"))

    assert _error(records[0]) == "ValueError: cannot route 'bad'"


def test_run_keyboard_interrupt_propagates_after_teardown_runs():
    from velari_harness.core.execution.runner import Runner

    candidate = _recording_candidate(fail_run_on="stop", run_exc=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        Runner().run(candidate, _cases("a", "stop", "never reached"))

    assert candidate.calls == ["setup", "run", "teardown", "setup", "run", "teardown"]


def test_run_passes_only_the_case_input_to_the_candidate():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.request.case import Case

    seen = []

    Runner().run(CallableCandidate(lambda value: seen.append(value) or value), [Case("a", "SECRET")])

    assert seen == ["a"]
