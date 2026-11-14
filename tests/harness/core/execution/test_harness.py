"""Tests for velari_harness.core.execution.harness."""

import pytest
from    typing  import Dict, Optional
from    velari_harness.core.schemas.response.record import Score, TrialRecord


def _recording_candidate():
    from velari_harness.core.protocols.candidate import Candidate

    class RecordingCandidate(Candidate):
        def run(self, input):
            if input == "bad":
                raise ValueError("unreadable ticket")
            return input.upper()

    return RecordingCandidate()


def _task(name, inputs, references=None):
    from velari_harness.core.schemas.response.record import Example
    from velari_harness.core.schemas.request.task import Task

    return Task(name, [Example(value, reference) for value, reference in zip(inputs, references or [None] * len(inputs))])


def _exact_match(output, reference):
    return output == reference


def _values(record: TrialRecord) -> Dict[str, Optional[float]]:
    return {name: score.value for name, score in record.scores.items()}


def _error(score: Score) -> Optional[str]:
    return failure.error if (failure := score.failure) else None


def test_for_task_names_the_harness_after_the_task_and_an_override_wins():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = _task("ticket-routing", ["a"])
    candidate = CallableCandidate(str.upper)

    assert Harness.for_task(task, candidate).name == "ticket-routing"
    assert Harness.for_task(task, candidate, name="nightly-check").name == "nightly-check"


@pytest.mark.parametrize("inputs, expected_trials", [
    (["ok", "bad", "fine"], {"success": 2, "error": 1}),
    (["bad", "bad"],        {"success": 0, "error": 2}),
])
def test_run_counts_trials_per_status_and_still_returns_every_record(inputs, expected_trials):
    from velari_harness.core.execution.harness import Harness

    result = Harness.for_task(_task("ticket-routing", inputs), _recording_candidate()).run()

    assert (len(result.records), result.trials) == (len(inputs), expected_trials)


def test_for_task_with_scorers_scores_each_successful_trial():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = _task("ticket-routing", ["card charged twice", "login fails"], references=["CARD CHARGED TWICE", "wrong"])

    result = Harness.for_task(task, CallableCandidate(str.upper), scorers=[_exact_match]).run()

    assert [_values(r) for r in result.records] == [{"_exact_match": 1.0}, {"_exact_match": 0.0}]
    assert result.scores == {"_exact_match": 0.5}


def test_run_failed_trial_has_no_scores():
    from velari_harness.core.execution.harness import Harness

    task = _task("ticket-routing", ["bad", "ok"], references=["BAD", "OK"])

    result = Harness.for_task(task, _recording_candidate(), scorers=[_exact_match]).run()

    assert [_values(r) for r in result.records] == [{}, {"_exact_match": 1.0}]


def test_run_raising_scorer_gives_a_score_with_its_failure_and_later_trials_still_run():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    def judge(output, reference):
        raise RuntimeError("judge unavailable")

    task = _task("ticket-routing", ["a", "b"], references=["A", "B"])

    result = Harness.for_task(task, CallableCandidate(str.upper), scorers=[judge]).run()

    assert [_values(r) for r in result.records] == [{"judge": None}, {"judge": None}]
    assert [_error(r.scores["judge"]) for r in result.records] == ["RuntimeError: judge unavailable"] * 2
    assert result.scores == {"judge": None}


def test_run_keeps_a_score_a_scorer_returns_with_its_explanation_and_metadata():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.response.record import Score

    def relevance(output, reference):
        return Score(value=0.5, passed=False, label="partial", explanation="cites 1 of 2 sources", metadata={"judge": "relevance-v2"})

    task = _task("retrieval", ["a"], references=["A"])

    result = Harness.for_task(task, CallableCandidate(str.upper), scorers=[relevance]).run()

    assert result.records[0].scores["relevance"] == Score(
        value=0.5, passed=False, label="partial", explanation="cites 1 of 2 sources", metadata={"judge": "relevance-v2"})


def test_run_a_bool_score_sets_value_and_passed():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.response.record import Score

    task = _task("ticket-routing", ["a"], references=["A"])

    result = Harness.for_task(task, CallableCandidate(str.upper), scorers=[_exact_match]).run()

    assert result.records[0].scores["_exact_match"] == Score(value=1.0, passed=True)


def test_score_applies_the_scorers_to_records_that_were_already_run():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = _task("ticket-routing", ["a"], references=["A"])
    harness = Harness.for_task(task, CallableCandidate(str.upper), scorers=[_exact_match])
    unscored = harness.run().records[0].model_copy(update={"scores": {}})

    assert _values(harness.score([unscored])[0]) == {"_exact_match": 1.0}


@pytest.mark.parametrize("scorers, references", [
    ([_exact_match],                ()),
    ([_exact_match, _exact_match],  ["A"]),
])
def test_for_task_invalid_scorers_raise_valueerror(scorers, references):
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    with pytest.raises(ValueError):
        Harness.for_task(_task("ticket-routing", ["a"], references=references), CallableCandidate(str.upper), scorers=scorers)


def test_for_task_scorers_with_a_case_missing_its_reference_raise_valueerror():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate

    task = _task("ticket-routing", ["a", "b"], references=["A", None])

    with pytest.raises(ValueError):
        Harness.for_task(task, CallableCandidate(str.upper), scorers=[_exact_match])


def test_result_round_trips_through_json_with_records_and_scores_intact():
    from velari_harness.core.execution.harness import Harness
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.response.result import Result

    task = _task("ticket-routing", ["card charged twice", "login fails"], references=["CARD CHARGED TWICE", "x"])

    result = Harness.for_task(task, CallableCandidate(str.upper), scorers=[_exact_match]).run()

    assert Result.model_validate_json(result.model_dump_json()) == result
