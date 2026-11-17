"""Tests for velari_harness.core.schemas.response.result."""

import pytest
from pydantic import ValidationError


def _record(index, output=None, failure=None, scores=None, usage=None):
    from velari_harness.core.schemas.dataset import Example
    from velari_harness.core.schemas.response.record import Outcome, Output, Score, TrialRecord

    outcome = Outcome(output=Output(value=output, usage=usage or {}) if output is not None else None, failure=failure, duration_sec=0.0)
    return TrialRecord(index=index, example=Example("x"), outcome=outcome, seed=index,
                       scores={name: Score(value=value) for name, value in (scores or {}).items()})


def test_result_negative_duration_raises_validationerror():
    from velari_harness.core.schemas.response.result import Result

    with pytest.raises(ValidationError):
        Result(name="x", records=[], duration_sec=-1.0)


def test_result_trials_count_each_status_and_are_in_the_json():
    import json

    from velari_harness.core.schemas.response.record import Failure
    from velari_harness.core.schemas.response.result import Result
    from velari_harness.core.schemas.types import ReasonCode

    failure = Failure(error="ValueError: bad", reason_code=ReasonCode.INVALID_INPUT)

    result = Result(name="t", records=[_record(0, output="A"), _record(1, failure=failure)], duration_sec=0.0)

    assert result.trials == {"success": 1, "error": 1}
    assert json.loads(result.model_dump_json())["trials"]["error"] == 1


def test_result_usage_totals_each_key_over_the_trials_that_reported_it_and_is_in_the_json():
    import json

    from velari_harness.core.schemas.response.record import Failure
    from velari_harness.core.schemas.response.result import Result
    from velari_harness.core.schemas.types import ReasonCode

    failure = Failure(error="ValueError: bad", reason_code=ReasonCode.INVALID_INPUT)
    records = [_record(0, output="A", usage={"input_tokens": 10, "output_tokens": 2}),
               _record(1, output="B", usage={"input_tokens": 5}), _record(2, failure=failure)]

    result = Result(name="t", records=records, duration_sec=0.0)

    assert result.usage == {"input_tokens": 15.0, "output_tokens": 2.0}
    assert json.loads(result.model_dump_json())["usage"] == {"input_tokens": 15.0, "output_tokens": 2.0}


def test_result_without_records_counts_zero():
    from velari_harness.core.schemas.response.result import Result

    assert Result(name="t", records=[], duration_sec=0.0).trials == {"success": 0, "error": 0}


def test_result_summary_has_mean_and_counts_per_scorer_and_is_in_the_json():
    import json

    from velari_harness.core.schemas.response.result import Result

    records = [_record(0, scores={"exact_match": 1.0}), _record(1, scores={"exact_match": 0.0}), _record(2, scores={"exact_match": None})]

    result = Result(name="t", records=records, duration_sec=0.0)

    summary = result.summary["exact_match"]
    assert (summary.mean, summary.n, summary.invalid) == (0.5, 2, 1)
    assert json.loads(result.model_dump_json())["summary"]["exact_match"]["mean"] == 0.5
    assert Result.model_validate_json(result.model_dump_json()) == result


def test_result_summary_stderr_is_the_standard_error_of_the_mean():
    from velari_harness.core.schemas.response.result import Result

    records = [_record(0, scores={"exact_match": 0.0}), _record(1, scores={"exact_match": 1.0})]

    assert Result(name="t", records=records, duration_sec=0.0).summary["exact_match"].stderr == pytest.approx(0.5)


@pytest.mark.parametrize("values, expected", [
    ([0.8],        (0.8,  None, 1, 0)),
    ([0.8, None],  (0.8,  None, 1, 1)),
    ([None],       (None, None, 0, 1)),
])
def test_result_summary_counts_without_two_valid_scores(values, expected):
    from velari_harness.core.schemas.response.result import Result

    records = [_record(index, scores={"judge": value}) for index, value in enumerate(values)]

    summary = Result(name="t", records=records, duration_sec=0.0).summary["judge"]

    assert (summary.mean, summary.stderr, summary.n, summary.invalid) == expected
