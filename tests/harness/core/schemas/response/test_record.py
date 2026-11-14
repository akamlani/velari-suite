"""Tests for velari_harness.core.schemas.response.record."""

import pytest
from pydantic import ValidationError


def test_trial_record_json_roundtrip_returns_equal_model():
    from velari_harness.core.execution.runner import Runner
    from velari_harness.core.protocols.candidate import CallableCandidate
    from velari_harness.core.schemas.request.case import Case
    from velari_harness.core.schemas.response.record import TrialRecord

    record = Runner().run(CallableCandidate(str.upper), [Case("card charged twice", "CARD CHARGED TWICE")])[0]

    assert TrialRecord.model_validate_json(record.model_dump_json()) == record


def test_outcome_negative_duration_raises_validationerror():
    from velari_harness.core.schemas.response.record import Outcome

    with pytest.raises(ValidationError):
        Outcome(duration_sec=-1.0)


def test_outcome_status_is_derived_from_the_failure():
    from velari_harness.core.schemas.response.record import Failure, Outcome
    from velari_harness.core.schemas.types import ReasonCode, Status

    failure = Failure(error="ValueError: bad", reason_code=ReasonCode.INVALID_INPUT)

    assert (Outcome(duration_sec=0.0).status, Outcome(failure=failure, duration_sec=0.0).status) == (Status.SUCCESS, Status.ERROR)
