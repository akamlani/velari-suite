"""Tests for velari_harness.core.schemas.response.record."""

import pytest
from pydantic import ValidationError


class TestTrialRecord:
    def test_json_roundtrip_returns_equal_model(self):
        from velari_harness.core.execution.runner import Runner
        from velari_harness.core.protocols.candidate import CallableCandidate
        from velari_harness.core.schemas.dataset import Example
        from velari_harness.core.schemas.response.record import TrialRecord

        record = Runner().run(CallableCandidate(str.upper), [Example("card charged twice", "CARD CHARGED TWICE")])[0]

        assert TrialRecord.model_validate_json(record.model_dump_json()) == record

    def test_score_with_explanation_and_metadata_roundtrips_through_json(self):
        from velari_harness.core.schemas.dataset import Example
        from velari_harness.core.schemas.response.record import Outcome, Score, TrialRecord

        score = Score(value=0.5, label="partial", explanation="cites 1 of 2 sources", metadata={"judge": "relevance-v2"})
        record = TrialRecord(index=0, example=Example("a"), outcome=Outcome(duration_sec=0.0), seed=123, scores={"relevance": score})

        assert TrialRecord.model_validate_json(record.model_dump_json()) == record

    def test_events_roundtrip_through_json_in_order(self):
        from velari_harness.core.schemas.dataset import Example
        from velari_harness.core.schemas.response.record import Event, Outcome, TrialRecord

        events = [Event(name="skill_triggered", data={"skill": "write-invoice"}, elapsed_sec=0.01), Event(name="answered", elapsed_sec=0.02)]
        record = TrialRecord(index=0, example=Example("a"), outcome=Outcome(duration_sec=0.0), seed=123, events=events)

        assert TrialRecord.model_validate_json(record.model_dump_json()) == record


class TestOutcome:
    def test_negative_duration_raises_validationerror(self):
        from velari_harness.core.schemas.response.record import Outcome

        with pytest.raises(ValidationError):
            Outcome(duration_sec=-1.0)

    def test_status_is_derived_from_the_failure(self):
        from velari_harness.core.schemas.response.record import Failure, Outcome
        from velari_harness.core.schemas.types import ReasonCode, Status

        failure = Failure(error="ValueError: bad", reason_code=ReasonCode.INVALID_INPUT)

        assert (Outcome(duration_sec=0.0).status, Outcome(failure=failure, duration_sec=0.0).status) == (Status.SUCCESS, Status.ERROR)


class TestEvent:
    def test_negative_elapsed_raises_validationerror(self):
        from velari_harness.core.schemas.response.record import Event

        with pytest.raises(ValidationError):
            Event(name="rule_matched", elapsed_sec=-0.1)
