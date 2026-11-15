"""Tests for velari_harness.core.schemas.dataset."""

from dataclasses import FrozenInstanceError

import pytest


def _dataset(*examples, name="tickets"):
    from velari_harness.core.schemas.dataset import Dataset

    return Dataset(name, list(examples))


class TestDataset:
    def test_keeps_examples_in_order(self):
        from velari_harness.core.schemas.dataset import Example

        first, second = Example("card charged twice", "billing", uid="a"), Example("login fails", "general", uid="b")

        assert _dataset(first, second).examples == (first, second)

    def test_stores_examples_as_tuple_so_later_changes_to_the_source_do_not_leak_in(self):
        from velari_harness.core.schemas.dataset import Dataset, Example

        source = [Example("card charged twice", uid="a")]
        dataset = Dataset("tickets", source)
        source.append(Example("login fails", uid="b"))

        assert len(dataset) == 1

    def test_given_id_is_kept(self):
        from velari_harness.core.schemas.dataset import Example

        assert [e.uid for e in _dataset(Example("card charged twice", uid="ticket-1"))] == ["ticket-1"]

    def test_without_id_gets_same_content_hash_in_any_dataset(self):
        from velari_harness.core.schemas.dataset import Example

        one = _dataset(Example("card charged twice", "billing"), Example("login fails", "general"))
        other = _dataset(Example("login fails", "general"), name="other")

        assert one.examples[1].uid == other.examples[0].uid

    @pytest.mark.parametrize("examples, name", [
        ([], "tickets"),
        (["card charged twice"], ""),
        (["card charged twice", "card charged twice"], "tickets"),
    ])
    def test_invalid_raises_valueerror(self, examples, name):
        from velari_harness.core.schemas.dataset import Dataset, Example

        with pytest.raises(ValueError):
            Dataset(name, [Example(text) for text in examples])

    @pytest.mark.parametrize("changed", [
        ["card charged twice", "login fails 500", "refund"],
        ["card charged twice", "login fails", "refund", "extra"],
        ["login fails", "card charged twice", "refund"],
    ])
    def test_fingerprint_changes_when_dataset_changes(self, changed):
        from velari_harness.core.schemas.dataset import Example

        original = _dataset(*[Example(text) for text in ["card charged twice", "login fails", "refund"]])

        assert _dataset(*[Example(text) for text in changed]).fingerprint != original.fingerprint

    def test_fingerprint_changes_when_a_reference_changes(self):
        from velari_harness.core.schemas.dataset import Example

        assert _dataset(Example("login fails", "general", uid="a")).fingerprint != _dataset(Example("login fails", "billing", uid="a")).fingerprint

    def test_fingerprint_is_stable_for_identical_data(self):
        from velari_harness.core.schemas.dataset import Example

        assert _dataset(Example("login fails", "general")).fingerprint == _dataset(Example("login fails", "general"), name="copy").fingerprint


class TestTask:
    def test_empty_name_raises_valueerror(self):
        from velari_harness.core.schemas.dataset import Example, Task

        with pytest.raises(ValueError):
            Task("", _dataset(Example("card charged twice")))

    def test_is_immutable(self):
        from velari_harness.core.schemas.dataset import Example, Task

        task = Task("ticket-routing", _dataset(Example("card charged twice")))

        with pytest.raises(FrozenInstanceError):
            setattr(task, "version", "2")

    def test_runs_against_two_candidates_without_knowing_either(self):
        from velari_harness.core.execution.runner import Runner
        from velari_harness.core.protocols.candidate import CallableCandidate
        from velari_harness.core.schemas.dataset import Example, Task
        from velari_harness.core.schemas.response.record import Output

        task = Task("ticket-routing", _dataset(Example("Card charged twice for invoice INV-2291"), Example("Login fails with a 500 error")))
        baseline = CallableCandidate(lambda text: "general")
        variant = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")

        baseline_outputs = [r.outcome.output for r in Runner().run(baseline, task.dataset)]
        variant_outputs = [r.outcome.output for r in Runner().run(variant, task.dataset)]

        assert baseline_outputs == [Output(value="general"), Output(value="general")]
        assert variant_outputs == [Output(value="billing"), Output(value="general")]
