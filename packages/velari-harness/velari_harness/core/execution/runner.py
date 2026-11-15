import  logging
import  time
from    typing import Iterable, List, Optional
# package modules
from    ..protocols.candidate      import Candidate
from    ..schemas.dataset          import Example
from    ..schemas.response.record  import Failure, Outcome, Output, TrialRecord

logger = logging.getLogger(__name__)


class Runner(object):
    """Run a candidate over a sequence of examples, one trial per example, and record how each trial went.

    A trial that raises becomes an `ERROR` record and the run carries on; the first failure in a trial is the one reported.
    A candidate may return an `Output` to report usage and files; a plain return is wrapped in one.

    Examples:
        A plain function as the candidate, run over two incoming support messages:

        >>> def priority(text: str) -> str:
        ...     return "high" if "outage" in text.lower() else "normal"
        >>> records = Runner().run(CallableCandidate(priority), [Example("Checkout outage for all EU customers"), Example("Question about an invoice date")])
        >>> [(record.outcome.status.value, record.outcome.output.value) for record in records]
        [('success', 'high'), ('success', 'normal')]

        Applied (illustrative): an agent candidate over support questions. A question that makes the
        agent raise comes back as an `ERROR` record and the others still run:

        >>> records = Runner().run(BillingAgentCandidate(), [Example("What's the balance on ACC-10293?"), Example("Summarize ticket ORG-4471.")])
        >>> [record.outcome.failure for record in records]
        [None, None]
    """
    def run(self, candidate: Candidate, examples: Iterable[Example]) -> List[TrialRecord]:
        """Run `candidate` once per example, in order, and return one record per example.

        Args:
            candidate (Candidate): What to run; its lifecycle runs once per example.
            examples (Iterable[Example]): What to run on, one trial each; only each example's `input` is passed to `candidate.run()`.

        Returns:
            List[TrialRecord]: One record per example, in order; empty when there are no examples.
        """
        return [self._trial(candidate, index, example) for index, example in enumerate(examples)]

    def _trial(self, candidate: Candidate, index: int, example: Example) -> TrialRecord:
        start = time.perf_counter()
        output: Optional[Output] = None
        failure: Optional[Failure] = None
        try:
            candidate.setup()
            returned = candidate.run(example.input)
            output = returned if isinstance(returned, Output) else Output(value=returned)
        except Exception as exc:
            failure = Failure.of(exc)
        finally:
            # teardown always runs, also for KeyboardInterrupt and SystemExit, which then propagate
            try:
                candidate.teardown()
            except Exception as exc:
                failure = failure or Failure.of(exc)
        if failure is not None:
            logger.error(f"trial {index} failed: {failure.error}")
        outcome = Outcome(output=output, failure=failure, duration_sec=time.perf_counter() - start)
        return TrialRecord(index=index, example=example, outcome=outcome)
