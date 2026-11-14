import  logging
import  time
from    typing import Any, Iterable, List, Optional
# package modules
from    ..protocols.candidate      import Candidate
from    ..schemas.response.record  import Failure, Outcome, TrialRecord

logger = logging.getLogger(__name__)


class Runner(object):
    """Run a candidate over a sequence of inputs, one trial per input, and record how each trial went.

    A trial that raises becomes an `ERROR` record and the run carries on; the first failure in a trial is the one reported.

    Examples:
        A plain function as the candidate, run over two incoming support messages:

        >>> def priority(text: str) -> str:
        ...     return "high" if "outage" in text.lower() else "normal"
        >>> records = Runner().run(CallableCandidate(priority), ["Checkout outage for all EU customers", "Question about an invoice date"])
        >>> [(record.outcome.status.value, record.outcome.output) for record in records]
        [('success', 'high'), ('success', 'normal')]

        Applied (illustrative): an agent candidate over support questions. A question that makes the
        agent raise comes back as an `ERROR` record and the others still run:

        >>> records = Runner().run(BillingAgentCandidate(), ["What's the balance on ACC-10293?", "Summarize ticket ORG-4471."])
        >>> [record.outcome.failure for record in records]
        [None, None]
    """
    def run(self, candidate: Candidate, inputs: Iterable[Any]) -> List[TrialRecord]:
        """Run `candidate` once per input, in order, and return one record per input.

        Args:
            candidate (Candidate): What to run; its lifecycle runs once per input.
            inputs (Iterable[Any]): Values passed to `candidate.run()`, one trial each.

        Returns:
            List[TrialRecord]: One record per input, in input order; empty when there are no inputs.
        """
        return [self._trial(candidate, index, value) for index, value in enumerate(inputs)]

    def _trial(self, candidate: Candidate, index: int, value: Any) -> TrialRecord:
        start = time.perf_counter()
        output: Any = None
        failure: Optional[Failure] = None
        try:
            candidate.setup()
            output = candidate.run(value)
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
        return TrialRecord(index=index, input=value, outcome=outcome)
