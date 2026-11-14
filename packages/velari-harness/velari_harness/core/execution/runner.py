import  logging
import  time
from    typing import Any, Iterable, List, Optional
# package modules
from    ..protocols.candidate      import Candidate
from    ..schemas.request.case     import Case
from    ..schemas.response.record  import Failure, Outcome, TrialRecord

logger = logging.getLogger(__name__)


class Runner(object):
    """Run a candidate over a sequence of cases, one trial per case, and record how each trial went.

    A trial that raises becomes an `ERROR` record and the run carries on; the first failure in a trial is the one reported.

    Examples:
        A plain function as the candidate, run over two incoming support messages:

        >>> def priority(text: str) -> str:
        ...     return "high" if "outage" in text.lower() else "normal"
        >>> records = Runner().run(CallableCandidate(priority), [Case("Checkout outage for all EU customers"), Case("Question about an invoice date")])
        >>> [(record.outcome.status.value, record.outcome.output) for record in records]
        [('success', 'high'), ('success', 'normal')]

        Applied (illustrative): an agent candidate over support questions. A question that makes the
        agent raise comes back as an `ERROR` record and the others still run:

        >>> records = Runner().run(BillingAgentCandidate(), [Case("What's the balance on ACC-10293?"), Case("Summarize ticket ORG-4471.")])
        >>> [record.outcome.failure for record in records]
        [None, None]
    """
    def run(self, candidate: Candidate, cases: Iterable[Case]) -> List[TrialRecord]:
        """Run `candidate` once per case, in order, and return one record per case.

        Args:
            candidate (Candidate): What to run; its lifecycle runs once per case.
            cases (Iterable[Case]): What to run on, one trial each; only each case's `input` is passed to `candidate.run()`.

        Returns:
            List[TrialRecord]: One record per case, in order; empty when there are no cases.
        """
        return [self._trial(candidate, index, case) for index, case in enumerate(cases)]

    def _trial(self, candidate: Candidate, index: int, case: Case) -> TrialRecord:
        start = time.perf_counter()
        output: Any = None
        failure: Optional[Failure] = None
        try:
            candidate.setup()
            output = candidate.run(case.input)
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
        return TrialRecord(index=index, case=case, outcome=outcome)
