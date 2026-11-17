import  pandas as pd
from    typing      import Dict, List, Optional
from    pydantic    import BaseModel, Field, computed_field
# velari-core
from    velari_core.core.statistics.stats import calc_column_stats, n_na
# package modules
from    .record     import TrialRecord
from    ..types     import Status


class ScoreSummary(BaseModel):
    """One scorer's scores over a run: their mean and its standard error, and how many were valid or not.

    Args:
        mean (Optional[float]): Mean value over the scores that have one; `None` when there are none.
        stderr (Optional[float]): Standard error of the mean, the sample standard deviation over `sqrt(n)`; `None` below two values.
        n (int): How many scores have a value.
        invalid (int): How many have none because the scorer could not judge; a failed trial is never scored, so it is in neither count.

    Examples:
        A relevance judge that scored two answers and could not judge a third:

        >>> answers = Dataset("support-answers", [Example("a1", "ref1"), Example("a2", "ref2"), Example("a3", "ref3")])
        >>> grades = {"a1": 0.5, "a2": 1.0, "a3": None}
        >>> def relevance(output, reference):
        ...     return Score(value=grades[output])
        >>> result = Harness.for_task(Task("answer-quality", answers), CallableCandidate(str), scorers=[relevance]).run()
        >>> result.summary["relevance"]
        ScoreSummary(mean=0.75, stderr=0.25, n=2, invalid=1)
    """
    mean:     Optional[float]
    stderr:   Optional[float]
    n:        int
    invalid:  int


class Result(BaseModel):
    """What one harness run produced: the harness name, one record per task input in order, and how long it took.

    `summary` has each scorer's mean, standard error and counts.
    """
    name:          str
    records:       List[TrialRecord]
    duration_sec:  float              = Field(ge=0)

    @computed_field
    @property
    def trials(self) -> Dict[Status, int]:
        """How many trials ended in each status; a run is complete even if every trial failed, so check `ERROR`."""
        return {status: sum(record.outcome.status == status for record in self.records) for status in Status}

    @computed_field
    @property
    def usage(self) -> Dict[str, float]:
        """Total of each usage key over the trials that reported it, such as tokens for a whole run."""
        totals: Dict[str, float] = {}
        for record in self.records:
            if (output := record.outcome.output) is not None:
                for name, amount in output.usage.items():
                    totals[name] = totals.get(name, 0.0) + amount
        return totals

    @computed_field
    @property
    def summary(self) -> Dict[str, ScoreSummary]:
        """Each scorer's mean, standard error and counts over the trials it scored, sorted by scorer name."""
        df_stats = (
            pd.DataFrame(
                [(name, score.value) for record in self.records for name, score in record.scores.items()],
                columns=["scorer", "value"],
            )
            .pipe(calc_column_stats, "value", by="scorer", stats=(n_na,))
        )
        return {
            str(name): ScoreSummary(
                mean=None if pd.isna(stats["mean"]) else stats["mean"],
                stderr=None if pd.isna(stats["sem"]) else stats["sem"],
                n=int(stats["count"]),
                invalid=int(stats["n_na"]),
            )
            for name, stats in df_stats.to_dict("index").items()
        }
