from    typing    import Dict, List, Optional
from    pydantic  import BaseModel, Field, computed_field
# package modules
from    .record   import TrialRecord
from    ..types   import Status


class Result(BaseModel):
    """What one harness run produced: the harness name, one record per task input in order, and how long it took."""
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
    def scores(self) -> Dict[str, Optional[float]]:
        """Mean value per scorer over the trials it scored; `None` when all of its scores were `None`."""
        values: Dict[str, List[float]] = {}
        for record in self.records:
            for name, score in record.scores.items():
                scored = values.setdefault(name, [])
                if score.value is not None:
                    scored.append(score.value)
        return {name: sum(scored) / len(scored) if scored else None for name, scored in sorted(values.items())}
