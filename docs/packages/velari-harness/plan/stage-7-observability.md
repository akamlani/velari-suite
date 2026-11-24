# Stage 7: Observability integrations (`velari_harness/observability/`)

Part of [plan.md](plan.md). Requires stage 2 up to increment 17 (`Event` and `TrialContext`, implemented; see the [build order](stage-2-base-framework.md#build-order)). It gets more useful after stage 3 (candidates that emit model and tool events) and the [simulation layer](simulation.md) (world steps as events). Status: planned, nothing built. Goal: a minimal adjusted `Event` that works with no observability tool at all, and exporters that send the same records to Arize Phoenix and MLflow Tracing.

## Why the Event needs a small adjustment

An `Event` today is `(name, data, elapsed_sec)`: one point in time, which is what OpenTelemetry calls a **span event**. Observability tools are built around **spans**: an operation with a start, an end, a parent and a status. A point cannot say that a tool call took 0.55 s, or that a model call happened inside a step.

| Observability concept | Our equivalent today | Gap |
|---|---|---|
| Trace: one request, end to end | `TrialRecord`: one trial | HUD calls the same thing a "trace" |
| Span: name, start, end, parent, attributes, status | Nothing | No duration, parent or status |
| Span event: a timestamped note on a span | `Event` | The closest match |
| Attributes | `Event.data` | Free-form; observability tools accept only primitives and arrays |
| Trace and span ids, propagation | Nothing | Not needed inside one in-process trial |
| Exporter and backend (Phoenix, MLflow) | Nothing in the harness | `velari-ai` has connectors for both |

The adjustment is small and additive, so the harness stays useful on its own, and an exporter has enough to build spans from.

## The minimal adjusted Event (no integration needed)

```python
# pseudo-code, not implemented
class EventKind(StrEnum):
    """A small vocabulary that maps one to one onto the span kinds of both tools."""
    NOTE      = auto()    # a point-in-time fact (the default)        -> a span event
    STEP      = auto()    # a unit of work inside the candidate       -> CHAIN
    MODEL     = auto()    # a model call                              -> LLM
    TOOL      = auto()    # a tool or action call                     -> TOOL
    RETRIEVAL = auto()    # a lookup                                  -> RETRIEVER
    CHECK     = auto()    # a rule, guardrail or verification         -> GUARDRAIL

class Event(BaseModel):
    name:          str
    kind:          EventKind        = EventKind.NOTE
    data:          Dict[str, Any]   = Field(default_factory=dict)   # flat and JSON friendly; no sensitive content
    elapsed_sec:   float            = Field(ge=0)                   # start offset from the trial's start
    duration_sec:  Optional[float]  = None                          # None means a point event
    error:         Optional[str]    = None                          # set when the work raised
    parent:        Optional[int]    = None                          # index of the enclosing event in this trial's events

class TrialRecord(BaseModel):
    ...
    started_at:    Optional[float]  = None                          # wall-clock start (epoch seconds), so an exporter can place the trial in time
```

| What changes | What stays |
|---|---|
| Four optional fields on `Event` (`kind`, `duration_sec`, `error`, `parent`) and one on `TrialRecord` (`started_at`) | `Event(name, data, elapsed_sec)` keeps working as it does today; every existing record still loads |
| A new `ctx.span(...)` for work that takes time | `ctx.emit(name, data)` for a point in time, unchanged |
| Nothing outside the trial: no ids, no propagation, no exporter | Events are still plain data on the record |

```python
# pseudo-code, not implemented
class TrialContext:
    def emit(self, name, data=None, kind=EventKind.NOTE):
        self._events.append(Event(name=name, kind=kind, data=data or {}, elapsed_sec=self._now(), parent=self._current_parent()))

    @contextmanager
    def span(self, name, kind=EventKind.STEP, **data):
        index = len(self._events)
        start = self._now()
        parent = self._current_parent()
        self._events.append(None)                  # reserve the slot, so a child's parent index is stable
        self._open.append(index)
        error = None
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}" # record it, then let it propagate
            raise
        finally:
            self._open.pop()
            self._events[index] = Event(name=name, kind=kind, data=data, elapsed_sec=start,
                                        duration_sec=self._now() - start, error=error, parent=parent)
```

Usage in a candidate:

```python
# pseudo-code, not implemented
class BillingAgent(Candidate):
    def run(self, question, ctx):
        with ctx.span("answer_ticket", EventKind.STEP):
            ctx.emit("identity_verified", {"method": "otp"}, kind=EventKind.CHECK)
            with ctx.span("lookup_account_balance", EventKind.TOOL, account="ACC-10293"):
                balance = self.lookup("ACC-10293")
            with ctx.span("draft_reply", EventKind.MODEL, model="gpt-4o-mini"):
                reply = self.draft(question, balance)
        return reply
```

## Examples of events, and how each is used

| Event name | Kind | Typical `data` | Emitted by | Used for |
|---|---|---|---|---|
| `rule_matched` | NOTE | `{"rule": "outage"}` | Classifiers, routers | Explain a decision; count which rule fires |
| `skill_triggered` | NOTE | `{"skill": "write-invoice", "score": 7.2}` | Skill pickers (built in the skills example) | Tune `min_score`; check the expected skill fired |
| `tool_called` | TOOL | `{"tool": "lookup_account_balance", "ok": true}` | Agents | Tool-call accuracy, latency, cost |
| `retrieval` | RETRIEVAL | `{"query": "refund policy", "k": 5, "hits": 3}` | Retrievers | Debug a wrong answer: was the right document found? |
| `model_call` | MODEL | `{"model": "gpt-4o-mini", "input_tokens": 812, "output_tokens": 96}` | LLM wrappers | Per-step cost beyond the trial total in `Output.usage` |
| `retry` | NOTE | `{"attempt": 2, "reason": "timeout"}` | Anything with retries | Find quiet reliability problems |
| `identity_verified` | CHECK | `{"method": "otp"}` | Agents in regulated work | Gate: verified before the first change |
| `approval_requested` | CHECK | `{"action": "isolate_host"}` | Incident-response agents | Gate: destructive actions needed approval |
| `guardrail_triggered` | CHECK | `{"rule": "no_pii_in_reply"}` | Guardrail layers | Audit how often a rule fires |
| `fallback_used` | NOTE | `{"from": "primary", "to": "backup"}` | Anything with a fallback | Detect silent degradation |
| `world_step` | STEP | `{"action": "isolate_host", "done": false}` | The [simulation layer](simulation.md) | Replay a trajectory; trajectory scorers |
| `terminated` | NOTE | `{"reason": "max_steps"}` | The simulation layer | Tell finished from truncated episodes |
| `fit_started`, `metric_computed` | STEP, NOTE | `{"rows": 12000}`, `{"name": "recall", "value": 0.81}` | Model drivers | Audit a fit; keep partial metrics |

Conventions:

- **Names:** snake_case, stable, because analyses group by name.
- **`data`:** small, flat and JSON friendly; nested values are turned into JSON text on export. **No sensitive content** (patient text, card numbers, secrets): events persist on the record and may be saved or exported, so emit an id or a category.
- **Emit decisions and boundaries**, such as a tool call or a retry, not every line of code, and do not repeat what the output already carries.
- **Scorers do not see events today.** A scorer receives `(output, reference)`. Events are read from `TrialRecord.events`, or reach a scorer inside an output, as the simulation layer does with its trajectory.

## The no-integration default

With no observability tool configured, the events already sit on every `TrialRecord`. Stage 7 adds a small `observability/` package that needs no third-party packages:

```python
# pseudo-code, not implemented
def events_frame(result: Result) -> pd.DataFrame:
    """One row per event: trial, index, name, kind, elapsed_sec, duration_sec, error, parent, data."""
    return pd.DataFrame([
        {"trial": record.index, "index": index, **event.model_dump()}
        for record in result.records for index, event in enumerate(record.events)
    ])

class JsonlExporter(TraceExporter):
    def __init__(self, path): self.path = path
    def export(self, result):                      # one JSON line per trial, events included
        with open(self.path, "w") as handle:
            for record in result.records:
                handle.write(record.model_dump_json() + "\n")

class ConsoleExporter(TraceExporter):
    def export(self, result):                      # an indented tree built from `parent`
        ...
```

An illustrative console tree for one trial:

```
trial 0  seed 4121104260  1.42 s  success
├─ step  answer_ticket              1.40 s
│  ├─ check  identity_verified      point
│  ├─ tool   lookup_account_balance 0.55 s
│  └─ model  draft_reply            0.82 s
└─ note  answered                   point
```

## The integration seam

`velari-ai` has `TracerLike` and `SpanLike` (`ai/tracing/trace.py`), but they are too thin to export from: `SpanLike` only has `set_attribute`, with no start time, no events and no status. So the harness defines its own small protocol and reuses `TracingConnector` only for connection and configuration:

```python
# pseudo-code, not implemented
class TraceExporter(Protocol):
    def export(self, result: Result) -> None: ...      # post-hoc: after the run, from the records
```

A **post-hoc** exporter keeps the run offline and deterministic, and changes nothing in the Runner. Its risks are placing spans in time and, for MLflow, the open points below. If post-hoc export is not enough, a later **live `Observer`** hook in the Runner (`trial_started`, `event`, `trial_finished`) covers it.

### Phoenix (OpenTelemetry and OpenInference)

```python
# pseudo-code, not implemented
KIND = {EventKind.STEP: "CHAIN", EventKind.MODEL: "LLM", EventKind.TOOL: "TOOL",
        EventKind.RETRIEVAL: "RETRIEVER", EventKind.CHECK: "GUARDRAIL"}

class PhoenixExporter(TraceExporter):
    def __init__(self, tracer): self.tracer = tracer            # from the existing Phoenix connector (`phoenix.otel.register`)

    def export(self, result):
        for record in result.records:
            start = to_ns(record.started_at)
            root = self.tracer.start_span("trial", start_time=start, attributes=trial_attributes(record))
            spans = {}                                           # event index -> span, so children find their parent
            for index, event in enumerate(record.events):
                at = start + to_ns(event.elapsed_sec)
                parent = spans.get(event.parent, root)
                if event.duration_sec is None:
                    parent.add_event(event.name, attributes=flatten(event.data), timestamp=at)
                else:
                    span = self.tracer.start_span(event.name, context=set_span_in_context(parent), start_time=at,
                                                  attributes={"openinference.span.kind": KIND[event.kind], **flatten(event.data)})
                    if event.error:
                        span.set_status(Status(StatusCode.ERROR, event.error))
                    spans[index] = span
                    span.end(end_time=at + to_ns(event.duration_sec))
            root.end(end_time=start + to_ns(record.outcome.duration_sec))
```

### MLflow Tracing

```python
# pseudo-code, not implemented
SPAN_TYPE = {EventKind.STEP: "CHAIN", EventKind.MODEL: "LLM", EventKind.TOOL: "TOOL",
             EventKind.RETRIEVAL: "RETRIEVER", EventKind.CHECK: "GUARDRAIL"}

class MlflowExporter(TraceExporter):
    def export(self, result):
        for record in result.records:
            with mlflow.start_span(name="trial", span_type="CHAIN") as root:
                root.set_attributes({"velari.seed": record.seed, "velari.example.uid": record.example.uid})
                for event in record.events:
                    # UNVERIFIED: explicit start and end times and manual span events are not described on MLflow's
                    # manual-tracing page. The fallback keeps the original timing as attributes instead of on the timeline.
                    with mlflow.start_span(name=event.name, span_type=SPAN_TYPE.get(event.kind, "UNKNOWN")) as span:
                        span.set_attributes({"velari.elapsed_sec": event.elapsed_sec, "velari.duration_sec": event.duration_sec,
                                             **flatten(event.data)})
```

### How the pieces map

| Ours | OpenTelemetry | OpenInference (Phoenix) | MLflow | Status |
|---|---|---|---|---|
| `TrialRecord` | A trace with a root span | Root span, kind CHAIN | Root span, `CHAIN` | Concepts verified |
| Point `Event` | Span event (`add_event`) | Span event | No manual way described; a zero-length child span or attributes | OTel verified; MLflow open |
| Timed `Event` | Child span | Child span with a span kind | Child span with a `SpanType` | Verified |
| `kind` | n/a | `CHAIN`, `LLM`, `TOOL`, `RETRIEVER`, `GUARDRAIL` | `CHAIN`, `LLM`, `TOOL`, `RETRIEVER`, `GUARDRAIL` | Names verified in the installed packages |
| `data` | Attributes: primitives and arrays only | Attributes | Attributes | Nested values become JSON text |
| `error` | Status ERROR and a recorded exception | Same | Exceptions are recorded as span events | OTel verified |
| `seed`, `example.uid`, scores | Trial attributes (`velari.seed`, `velari.score.<name>`) | Same | Same | Our names |
| `Output.usage` | `gen_ai.*` attributes (still marked "Development") | `llm.token_count.*` | Token attributes | Names to confirm at build |
| Example input and output | n/a | `input.value`, `output.value` (already used in `integrations/arize/trace.py`) | `set_inputs`, `set_outputs` | Verified for both |

## Package layout and dependencies

```
velari_harness/observability/          # L3: reporting; imports only the core
├── exporter.py     # TraceExporter protocol, events_frame
├── jsonl.py        # JsonlExporter
├── console.py      # ConsoleExporter
├── phoenix.py      # PhoenixExporter (needs the velari-ai `evals` extra)
└── mlflow.py       # MlflowExporter (needs the velari-ai `mlflow` extra)
```

- **One direction:** `observability/` imports `core/`; the core never imports it.
- **Optional packages** come through the `velari-ai` extras (`evals` for Phoenix and OpenTelemetry, `mlflow`). Each integration is its own module, imported only when used; there are no `TYPE_CHECKING` guards.
- **Connection and config** reuse `TracingConnector` and the existing connectors in `velari_ai/integrations/{arize,mlflow}/connection.py`.

## Build order

| Increment | Adds | Status |
|---|---|---|
| 7a | The `Event` adjustment: `EventKind`, `duration_sec`, `error`, `parent`, `TrialRecord.started_at`, and `ctx.span`; tests; docs | Planned |
| 7b | `TraceExporter`, `events_frame`, `JsonlExporter`, `ConsoleExporter` | Planned |
| 7c | `PhoenixExporter` over the existing connector, tested with a fake tracer | Planned |
| 7d | `MlflowExporter`; first settles the open MLflow questions | Planned |
| 7e | An example under `examples/ai/harness/observability/`: offline by default, Phoenix or MLflow by config; README | Planned |
| 7f | Later: a generic OTLP exporter, Langfuse and Weave, the live `Observer`, alignment with the OpenTelemetry GenAI conventions | Not planned in detail |

## Not in this stage

Distributed tracing across processes, sampling, a hosted dashboard of our own, per-token streaming events, and changing the scorer contract.

## Open questions

- **MLflow timestamps and span events:** can a span be created with explicit start and end times, and can an event be added manually? Its manual-tracing page does not say. If not, the fallback is attributes, or the live `Observer`.
- **Wall-clock start:** post-hoc export needs an absolute start time per trial, which a record does not carry today. This stage proposes `TrialRecord.started_at`.
- **Post-hoc or live:** post-hoc is simpler and deterministic; live gives true timing for slow trials and streams to a backend while a run is going.
- **Attribute names for tokens and scores:** `llm.token_count.*` and the OpenTelemetry `gen_ai.*` names are to be confirmed against the specs at build time; the `gen_ai.*` conventions are still marked "Development".
- **How Phoenix attaches evaluation scores** (as attributes, or as annotations on a span) is not yet checked.
- **Sensitive data:** whether an exporter should redact or drop `data` keys by name.

## Verified and unverified

| Verified (read in this work) | Not verified |
|---|---|
| OpenTelemetry trace, span and span-event concepts and the attribute value types; `Span.add_event`, `set_status` and `record_exception` in the installed package | MLflow explicit timestamps and manual span events |
| The OpenInference span kinds and the MLflow `SpanType` values, from the installed packages | Token-count attribute names in OpenInference and MLflow |
| MLflow manual tracing: `start_span`, `span_type`, `set_attributes`, automatic nesting, exceptions recorded as events | How Phoenix records evaluation scores |
| `velari-ai`'s `TracingConnector`, `TracerLike`, `SpanLike`, and the Phoenix and MLflow connectors | The full `gen_ai.*` attribute list |

## Sources

- OpenTelemetry: [traces concepts](https://opentelemetry.io/docs/concepts/signals/traces/), [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- OpenInference: [semantic conventions](https://openinf.dev/spec/attributes/semantic-conventions.html) (span kinds checked in the installed package)
- MLflow: [tracing](https://mlflow.org/docs/latest/genai/tracing/), [manual tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/)
- In this repo: `velari_ai/ai/tracing/trace.py`, `velari_ai/integrations/arize/{connection,trace}.py`, `velari_ai/integrations/mlflow/connection.py`
