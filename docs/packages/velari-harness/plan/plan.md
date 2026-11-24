# Plan: `velari-harness` package

Status: stage 1 implemented and merged; stage 2 in progress (increments 1 to 17 implemented: `Harness` and its config, `Candidate`, `Runner`, `Task`, `Environment`, scorers, `Harness` as the container that chains run, score and package, `Example`, `Output`, a structured `Score`, a `Dataset`, a wired `Environment`, a `Result` summary, a skills layer and a `TrialContext`); stages 3-6 not yet detailed; stage 7 (observability integrations) is planned.

## What a harness is

"Harness" is used in three senses (run-and-judge, agent, pipeline). The definitions of these senses and of every term used in these plans are in the [glossary](../glossary.md#harness-vocabulary); the tools that illustrate each are listed under [Research references](#research-references).

The first version of stage 2 modelled only the third sense (an ordered list of stages over a shared context). That has no system under test, no repeated trials, no run record and no scoring, so it is not what makes something a harness. Pipeline steps stay useful, but only as an optional way to *prepare* examples inside a task.

**Definition used here.** See **Harness** in the [glossary](../glossary.md#harness-vocabulary).

### Research references

The sources behind the design, with the harness category each one informs and the idea taken from it.

| Reference | Category | What it is | Idea adopted |
|---|---|---|---|
| [Inspect AI](https://inspect.aisi.org.uk/tasks.html) | Evaluation | UK AISI evaluation framework | `Task(dataset, solver, scorer)`; the model is chosen at run time, not by the task; epochs and limits are task options; a run returns a re-readable log; components register by name |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Evaluation | EleutherAI benchmark runner | Declarative, named, groupable tasks; a pluggable model interface |
| [openai/evals](https://github.com/openai/evals) | Evaluation | OpenAI eval registry and runner | The candidate is a single call protocol (completion function), so chains and agents plug in without special cases |
| [HELM](https://github.com/stanford-crfm/helm) | Evaluation | Stanford holistic LM evaluation (maintenance mode since June 2026: design reference only) | A run is a cell of task x candidate x metric; request states keep everything a metric needs, so scores can be recomputed from stored records |
| [MLflow GenAI evaluation](https://mlflow.org) | Evaluation | Evaluation inside an experiment tracker | dataset -> predict function -> scorers, with traces recorded; a scorer error is stored as an error, not a score; paired comparison with confidence intervals (`compare_evaluations`, experimental) |
| [MLE-bench](https://openai.com/index/mle-bench/) | Data science (agentic) | Kaggle-based ML engineering benchmark for agents | A data science task is examples plus a grading script; the grader holds the hidden test labels and the prepare step splits public data; the agent returns a submission file; at least 3 seeds reported as mean +/- SEM, with incomplete seeds counted as failures; Docker setups and a "lite" split |
| [DSBench](https://github.com/LiqiangJing/DSBench) ([paper](https://arxiv.org/abs/2409.07703)) | Data science (agentic) | 540 data analysis and data modeling tasks from ModelOff and Kaggle | A case is a task with its data files and question; scoring depends on task type; results are reported as a gap to a human baseline |
| [AIDE ML](https://github.com/WecoAI/aideml) ([paper](https://arxiv.org/abs/2502.13138)) | Data science (agentic) | Tree-search agent that drafts, debugs and benchmarks code to optimise a metric | A data science agent is just a candidate that iterates internally; the harness records its trajectory and final metric. It is the reference agent in MLE-bench |
| [DSAgentBench](https://huggingface.co/papers/2608.10366) | Data science (agentic) | 275 end-to-end data science workflow tasks (the appendix table sums to 265) run in a real Ubuntu environment | Output files are collected after the run and checked by a deterministic evaluator; the continuous score is kept apart from the pass threshold; the time and step limits used are recorded |
| [Harbor](https://github.com/laude-institute/harbor) ([docs](https://docs.harborframework.com)) | Evaluation (agent), Runtime (L3) | Agent evaluation framework from the Terminal-Bench authors; official harness for Terminal-Bench 2.0 (license not confirmed: sources disagree) | Agent, environment, verifier and dataset are separate; each trial starts a sandbox, runs the agent, collects artifacts, verifies, and always stops the sandbox; a per-exception error policy says whether to retry, score and treat the reward as valid; `regrade` re-runs only the verifier on recorded artifacts; datasets are versioned and pinned; runs scale out to Daytona and Modal (the runtime layer) |
| [tau-bench / tau²](https://ukgovernmentbeis.github.io/inspect_evals/evals/tau2/index.html) ([paper](https://arxiv.org/abs/2406.12045)) | Evaluation (simulation) | Customer-service agent benchmark; an LLM plays the user, a database is the environment | Not adopted yet: an episode is a loop of user turn, agent turn and deterministic tool calls that ends on a stop token or an action cap; success is the final state matching one expected state, so the score does not depend on the simulated user; pass^k reports reliability over repeated trials |
| [Coval](https://docs.coval.ai/) | Evaluation (simulation, commercial) | Simulation and evaluation platform for voice and chat agents | Not adopted yet: the test case (what the user wants) is separate from the persona (who they are and how they sound), so one set of cases runs under many personas; no seed documented, repeats advised |
| [HUD](https://www.hud.ai/) | Environments (RL) | Platform for building environments to train and evaluate agents (coding, browser, computer use, research) | Not adopted yet: environments are versioned and reviewed like tasks, Harbor tasks import, and QA agents audit traces for reward hacking and false passes; whether its environments are simulated is not stated |
| [Raindrop](https://www.raindrop.ai/) | Agent runtime (observability, simulation) | Production monitoring for agents, with Simulate in early access | Not adopted yet: Simulate runs the agent in a sealed synthetic world of simulated services, data and state with no production calls; authoring and fidelity checks not described |
| [Simile](https://www.simile.com/) | Simulation (of people) | Simulates how customers or populations respond to a change | Noted only: a different meaning of simulation (people, not an agent's world); results are validated against real human responses and carry a confidence level |
| [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/) | Observability | The standard trace, span and span-event model | Not adopted yet: our `Event` is a span event; a timed event with a parent maps to a span; attribute values are primitives and arrays only, so nested `data` is flattened on export |
| [OpenInference](https://openinf.dev/spec/attributes/semantic-conventions.html) and [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Observability | OpenTelemetry-based conventions for LLM spans, and the Phoenix backend | Not adopted yet: the span kinds CHAIN, LLM, TOOL, RETRIEVER and GUARDRAIL map one to one onto `EventKind` |
| [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) | Observability | Tracing inside the experiment tracker, OpenTelemetry compatible | Not adopted yet: its `SpanType` values match `EventKind`; manual span events and explicit timestamps are not described and need checking |
| [Anthropic: effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | Agent runtime | Scaffold for multi-session agents | Recoverable state, clean end state, end-to-end verification; deferred to a later `agent-runtime` category |
| [Anthropic: Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) | Agent runtime | Filesystem skills for agents, loaded in three levels | A skill is a `SKILL.md` directory; name and description are always loaded and are the only trigger signal, the body loads when triggered and bundled files on demand; scripts run in a sandbox and skills from outside are untrusted |
| [Agent Skills specification and skills-ref](https://agentskills.io/specification) | Agent runtime | The open format (Anthropic originated it) and its reference library, for demonstration only | The frontmatter and name rules, the folder named after the skill, `compatibility` at most 500 characters; rules copied, not code |
| [Integrating skills into an agent](https://agentskills.io/integrate-skills) | Agent runtime | The official guide to discovering, disclosing and activating skills | The `<available_skills>` catalog (name, description, location), progressive disclosure, hide filtered skills; its lenient loading, precedence and trust advice are open choices for us |
| [Kedro](https://kedro.org), [ZenML](https://www.zenml.io), [Metaflow](https://metaflow.org), [Dagster](https://dagster.io) | Pipeline (inside a task) | Pipeline and orchestration tools | Steps, artifacts, lineage, used only to prepare examples inside a task |

Enterprise platforms (Vertex AI, Bedrock, Azure AI Foundry) were reachable only through secondary sources. They show the same shape (dataset, judge or metrics, run), so no specifics are taken from them.

The references were read through their docs and READMEs, not source code; several details (HELM per-run file names, lm-eval result files, MLflow `evaluate()` retry behavior, Harbor's `result.json` fields) are unverified.

### What the references changed in stage 2

A detailed read of the references above confirmed the core design (candidate separate from task, per-trial records, re-scoring, failures as data, derived seeds, preparation steps, runner-enforced limits) and changed seven things. Details are in [stage-2-base-framework.md](stage-2-base-framework.md).

| # | Change | Source |
|---|---|---|
| 1 | The candidate receives a `ExampleView`; `target` and `grader_files` reach only the scorer | MLE-bench, Harbor, DSBench |
| 2 | `Output` carries file artifacts, collected before scoring | MLE-bench, DSAgentBench, Harbor |
| 3 | `Score` has `valid`, `passed` and `error`; invalid is not zero | Harbor, MLflow, MLE-bench, DSAgentBench |
| 4 | An `ErrorPolicy` and `RetryPolicy` classify failures; `TIMEOUT` is a status | Harbor |
| 5 | `Candidate` has `setup/run/teardown`, a `TrialContext` and a `Sandbox` seam with a `LocalSandbox`; container sandboxes are L3 | Harbor, Inspect, openai/evals |
| 6 | `Runner.run(score=False)` and `rescore`: running and scoring are separate | lm-eval, Harbor, HELM |
| 7 | Content-hash case ids, `Dataset.fingerprint`, `Task.version`, and a spec recording the applied limits | openai/evals, MLflow, Harbor, DSAgentBench |
| 8 | Skills: the core loads and validates a `Skill` into a `SkillRegistry` (a table on `velari_data`'s `Registry`, built); the candidate triggers it and `TrialContext` events record it; scripts run only in a `Sandbox` | Anthropic Agent Skills |

Deferred to a possible stage 2b or later stages: run groups and result sets, resume, grouped statistics, paired comparison, concurrency, trajectory interchange, LLM-judge validation, container sandboxes. See "Deliberately not in stage 2" in the stage 2 doc.

## Layers

```
L3  Runtimes & reporting   local (default) | ray | ...  ·  table / json / report writers  ·  trace exporters (stage 7)     (orthogonal axis)
L2  Applied harnesses      rag-eval, tool-call-eval | clustering-experiment, supervised-fit  (presets of L1)
L1  Categories             evaluation  |  experiment (data science)  |  [later: agent-runtime]
L0  Core kernel            Candidate · Example/Dataset · Task · Scorer/Metric · Environment · Record · Result · Registry · Harness
```

Layer and category definitions are in the [glossary](../glossary.md#layers).

## Stages

Each stage is verified before the next starts. Stages 1, 2 and 7 have their own docs (stage 7 is planned, nothing built); the rest are outlined and get their own docs once 1 and 2 are settled.

| Stage | Category | Doc | Outcome |
|---|---|---|---|
| 1 | Package | [stage-1-scaffolding.md](stage-1-scaffolding.md) | Done: package created, wired into the workspace, shared version bootstrap from `velari-core`, no runtime extras yet |
| 2 | Core (L0) | [stage-2-base-framework.md](stage-2-base-framework.md) | Core kernel under `core/`, built in increments. Increments 1 to 17 (done): `Harness` and `HarnessConfig`, `Candidate`, a minimal `Runner`, a minimal `Task`, a minimal `Environment`, scorers as plain functions, `Harness` as the container that chains run, score and package, `Example`, `Output`, a structured `Score`, a `Dataset`, a wired `Environment`, a `Result` summary and a skills layer (`Skill`, `SkillRegistry`, built on a new `RecordRegistry` in `velari-data`) and `TrialContext`. Then: a [simulation layer](simulation.md), metrics, record store |
| 3 | Evaluation (L1) | not yet detailed | `evaluation/`: agent, gateway and callable candidate adapters, scorer wrappers over `ai/evals`, LLM judge, a skill adapter (exposes a catalog to an agent) and trigger scorers (did the expected skill fire); retire the stub `EvaluationHarness` |
| 4 | Data science (L1) | not yet detailed | `experiment/`: `ModelDriver` candidate, partition examples, metric scorers |
| 5 | Applied (L2) | not yet detailed | Applied harnesses, config groups under `config/harness/`, runnable examples |
| 6 | Runtime (L3) | not yet detailed | Runtimes (extras), reporters, optional CLI; revisit the `agent-runtime` category |
| 7 | Observability (L3) | [stage-7-observability.md](stage-7-observability.md) | `observability/`: a minimal adjusted `Event` (kind, duration, error, parent), a no-integration default (events table, JSONL and console), and Phoenix and MLflow exporters over the existing `velari-ai` connectors. Planned, nothing built |

## Existing code to reuse

| Existing piece | Path | Used by |
|---|---|---|
| `Experiment`, `ConfigOmegaConf` | `velari_core/core/experiment.py` | Seed, run identity, outputs (lazy, opt-in) |
| `Agent.run()` -> `AgentResponseInfo` (latency, usage, message stats) | `velari_ai/integrations/langchain/agents/agent.py`, `.../models/types.py` | Evaluation candidate (stage 3) |
| `LLMGateway.chat()` -> `GatewayResponse` (perf, usage) | `velari_ai/infra/gateway/llm/gateway.py` | Evaluation candidate (stage 3) |
| `ModelDriver` (seeded, `from_config`) and clustering, projection | `velari_ai/integrations/sklearn/{driver,clustering,projection}.py` | Experiment candidate (stage 4); `from_config` pattern (stage 2) |
| `DataPartition`, `cross_val_scores`, `scorer_clustering` | `velari_ai/integrations/sklearn/{partition,scoring}.py` | Experiment examples and scorers (stage 4) |
| Eval functions | `velari_ai/ai/evals/{scoring,heuristics,distribution}.py` | Scorer wrappers (stage 3) |
| Phoenix evaluators and judge | `velari_ai/integrations/arize/evaluation.py` | Optional judge scorer (stage 3) |
| `TracingConnector`, `TracerLike`, `SpanLike` | `velari_ai/ai/tracing/trace.py` | Connection and config for the exporters (stage 7); the two protocols are too thin to export from |
| Phoenix connector and OpenInference spans | `velari_ai/integrations/arize/{connection,trace}.py` | `PhoenixExporter` (stage 7) |
| MLflow connector | `velari_ai/integrations/mlflow/connection.py` | `MlflowExporter` (stage 7) |
| Datasets and registry | `velari_data/{datasets,registry}.py` | Example sources; `Registry` is the base of `RecordRegistry` and `SkillRegistry` (increment 16) |
| Skill format and sample skills | `_build/agent-skills/templates/skill-name/SKILL.md`, `config/prompts/skills/` | `Skill` and `SkillRegistry` tests and the skill-trigger example (increment 16) |

Gaps found (the core must supply them): a candidate abstraction, a case/dataset type, a scorer protocol, run records, aggregation, repeated trials, an isolated environment, and LLM seeding. Existing scorer signatures are inconsistent (`wordiness(expected, output)` vs `exact_match(output, expected)`; Phoenix evaluators return dicts), so stage 3 wraps them rather than changing them. `examples/ai/evals/data/01-scores.csv` (`experiment_id, variant_id, example_id, score, judge_model, seed`) is the target shape for `Result.to_frame()`.

## Final package layout

Import name `velari_harness`. Subpackages have no `__init__.py`, per `dev_guides.md`.

```
packages/velari-harness/
├── Makefile
├── README.md
├── pyproject.toml
└── velari_harness/
    ├── __init__.py
    ├── core/            # stage 2: L0 kernel
    │   ├── schemas/         # plain data: request/ (task, environment, case), response/ (result, record, output, score), types
    │   ├── protocols/       # extension points: Candidate, Sandbox, Scorer
    │   ├── results/         # metrics, store; created when there is behavior
    │   ├── execution/       # HarnessConfig, limits and error policy, LocalSandbox, Runner, Harness
    │   └── registry/        # Registry, named registries, config build
    ├── evaluation/      # stage 3: L1
    ├── experiment/      # stage 4: L1
    ├── applied/         # stage 5: L2
    ├── observability/   # stage 7: L3 trace exporters (JSONL, console, Phoenix, MLflow)
    └── cli.py           # stage 6 (optional)
```

## Cross-cutting rules

- Follow `dev_guides.md`, `import_guides.md`, `doc_guides.md` and `test_guides.md`.
- Tests live under `tests/harness/`, mirroring modules, with plain pytest fixtures and small local fakes.
- Package `__init__.py` files take `__version__` from `velari_core.core.utils.package_utils.package_version(__name__)` and define their own `logger`; no per-package `version.py`.
- No `TYPE_CHECKING` guards. Heavy optional dependencies come in through `velari-ai` extras.
- Dependency direction is one way: L0 imports nothing from L1, and L1 never imports L2. Inside `core/` the order is `schemas` <- `protocols` <- `results` <- `execution` <- `registry`; a folder imports only from folders to its left.
- Optional install extras are for harness runtimes (L3), not forwarded `velari-ai` dependencies. None are defined yet.
- Verify with real calls, not just diagnostics (`refactor_guides.md`). Use `make typecheck` if the IDE MCP is unavailable, and say so.
- Remove scratch files, caches or registered state created during verification (`agent_guides.md`).

## Open questions

Resolved:
- `Experiment` tracking is lazy and opt-in (stage 2).
- The package has a minimal `Makefile` and the root `install_kernel` target is left alone (stage 1).
- Layering: a core layer for all harnesses, category layers on top (user decision).
- Component name: **Candidate**, not Subject (user decision); see the glossary's naming note.

Still open:
1. Default candidate type. Recommendation: the core knows only `Candidate` and `CallableCandidate`; stages 3 and 4 add the agent, gateway and `ModelDriver` adapters.
2. ~~Sync or async~~ Resolved: sequential and synchronous in stage 2; concurrent trials arrive with the evaluation category in stage 3, async later.
3. ~~Dataclasses or pydantic~~ Resolved: core schemas are pydantic (as in `velari_ai` schemas); `HarnessConfig` is a dataclass because it is an OmegaConf config.
4. L0 directory name: `core/` (recommended) or `base/`.
5. Rename or retire the stub `EvaluationHarness` in `velari_ai/integrations/arize/evaluation.py` (stage 3).
6. Whether `agent-runtime` is wanted, or stays a noted future category.
7. Registering a duplicate name: raise (current plan) or replace.
8. Define the runtimes, and add one extra per runtime that needs its own packages (stage 6).
9. Whether run groups, resume and grouped statistics become a stage 2b or fold into stage 3.
10. Revisit the name **Task** (deferred, user decision). It came from Inspect, lm-eval and Harbor, but may depend on how it is used; **Scenario** is a candidate name where an individual record is sliced, which overlaps what the glossary calls an **Example**.
11. Stage 7: whether MLflow can create spans with explicit timestamps and add events manually (its manual-tracing page does not say), and so whether post-hoc export is enough or a live `Observer` is needed.
12. Stage 7: adding `TrialRecord.started_at` so an exporter can place a trial in time.
