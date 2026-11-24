# Simulation layer (`velari_harness/simulation/`)

Part of [plan.md](plan.md). Requires stage 2 up to increment 17 (`TrialContext`, implemented; see the [build order](stage-2-base-framework.md#build-order)). Status: planned, nothing built. Goal: a base layer on top of the core where a candidate acts on a simulated world instead of answering one fixed input, with worlds that are deterministic or LLM-based and that both run offline.

## What simulation means here

A **world** holds state that a candidate changes by acting; it says when the episode ends; and a scorer checks its final state. The references simulate different things with this shape:

| Kind of world | Example | What reacts |
|---|---|---|
| Static | HUD ops-diagnostics cookbook: seeded log files the agent reads | Nothing; the grade reads the answer |
| Stateful | tau²: a database changed by tool calls. HUD coding cookbook: a repository graded by hidden tests | State that the agent's actions change |
| Reactive | tau², MLflow `ConversationSimulator`, Coval: an LLM plays the user | A second party answers back |
| Continuous | HUD robot cookbook: a physics simulator stepped at a control rate | Time advances in ticks |

Inspect and Harbor provision a real sandbox, which is isolation, not simulation. lm-evaluation-harness, HELM and MLE-bench have no simulation. Simile simulates people, not an agent's world, so it is out of scope. The per-framework comparison is in [framework-comparison.xlsx](../framework-comparison.xlsx), and the sources are in the [research references](plan.md#research-references). The full taxonomy of worlds and purposes is on the Taxonomy sheet of [framework-comparison.xlsx](../framework-comparison.xlsx).

The base runner does not change: none of the references puts the loop in its runner. The shared base is dataset, candidate, scorer and repeats, which the core already has.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Placement | A separate `velari_harness/simulation/` package that depends on `core/` only | The core stays free of the loop and the world concept; static harnesses never use them |
| Sequencing | Increment 17 (`TrialContext`) first, then this layer as increment 18 | A world cannot be seeded until the candidate receives its seed, and a trajectory needs events on the record |
| First slice | A deterministic world and an LLM-played user, both runnable offline | Proves one abstraction over both kinds, and runs in CI |
| Scorers (assumed, not yet confirmed) | `fn(output, reference)` stays; the final world state is returned inside the output | Keeps "the scorer only scores" and the existing contract; tau² and HUD also grade from the final state (a side-effects check needs both states; see [reward functions](../reward-functions.md)) |
| Step cap | Reaching it is a recorded termination, not a failure | The scorer decides what an unfinished episode means |

## Design

```
simulation/
├── world.py        # Step, World protocol, Policy protocol, Termination
├── episode.py      # run_episode(): the loop, the trajectory, the events
├── candidate.py    # SimulatedCandidate: a core Candidate that runs one episode per example
├── users.py        # ScriptedUser (deterministic, no model); the UserModel protocol
└── llm_user.py     # LLMUser and a gateway-backed UserModel (the only file importing velari_ai)
```

- **`World`**: `reset(input, seed, workdir)` builds the world from the example's input and the trial seed and returns what the agent sees first. `step(action)` returns a `Step(observation, done, info)`. `state()` is what graders read, and `close()` always runs. A static world returns `done=True` on its first step, a conversation returns the user's reply, and a physics world would advance a tick.
- **`Policy`**: `act(observation) -> action`. It is the thing under test, and any callable fits.
- **`run_episode`**: reset, then act and step until `done` or `max_steps`. Each step is a `Turn(action, observation)` and is emitted as an event through `TrialContext.emit`. It returns `EpisodeResult(initial_state, final_state, trajectory, termination, steps)`; `initial_state` lets a scorer compute what the episode changed, for a side-effects check (see [reward functions](../reward-functions.md)). An exception propagates to the Runner's existing `Failure` handling.
- **`SimulatedCandidate(make_world, make_policy, max_steps)`**: a core `Candidate` that runs one episode per example, using its trial seed and working directory, and returns an `Output` whose value is the `EpisodeResult`. `Runner` and `Harness` are unchanged. The candidate never sees `Example.reference`, so the expected end state stays hidden, as with tau² and HUD.
- **`ScriptedUser`**: a conversation partner that follows a script (replies by turn, or by seed among scripted variants) and stops on a marker. It uses no model, so the same seed gives the same trajectory.
- **`LLMUser`**: plays the user from a goal and an optional persona and rules (the split MLflow and Coval use), and sees only the conversation, never the tool calls (as in tau-bench). It depends on a small `UserModel` protocol (`reply(history) -> str`), so offline runs and tests inject a scripted or recorded responder. `llm_user.py` adds a `UserModel` backed by `LLMGateway.chat`. LLM output is only best-effort deterministic.
- **Offline**: the deterministic world needs nothing. The LLM world runs offline through an injected responder, and the example has an `offline` switch (on by default) and a live mode.

## Build order

| Step | Adds | Status |
|---|---|---|
| 17 | `TrialContext`: the seed, a per-trial working directory and `emit` events on the `TrialRecord` (the prerequisite; built in [stage 2](stage-2-base-framework.md#build-order)) | Implemented |
| 18a | `world.py` and `episode.py`, tested with a small in-test world | Planned |
| 18b | `candidate.py`: `SimulatedCandidate` under the real `Runner` and `Harness` | Planned |
| 18c | `users.py`: `ScriptedUser` | Planned |
| 18d | `llm_user.py`: `LLMUser` with a scripted responder and a fake gateway | Planned |
| 18e | The example `examples/ai/harness/simulation/`: a scripted user, a seeded incident-files world graded from its final state, and the LLM user offline, with a README | Planned |
| 18f | Docs: glossary rows and a stage 2 section, once built | Planned |

## Not in this layer

Repeated trials and `pass^k` reliability (`Environment.seed_for` already takes a trial number), concurrency, container sandboxes, physics or robot bridges, a live model as the default, a change to the scorer signature, and a record store.

## Open questions

- **Scorer contract:** confirm it stays `fn(output, reference)`. If scorers should see the trajectory directly, that is a core change to settle before 18b.
- **Shape of increment 17:** built as this layer assumed: a seed, `emit` and a working directory, passed explicitly as `ctx` (a temporary directory by default, kept under a root).
- **`reset` input:** passing only the example's input keeps the reference out of the world; revisit if a world needs more.
- **LLM determinism:** an LLM user can differ between runs even with a fixed seed; repeats are the later answer.
- **Timed steps:** once [stage 7](stage-7-observability.md) adds `ctx.span`, a world step can be a timed, nested event; until then a step is a point event.
- **The `velari_ai` import** in `llm_user.py` is the one coupling. If it becomes a problem, the file moves to the evaluation layer (stage 3).
