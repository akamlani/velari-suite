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
| Multi-party | Concordia, OASIS, PettingZoo, OpenSpiel: several agents or simulated people act in one world | Every party; one `step(action)` takes a single party's action, so this needs a different shape |
| Replay | Raindrop Simulate: recorded production events replayed against a changed agent, with its services simulated and stateful | The services; user turns come from the recording |
| Learned | Qwen-AgentWorld: a trained language world model predicts the observation for each action | A model; it has no ground-truth state, only the transcript |

Inspect and Harbor provision a real sandbox, which is isolation, not simulation. lm-evaluation-harness, HELM and MLE-bench have no simulation. Simile simulates people, not an agent's world, so it is out of scope. The per-framework comparison is in [framework-comparison.xlsx](../framework-comparison.xlsx), and the sources are in the [research references](plan.md#research-references). The full taxonomy of worlds and purposes is on the Taxonomy sheet of [framework-comparison.xlsx](../framework-comparison.xlsx).

The base runner does not change: none of the references puts the loop in its runner. The shared base is dataset, candidate, scorer and repeats, which the core already has.

### Archetypes and how they combine

The references fall into eight archetypes. This is a synthesis of the research, not a taxonomy any one framework uses; the per-framework classification it draws on is on the Taxonomy sheet of [framework-comparison.xlsx](../framework-comparison.xlsx).

| Archetype | What it is | Seen in | Grading usually reads | Fits `World` as built |
|---|---|---|---|---|
| No world | Input in, output out | HELM, classic evals, our `Runner` | The output against a reference | Not needed: the `Candidate` path |
| Provisioned real world | A real sandbox with real tools, files and services | Harbor, Terminal-Bench, APEX-SWE, OpenWorldEngineer, HUD, verifiers runtimes | Tests or a verifier on the final workspace | Partly: `state()` is a workspace reference, and the agent usually drives |
| Simulated stateful app world | Coded fake services with a database | tau-bench and tau², AppWorld, ToolSandbox, Archipelago, Raindrop's services | Final state against expected state, plus information checks | Yes: the target case |
| Simulated party | A scripted or LLM user or counterpart | tau² user simulation, verifiers `user-sim`, Coval, MLflow | The conversation and the resulting state | Yes: the target case |
| Multi-party or social | Several agents or people in one world | Concordia, OASIS, PettingZoo, OpenSpiel | Per-party returns, behavior metrics, or none | No: needs one action per party |
| Continuous physical or control | Time passes; numeric observations | Gymnasium, dm_control, Isaac Lab, robosuite, CARLA, Habitat | A return from a reward function | Not directly: needs timed steps |
| Learned world | A model predicts the environment's response | Qwen-AgentWorld; LLM-simulated tools in Simia and VULCAN | A judge on fidelity, or a trajectory scorer | Yes, but `state()` is only the transcript |
| Replay world | Recorded events replayed against a changed agent | Raindrop Simulate; AgentWorldBench | A paired comparison with the recording | Yes: actions come from the recording |

A ninth item is not a world: **control flow across agents**, which is verifiers v1's "Environment" (best-of-n, an agent plus a judge, an agent plus a simulated user). It composes worlds and agents.

Archetypes combine along independent axes:

| Axis | Values seen |
|---|---|
| Who drives the loop | The harness calls `step` (Gymnasium, DSGym), or the agent runs its own loop and calls the world's tools (HUD, Archipelago, Raindrop, OpenWorld-harness, verifiers harnesses) |
| Time | Turn-based, event-driven, continuous |
| Who plays the other parties | Nobody, a script, an LLM, a recording, a trained model |
| What grading reads | The output, final state against initial, tests, a rubric with a judge, a return |
| Where the world runs | In process, a local subprocess, Docker, a remote sandbox |
| Purpose | RL training, evaluation, data generation, behavior research, monitoring, change-impact testing |
| Fidelity | Exact (real systems), coded approximations, learned approximations; only Raindrop and Qwen-AgentWorld measure it |

For example, tau² is a stateful app world plus a simulated party, driven by the harness and graded on state; Raindrop is a replay world over simulated services, driven by the agent; Archipelago is a real sandbox with simulated apps, driven by the agent and graded on snapshots.

The base layer covers the middle: turn-based, one agent, a world the harness steps (stateful, party, learned and replay worlds). The edges are open questions below, not oversights: multi-party, continuous time, and agent-driven or sandboxed worlds.

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

## Applied use on top of this layer

Applied harnesses can sit on top of this layer. Raindrop's change-impact replay is an example: examples taken from production traces, a world built from those traces, the new agent version as the candidate, and a paired comparison with the production run as the baseline. It needs worlds built from traces, a loader from events, a baseline comparison (stage 3) and a measure of world fidelity. None of these changes this layer's interface; the loop can be driven by the harness (`run_episode`) or by the agent itself, which calls the world's tools (see the open question on who drives the loop in the [environment note](../architecture/environment.md)).

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
- **World fidelity:** Raindrop reports a faithfulness measure for its simulated World, and we have none. How well a world reproduces reality is not part of this layer, but it belongs in the evaluation layer. Qwen-AgentWorld's benchmark shows a method: replay recorded real actions into the world and score each predicted observation against the real one with a judge, so a world can be evaluated as a candidate.
- **Worlds without ground-truth state:** a world that is itself a language model (such as Qwen-AgentWorld) has no state of its own, so `state()` returns the transcript and state-based grading is unavailable; a judge or a trajectory scorer is needed.
- **Agent-driven worlds speak MCP:** Archipelago, HUD and OpenEnv all give the agent the world through MCP tools, so an agent-driven candidate would likely expose a `World` as an MCP server that any MCP agent can attach to; not built here.
- **Large states:** Archipelago snapshots a whole file tree before and after, and OpenWorld-harness submits work as Git commits. For a file-based world, `state()` could return a reference to a copy in the trial's directory instead of an in-memory value; to decide when such a world is built.
- **Wall-clock budgets:** OpenWorld-harness paces the agent against a time budget and keeps the hard timeout above it so the final submit can finish. A time limit is an outside stop like `max_steps`, so it would be `TRUNCATED`; where its limit lives is open.
- **Where a world runs:** verifiers offers subprocess, Docker and remote-sandbox runtimes and refuses tasks that need an image or a network policy in the local one; a world that needs isolation or services would need the same choice, which is not built here.
- **Agent-driven worlds:** `SimulatedCandidate` covers a harness-driven loop. A candidate whose agent runs its own loop against the world's tools is an applied-layer piece, not built here.
- **The `velari_ai` import** in `llm_user.py` is the one coupling. If it becomes a problem, the file moves to the evaluation layer (stage 3).
