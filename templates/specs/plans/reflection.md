# Self-Reflection Loop: Single-Node Critique + Revision Template

Reusable template for a lightweight quality-improvement pass on any single piece of LLM-generated
content: one node critiques its own draft and revises it in the same call, looping back on itself
(a genuine self-edge in the graph) until it reports the draft sufficient or an iteration cap forces
termination. Framed around a LangGraph-style `StateGraph` (conditional edges, structured-output
nodes), but the state shape, prompt structure, and iteration-accounting decisions below apply to
any graph/agent framework with equivalent primitives.

This is the cheaper sibling of the [actor-critic template](actor_critic.md) — one model call per
pass instead of two, at the cost of not being able to give the "critic" role a different model or
temperature than the "actor" role. Reach for this one first; split into actor-critic only once you
actually need role-specific models.

## Key decisions to make up front

1. **One combined call, not two.** Critique and revision happen in a single structured-output call
   per pass, not a separate critique-then-revise round trip. Cheaper, and mirrors the common
   pattern of "silently diagnose, then act" in one call rather than exposing the diagnosis as a
   separate step.
2. **Bounded self-loop, not single-shot.** Wire a conditional edge from the node back to itself so
   a `needs_revision` verdict re-enters the same node, capped by a dedicated iteration budget.
   Give this budget its own constant — don't reuse an iteration cap that already means something
   else in the same codebase (e.g. a retrieval-retry budget); the two failure modes are unrelated
   and want independently tunable limits.
3. **Generic state, not domain-specific.** Use neutral field names (`task`, `content`, `critique`,
   `verdict`, `iterations`, `exhausted`) instead of domain terms like `query`/`answer`/`context` —
   keeps the same state and graph reusable across unrelated problem domains.
4. **Boolean verdict in the schema, enum verdict in state.** The structured-output schema uses a
   plain `sufficient: bool` field; the calling node converts that bool into a `StrEnum` verdict
   (`SUFFICIENT`/`NEEDS_REVISION`) before writing it to state — keeps routing logic
   (`route_map={...}`) working off a small, closed set of enum values rather than raw booleans or
   strings scattered through the graph.
5. **Exhaustion is visible, not silent.** When the iteration cap is hit before a genuine
   `sufficient` verdict, force-accept to guarantee the graph terminates, but set an explicit
   `exhausted: True` flag alongside the forced verdict — callers can then distinguish "confidently
   approved" from "ran out of budget," instead of the two looking identical downstream.

## Generic steps

1. Define the state: `task: str` (required — what the content must satisfy), plus
   `content`/`critique`/`verdict`/`iterations`/`exhausted`, all optional with sensible empty
   defaults on the first pass.
2. Define a two-member verdict enum: `SUFFICIENT` / `NEEDS_REVISION`.
3. Define one structured-output schema for the combined call: `sufficient: bool`,
   `critique: Optional[str]` (populated when insufficient), `revised_content: Optional[str]`
   (populated when insufficient — the full revision, not a description of what to fix).
4. Write one prompt instructing the model to: judge completeness/correctness/clarity against the
   task, not just writing quality; when insufficient, produce a complete revised draft rather than
   describing the fix; preserve whatever in the draft already works rather than rewriting from
   scratch each pass.
5. Implement one node function:
   - Check the iteration cap **first**. Past the cap, return `{"verdict": SUFFICIENT, "exhausted":
     True}` immediately, without calling the model.
   - Otherwise, render the prompt with `task`/`content`, invoke the structured-output model,
     narrow its return type (structured-output APIs commonly return a broad union type even when
     passed a concrete schema class — narrow via `isinstance`, don't `cast()`).
   - On `sufficient=True`, return `{"verdict": SUFFICIENT}` with no other state changes.
   - On `sufficient=False`, return the `revised_content` as the new `content`, the `critique` text,
     `verdict: NEEDS_REVISION`, and `iterations` incremented by one — increment **only** on this
     branch, so the first pass doesn't already count against the budget.
6. Wire the graph: `START -> reflect_and_revise`; a conditional edge from `reflect_and_revise` to
   itself on `NEEDS_REVISION`, and to `END` on `SUFFICIENT`.
7. Expose a thin driver (CLI command, function, whatever fits the codebase) that invokes the
   compiled graph with just `{"task": "..."}` — useful for standalone verification independent of
   whatever larger pipeline eventually consumes this pattern.

## Gotchas checklist

- **Prompt-variable collisions with shared prompt metadata.** If the prompt-loading convention
  spreads a shared "meta" dict (e.g. a human-readable label describing what the *prompt itself* is
  for) into the same `.invoke()` call as the real template variables, and a template variable name
  happens to collide with a meta key, the metadata silently wins if it's spread *after* the real
  value in the same dict literal — Python's dict construction lets a later duplicate key silently
  overwrite an earlier one, no error raised. Symptom: the model runs and returns a plausible-looking
  answer, but it's responding to a completely different, nonsensical prompt — easy to misdiagnose
  as a model-quality problem instead of a variable-shadowing bug. Fix: put explicit,
  call-specific variables **after** any spread of shared/default values in the dict literal (e.g.
  `{**shared_meta, "task": task, ...}`, not `{"task": task, ..., **shared_meta}`). Verify by
  rendering the prompt template standalone (no LLM call) and reading the literal rendered text
  before wiring a new node into a graph.
- **Sharing one iteration/retry counter across unrelated loops.** Don't reuse an existing
  `MAX_ITERATIONS`-style budget meant for a different retry loop — give this pattern its own
  dedicated constant.
- **Unconditional increment counts the first pass against the budget.** Increment `iterations`
  only on the "needs more work" branch, not on every node entry — otherwise the loop terminates
  faster than the configured budget suggests.

## Applied in

`examples/ai/agents/retrieval_agent.py`'s `GraphSelfReflection` class — reference implementation
only; not currently fully wired (its prompt file, `self_reflect.yaml`, doesn't exist in this repo).
