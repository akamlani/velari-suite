# Actor-Critic Loop: Separate Drafting + Grading Nodes Template

Reusable template for a two-role critique/revision loop: a dedicated **actor** node drafts or
revises content, a dedicated **critic** node grades it, and the two alternate until the critic
signals sufficiency or an iteration cap forces termination. Unlike the
[single-node self-reflection template](reflection.md), the two roles are separate calls (and can
be separate models), which costs more per pass but allows genuinely different models, temperatures,
or prompting per role — e.g. a stronger/pricier critic model paired with a cheaper actor model.
Framed around a LangGraph-style `StateGraph`, but the state shape and role-separation decisions
below apply to any graph/agent framework with equivalent primitives.

Start with the self-reflection template unless the actor and critic genuinely need different
models or prompting strategies — splitting into two nodes only pays for itself once that
divergence is real.

## Key decisions to make up front

1. **Independent, optional model config per role.** Accept an optional `ModelConfig` for the actor
   and one for the critic in the constructor, each defaulting independently (e.g. actor defaults to
   a general-purpose model, critic defaults to the same model at a lower/zero temperature for more
   consistent grading). This makes same-model and different-model configurations both a
   zero-code-change choice for the caller — nothing forces the two to diverge, but nothing
   prevents it either.
2. **The critic node owns the iteration cap and its increment**, not the actor node. This mirrors
   the common pattern of putting cap/exhaustion logic on the "gate" node rather than the
   "produce" node, and keeps the actor a pure produce/revise function with no budget bookkeeping.
3. **One actor node handles both the first draft and every revision**, distinguished by whether
   `critique` is present in state (empty on the very first call) — not a separate "first draft"
   node. Keeps the graph at exactly two nodes regardless of how many revision passes occur.
4. **Two prompts, strictly separated by responsibility.** The actor's prompt only ever produces or
   revises content; the critic's prompt only ever judges and must not rewrite content itself.
   Blurring this (e.g. letting the critic "helpfully" suggest a rewrite inline) undermines the
   reason to split actor/critic in the first place — the critic's output should never leave
   `content` out of sync with what the actor node actually produced.

## Generic steps

1. Define a shared, generic state: `task: str` (required), plus `content`/`critique`/`verdict`/
   `iterations`/`exhausted`, all optional. Reuse the exact same state/verdict shape as the
   self-reflection template if both live in the same codebase — no reason to duplicate it.
2. Define two structured-output schemas:
   - Actor: `content: str` — just the draft or revision, no verdict fields.
   - Critic: `sufficient: bool`, `critique: Optional[str]` (populated when insufficient).
3. Write two prompts:
   - **Act**: given `task`, current `content` (empty on the first pass), and `critique` (empty
     until the critic has run once), produce a first draft or a revision addressing every point in
     the critique. Instruct it to preserve whatever the critique didn't flag.
   - **Critique**: given `task` and `content`, judge sufficiency and, when insufficient, name
     concrete, actionable gaps — not generic feedback. Instruct it explicitly not to rewrite the
     content itself.
4. Implement the actor node: render the act prompt, invoke, narrow the result type via
   `isinstance` (not `cast()`), return the new `content`. No cap check, no iteration increment here.
5. Implement the critic node:
   - Check the iteration cap **first**; past it, return `{"verdict": SUFFICIENT, "exhausted":
     True}` without calling the model.
   - Otherwise render the critique prompt, invoke, narrow the result type.
   - On `sufficient=True`, return `{"verdict": SUFFICIENT}` only.
   - On `sufficient=False`, return `critique` text, `verdict: NEEDS_REVISION`, and `iterations`
     incremented by one.
6. Wire the graph: `START -> act -> critique`; a conditional edge from `critique` back to `act` on
   `NEEDS_REVISION`, and to `END` on `SUFFICIENT`.
7. Expose a thin driver (CLI command, function, etc.) that invokes the compiled graph with just
   `{"task": "..."}`, for standalone verification independent of any larger pipeline.

## Gotchas checklist

- **Prompt-variable collisions with shared prompt metadata** — same root cause as the
  self-reflection template's gotcha, but hits *twice* here since both the act and critique prompts
  introduce a `{{ task }}` (or similarly-named) placeholder. If the prompt-loading convention
  spreads a shared "meta" dict into `.invoke()` alongside real template variables, and any variable
  name collides with a meta key, whichever is spread *last* in the dict literal silently wins — no
  error raised, the model just responds to the wrong prompt. Put explicit call-specific variables
  after the meta spread in both node functions, and verify by rendering each prompt standalone
  before wiring it into the graph.
- **Iteration accounting split across two nodes is easy to get wrong.** Decide up front which node
  owns the cap check and the increment (recommended: the critic, since it's the "gate"). Checking
  or incrementing in both nodes double-counts; checking in neither never terminates the loop
  outside the model's own judgment.
- **Critic prompt scope creep.** If the critic's prompt doesn't explicitly forbid rewriting content
  itself, it can start proposing revisions in its critique text — which blurs the actor/critic
  separation and risks the actor's next draft diverging from what the critic actually meant.
- **Defaulting both roles to identical temperature.** A critic generally benefits from low/zero
  temperature for consistent, repeatable grading; the actor can use a higher temperature for more
  varied drafts. Don't inherit one default for both just because they default to the same model
  name.

## Applied in

`examples/ai/agents/retrieval_agent.py`'s `GraphActorCritic` class — reference implementation only;
not currently fully wired (its prompt files, `act.yaml`/`critique.yaml`, don't exist in this repo).
