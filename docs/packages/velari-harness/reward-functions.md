# Reward functions in non-physical simulated environments

A reference note for the [simulation layer](plan/simulation.md). It explains what a reward function is when the world is software (apps, databases, conversations, logs) and not physics, how it works in RL and non-RL settings, and how to design one for applied cases. It is not a build increment, and nothing in it is implemented. Terms are defined in the [glossary](glossary.md); the frameworks mentioned are in [framework-comparison.xlsx](framework-comparison.xlsx) and the [research references](plan/plan.md#research-references).

The vertical examples (healthcare, finance, cybersecurity and others) are illustrative designs built from the structure of the sources, not clinical, financial or security guidance. The cybersecurity examples are defensive and run only in a fully simulated environment.

## Terms

| Term | Meaning here |
|---|---|
| **Reward function** | A function from what happened in an episode, judged against a specification, to a number (or a small vector) saying how well the objective was met |
| **Reward** | The value it returns. In RL it can be returned at every step; in evaluation it is usually computed once, at the end |
| **Return** | The sum (often discounted) of the rewards over an episode; what an RL agent tries to raise |
| **Score** | The evaluation-side name for the same thing: what a scorer returns for one trial. Our `Score` model |
| **Metric** | An aggregate of scores or rewards over many trials (mean, standard error, `pass^k`) |
| **Spec** | The hidden definition of success: an expected end state, a policy document, a rubric. The candidate never sees it |
| **Grader, verifier, scorer** | Names for the code that computes the reward or score (HUD: grader; Harbor: verifier; ours: scorer) |

## Definition

```python
# R is the reward function. Its four inputs:
#   spec           the hidden definition of success (an expected end state, a policy, a rubric); the candidate never sees it
#   trajectory     what happened: the sequence of actions and observations
#   initial_state  the world before the episode (for example, database contents)
#   final_state    the world after it; comparing the two shows exactly what the agent changed
reward = R(spec, trajectory, initial_state, final_state)     # a scalar, or named parts
```

- `spec`: what success means, hidden from the candidate.
- `trajectory`: the actions and observations of the episode.
- `initial_state` and `final_state`: the world before and after, such as a database, a ledger or a repository.

In a physical world the reward can lean on physics (distance, collision, balance). In a software world there is nothing to read the answer from, so the reward is **specification-based**: it comes from expected state changes, constraints and required information that someone wrote down. That is why tau², AppWorld and HUD all build it from hidden expected states, tests or rubrics.

## What a reward is made of

| Component | Question it asks | Seen in |
|---|---|---|
| **Outcome** | Did the world end in the right state? | tau²: the final database equals the one expected state |
| **Information** | Did the agent say what had to be said? | tau²: required amounts appear as substrings in the agent's messages |
| **Process** | Did it follow the rules and the order? | ToolSandbox: milestones matched over any trajectory |
| **Side effects** | Did it change anything it should not have? | AppWorld: tests for "collateral damage" |
| **Quality** | Is the answer good where no exact answer exists? | HUD ops cookbook: an LLM judge against criteria |
| **Efficiency** | How many steps, how much cost or time? | HUD's step cap (default 200); MLE-bench's runtime budget |
| **Safety gate** | Was a hard rule broken? | A design choice: a broken gate zeroes the reward whatever else happened |

### Ways to combine components

| Combiner | Behavior | Use when | Seen in |
|---|---|---|---|
| **Product** | Any zero factor gives zero | Every part is required | tau²: `r = r_action × r_output`, each 0 or 1 |
| **Geometric mean** | One zero zeroes the whole, otherwise parts trade off smoothly | Parts are all needed but partial credit is useful | ToolSandbox: a milestone's similarity is the geometric mean of its measures |
| **Weighted sum** | Strength in one part pays for weakness in another | Parts are genuinely substitutable | HUD's `combine`, which weights several graders |
| **Gate, then score** | A hard rule vetoes; otherwise score normally | Safety, compliance, permissions | A design choice; weights should not buy back a violation |
| **Vector, not collapsed** | Report each part separately | Trade-offs are a decision for a person, such as fairness versus accuracy | Our multiple scorers, one `Score` each |

```python
# pseudo-code, not implemented
@dataclass(frozen=True)
class Reward:
    """One reward: a number plus the reasons behind it, so a low score can be explained."""
    value: float                 # 0..1; the single number that RL optimizes or an evaluation reports
    parts: Dict[str, float]      # named components, e.g. {"outcome": 1.0, "side_effects": 0.0}; keeping them is what makes a bad score debuggable
    vetoed: bool = False         # True when a safety gate fired; the value is then forced to 0.0
    why: str = ""                # a human-readable reason; it ends up in Score.explanation

def combine(parts, weights, gates):
    # parts:   component scores in 0..1, e.g. {"outcome": 1.0, "process": 0.5}
    # weights: how much each part counts; they should sum to 1 so the result stays in 0..1
    # gates:   hard rules by name, True when satisfied, e.g. {"identity_verified": True}
    # 1) Gates first: one broken rule vetoes the whole reward, whatever the parts say.
    if broken := [name for name, ok in gates.items() if not ok]:
        return Reward(0.0, parts, vetoed=True, why=f"violated {broken}")
    # 2) Otherwise a weighted sum: strength in one part can offset weakness in another.
    return Reward(sum(weights[k] * parts[k] for k in weights), parts)
```

## Verifiable versus judged rewards

| | **Verifiable (state or rule based)** | **Judged (LLM or human)** |
|---|---|---|
| What it does | Computes the reward from the world's state or from exact rules | Scores against a rubric or criteria |
| Examples | tau² database comparison; AppWorld unit tests; HUD `BashGrader` (a command's exit code); MLE-bench's grading script | HUD ops cookbook `LLMJudgeGrader`; MLflow scorers such as `ConversationCompleteness`; Coval's AI judge |
| Strengths | Exact, repeatable, cheap, hard to argue with | Handles open answers and quality |
| Risks | Brittle to the spec being wrong; cannot judge what it did not encode | Variance, bias, cost, and exploitable wording |
| Rule of thumb | Prefer it wherever the answer can be checked | Use it only where no exact check exists, and audit it |

A judged reward needs its own validation (see below), because the judge can be wrong in the same direction as the agent.

## RL and non-RL

The same world and the same reward logic serve both. What differs is **when the number is produced and what it is used for**.

| | **RL** | **Non-RL (evaluation)** | **No reward (behavior study)** |
|---|---|---|---|
| When | After each step as `r_t` (often 0 until the end, then a terminal reward) | Once, at the end of the episode | Never; outcomes are analyzed as distributions |
| Used for | An optimization signal that changes the policy | Measurement only; nothing is updated | Understanding behavior (Concordia, OASIS, Simile) |
| Shape | Dense or shaped rewards matter, because a sparse 0 or 1 is a weak learning signal | A terminal score is enough; shaping would hide what you measure | Not applicable |
| Aggregation | The sum or discounted sum over the episode | Mean over examples and over repeats; `pass^k` | Distributions and comparisons |
| Main risk | Reward hacking: gaining reward without doing the task | A scorer that passes bad work or fails good work | Simulator not matching reality |
| Output shape | Usually one scalar | Often a vector, one score per scorer | A record of what happened |

RL training is outside our scope; this section explains the contrast so the evaluation form is clear.

**RL view, with shaping from milestones.** Gymnasium's `step` returns `(observation, reward, terminated, truncated, info)`, where *terminated* is the task's own end state and *truncated* is an outside stop such as a time limit. A common technique is potential-based shaping, which adds `γΦ(next) - Φ(current)` to each step's reward, where `Φ` is a progress measure and `γ` is the discount factor of the return (1 when there is no discounting). It leaves the best behavior unchanged only under two conditions: the potential of a terminal state is 0, and `γ` is the same one the return uses. Otherwise the shaping terms add up to `γ^T Φ(final) - Φ(initial)` (just `Φ(final) - Φ(initial)` without discounting), so an episode that ends with more progress earns a bigger bonus, and a run that makes progress but misses the real objective can score higher than one that achieves it.

```python
# pseudo-code, not implemented
GAMMA = 1.0                                          # the discount factor of the return; 1.0 means no discounting

def potential(state, spec):                          # progress toward the goal, 0..1
    # A milestone is a checkable intermediate goal, e.g. "identity verified" or "ledger entry created".
    # Progress = the fraction of milestones the world's current state satisfies.
    return sum(m.met(state) for m in spec.milestones) / len(spec.milestones)

def step(self, action):                              # Gymnasium-style
    # Progress before the action, measured on the world's state.
    before = potential(self.world.state(), self.spec)
    obs, done = self.world.step(action)
    # Progress after it; the world has now changed. The potential of a terminal state must be 0: otherwise the shaping
    # terms add up to (final progress - starting progress), which rewards where an episode ends and can change which behavior is best.
    after = 0.0 if done else potential(self.world.state(), self.spec)
    # The terminal reward scores the whole episode, once, and only when the episode ends.
    terminal = self.reward_fn(self.spec, self.trajectory, self.initial, self.world.state()).value if done else 0.0
    # Reward for this step = the shaping term, plus the terminal reward on the last step.
    # GAMMA * after - before is the potential-based shaping term. With a terminal potential of 0 and the same GAMMA as the
    # return, it pays for progress without changing which overall behavior is best (Ng, Harada and Russell, 1999).
    # `truncated` (a step cap stopped the episode, not the goal) and `info` are set elsewhere in the class; whether a cut
    # episode also gets a potential of 0 is a choice to make explicitly.
    return obs, (GAMMA * after - before) + terminal, done, truncated, info
```

**Non-RL view, matching our `Score` and `fn(output, reference)`.**

```python
# pseudo-code, not implemented
def scorer(output: EpisodeResult, reference: Spec) -> Score:
    # `output` is what the simulated candidate returned (the episode result);
    # `reference` is the hidden spec stored on the Example, which the candidate never saw.
    r = reward_fn(reference, output.trajectory, output.initial_state, output.final_state)
    # Map the reward onto our Score model:
    #   value       <- the combined number
    #   passed      <- a pass needs a high value AND no safety veto
    #   label       <- names the veto when there was one
    #   explanation <- the human-readable reason
    #   metadata    <- the named parts, so a failure can be read afterwards
    return Score(value=r.value, passed=(r.value >= 0.9 and not r.vetoed),
                 label="vetoed" if r.vetoed else None, explanation=r.why, metadata=r.parts)
```

## Sparse, dense, partial and delayed rewards

| Property | Meaning | In software worlds |
|---|---|---|
| **Sparse** | Only the end of the episode gives a signal | The default for state-based grading (tau², HUD tasks): 0 or 1 at the end |
| **Dense** | A signal at many steps | Needs milestones or a progress measure; ToolSandbox's milestone DAG is a dense-ish evaluation signal |
| **Partial credit** | Between 0 and 1 for partly right work | Fraction of tests passed (HUD coding example); milestone similarity (ToolSandbox) |
| **Delayed** | The consequence arrives after the action | A fraud decision judged by a later chargeback, or a missed filing deadline; the world must simulate the later event |
| **Terminated versus truncated** | Reaching the goal versus running out of steps | Do not treat an unfinished episode as a failure by default: record it and let the scorer decide (the plan's `Termination`) |

## Aggregating over trials

One trial is one sample. Agents and simulated users are stochastic, so repeats are how a reward becomes a measurement.

| Aggregate | What it says | Source |
|---|---|---|
| Mean and standard error | Typical reward and its uncertainty | Our `Result.summary` |
| `pass@k` | Chance that at least one of `k` trials succeeds: `1 - E[C(n-c, k) / C(n, k)]` | tau-bench paper |
| `pass^k` | Chance that all `k` trials succeed (reliability): `E[C(c, k) / C(n, k)]`, with `c` successes in `n` runs | tau-bench paper |
| At least 3 seeds, mean and SEM | A practical floor for noisy agents | MLE-bench |

```python
# pseudo-code, not implemented
def pass_hat_k(successes: int, n: int, k: int) -> float:    # per task; then average over tasks
    # One task run n times, `successes` of them succeeded. This is the chance that k runs
    # picked at random ALL succeed, which measures reliability, not luck.
    # Example: 8 successes in 10 runs gives pass^1 = 0.80 but pass^3 = C(8,3) / C(10,3) = 56 / 120 = 0.47.
    # The more runs you require to all succeed, the lower it falls.
    return comb(successes, k) / comb(n, k)
```

Repeats are deferred in our plan; `Environment.seed_for(key, trial)` already takes a trial number for it.

## Applied cases by vertical

Each case below raises the complexity. The "easy way to cheat" column is what a poorly designed reward would accidentally pay for.

| Vertical | Case | Complexity | World | Reward is mostly | Easy way to cheat |
|---|---|---|---|---|---|
| **Healthcare** | Benefits or prior-authorization inquiry | Medium: stateful, with a hard gate | Policy database, member record, simulated caller | Correct determination (outcome), required documentation cited (information), no disclosure before identity verification (gate), no unrelated record changes (side effects) | Always approve, or always deny |
| | Symptom triage with a simulated patient | High: partial observability, asymmetric cost | A simulated patient who withholds details until asked | A safe disposition against a guideline, with under-triage costing more than over-triage; required red-flag questions asked (process) | Always escalate to the highest level |
| **Finance** | Card dispute handling | Medium: stateful ledger | Accounts, ledger, dispute policy | Right disposition (outcome), correct ledger entries with no duplicate credit (state diff), identity check first (gate) | Refund everything |
| | Fraud or AML alert review | High: asymmetric and delayed | An alert queue, seeded transactions, deadlines | A cost matrix where a missed true positive costs far more than a false alarm; evidence cited; decided before the deadline | Mark every alert suspicious, or every alert benign |
| **Cybersecurity** | Alert triage on seeded logs | Medium: a static world, like HUD's ops cookbook | Log files and deploy history | Root cause named and evidence cited (judge against criteria) | A vague answer that fits many causes |
| | Incident response in a simulated enterprise network (defensive) | High: stateful, reactive adversary, trade-offs | Hosts, accounts, firewall rules; a scripted or LLM-driven attacker | Containment achieved (state) minus business disruption (isolating a production database is collateral), time to contain, no destructive action without approval (gate) | Isolate every host |
| **Insurance** | Claims adjudication | Medium: stateful, policy lookups | Policy and claim records, coverage rules | Correct payout and reasoning, fraud indicators raised, no payout above the limit (gate) | Approve small claims, deny large ones |
| **Compliance** | Document review against a policy set | Medium: judged | A document set and a policy | Findings match the labeled violations (precision and recall), each cited to a clause | List every clause as a finding |
| **Data operations** | Pipeline incident (late files, schema drift) | Medium: static or stateful | Seeded logs and a faulty dataset | Right cause, right fix, no other tables touched | Restart everything |

### Healthcare: triage with an asymmetric cost

```python
# pseudo-code, not implemented
# Cost of a wrong disposition, keyed by (true level, predicted level). Only the listed mistakes cost
# anything; a full table would list every pair. Under-triage costs 10x more than over-triage.
COST = {("emergency", "routine"): 10.0,        # (true level, predicted level): under-triage
        ("routine", "emergency"): 1.0}         # over-triage

def triage_reward(spec, trajectory, final_state):
    # Process: which of the required red-flag questions did the agent actually ask the simulated patient?
    asked = {q for q in spec.red_flag_questions if question_was_asked(trajectory, q)}
    process = len(asked) / len(spec.red_flag_questions)
    # Outcome: look up the cost of the agent's final disposition against the true level.
    # 0 cost means it matched or the mistake is not in the table.
    cost = COST.get((spec.true_level, final_state["disposition"]), 0.0)
    # Turn the cost into a 0..1 score: no cost -> 1.0, the worst cost (10) -> 0.0, over-triage (1) -> 0.9.
    outcome = max(0.0, 1.0 - cost / max(COST.values()))
    # Weighted sum, no hard gates in this simple version.
    return combine({"outcome": outcome, "process": process}, {"outcome": 0.7, "process": 0.3}, gates={})
```

The simulated patient shows the same trait as tau²'s user: it holds information back until the agent asks, so the reward must credit asking the right questions, not only the final answer.

### Finance: a dispute with a state diff and a gate

```python
# pseudo-code, not implemented
def dispute_reward(spec, initial, final, trajectory):
    # Gate: the agent must have verified identity BEFORE its first change to any record.
    verified = identity_verified_before_first_change(trajectory)           # gate
    # What the episode changed, as {table: {key: (old, new)}}; this is why the scorer needs both states.
    changed = diff(initial, final)                                         # {table: {key: (old, new)}}
    # Outcome: the ledger changes match exactly the expected ones, so no missing and no duplicate credit.
    outcome = float(changed.get("ledger") == spec.expected_changes["ledger"])
    # Side effects: any table touched that the spec did not expect to change.
    collateral = set(changed) - set(spec.expected_changes)                 # tables touched that should not be
    parts = {"outcome": outcome, "no_collateral": float(not collateral)}
    # Gate first (identity), then a weighted sum of the two parts.
    return combine(parts, {"outcome": 0.8, "no_collateral": 0.2}, gates={"identity_verified": verified})
```

### Cybersecurity: containment versus disruption

```python
# pseudo-code, not implemented
def incident_reward(spec, initial, final, trajectory):
    # Outcome, read from the final state: can the attacker still reach the compromised foothold?
    contained = not attacker_reachable(final, spec.attacker_foothold)       # outcome, from state
    # Side effects: services that are down now but were not down at the start, weighted by how
    # important each is; isolating a production database costs more than isolating a test box.
    disruption = sum(spec.service_weight[s] for s in services_down(final) - services_down(initial))
    # Gate: every destructive action (isolate a host, disable an account) needed approval first.
    approved = all(was_approved(a) for a in trajectory.actions if a.destructive)   # gate
    # Efficiency: faster containment scores higher, relative to the step budget.
    speed = max(0.0, 1.0 - steps_to_containment(trajectory) / spec.step_budget)
    # The disruption is capped at 1 and inverted, so low disruption scores high.
    parts = {"contained": float(contained), "low_disruption": 1.0 - min(1.0, disruption), "speed": speed}
    # Containment counts most, then low disruption, then speed; an unapproved destructive action vetoes it all.
    return combine(parts, {"contained": 0.6, "low_disruption": 0.3, "speed": 0.1}, gates={"approved": approved})
```

The adversary here is a reactive world (kind 3 in the taxonomy). Its behavior is part of what makes the reward noisy, so results should be reported over repeats.

## Designing a reward: a checklist

1. **Write the spec first.** What end state, constraints and required information define success? If two reviewers disagree, the reward will too.
2. **Choose verifiable checks wherever the answer can be checked**, and a judge only for the rest.
3. **List the hard rules and make them gates**, not weights.
4. **Add a side-effects check.** Compare the initial and final state and penalize changes that should not have happened (AppWorld).
5. **Use the real cost structure.** In many verticals a miss is worse than a false alarm; encode it as a cost matrix instead of accuracy.
6. **Keep the parts.** Return named components so a failure can be read, and report them as a vector where the trade-off is a human decision.
7. **Decide what unfinished means.** A truncated episode is recorded, not automatically zero.
8. **Plan the repeats.** If anything in the loop is stochastic, a single run is not a measurement.

## Validating a reward

A reward is code that can be wrong. HUD's QA agents audit completed traces for reward hacking, false positives, false negatives, and grader checks that the prompt never states; the same questions apply to any reward we write.

| Check | How | Catches |
|---|---|---|
| **Known-good trajectories** | Hand-write a few correct runs; they must score near 1 | A grader that rejects correct work (false negatives) |
| **Known-bad trajectories** | Hand-write wrong runs, including plausible shortcuts; they must score near 0 | A grader that passes broken work (false positives) |
| **Cheat probes** | Run the easy cheats from the table above (always approve, isolate everything) | Rewards that pay for degenerate behavior |
| **Perturbation tests** | Change irrelevant details of a good run (wording, order); the score should hold | Brittle checks |
| **Judge calibration** | Compare a judge's scores with human labels on a sample | A judge that is biased or noisy |
| **Variance across repeats** | Run the same trial many times | A reward or simulator too noisy to rank candidates |
| **Spec versus prompt** | Check that every graded requirement is stated to the candidate | Hidden requirements the agent cannot know |

## Pitfalls

| Pitfall | Example | Mitigation |
|---|---|---|
| Reward hacking | An agent weakens a test, edits the grader, or special-cases a checked input (HUD's QA list) | Keep the grader out of the agent's reach; audit traces |
| Symmetric costs | Accuracy rewards "always escalate" in triage | A cost matrix |
| Weights buying back a violation | High outcome score offsets a privacy breach | A gate |
| Judge as ground truth | An LLM judge approves fluent but wrong answers | Prefer state checks; calibrate the judge |
| Simulator not matching reality | The mock database is too forgiving, so good scores do not transfer | Compare simulated and real traces where possible; keep the simulator's limits in the record |
| Underspecified spec | A correct agent scores badly | Review the spec; run known-good trajectories |
| Noisy simulated user | The same agent passes one run and fails the next | Repeats, `pass^k`, and seeded variants |

## What this means for the harness

- **The scorer is the reward function in its evaluation form.** `Score` already carries what a combined reward needs: `value`, `passed` (for a gate), `label`, `explanation` and `metadata` (for the named parts). Multiple scorers give a vector.
- **State-based and side-effect rewards need the initial state as well as the final one.** The [simulation plan](plan/simulation.md) lists `EpisodeResult(initial_state, final_state, trajectory, termination, steps)`, so a scorer can compute a diff.
- **A per-step shaped reward belongs in the world's `step`**, not in the harness, and is out of scope here.
- **Terminated and truncated stay distinct**, so a scorer can treat "reached the goal" and "ran out of steps" differently.
- **The reward does not change the Runner or the Harness**: it is a scorer over the episode result, as in the plan.

## Open questions

- **Where gates live:** as a field on `Score` (a `vetoed` flag), or only via `passed` plus `label`.
- **Reward helpers:** whether a small shared module for `combine`, state `diff` and `pass^k` belongs in the simulation layer or in each example.
- **Delayed outcomes:** how a world should model a consequence that arrives after the episode (a later chargeback), if at all in the first slice.
- **Judge validation:** the existing stage-3 item "LLM-judge pinning and validation against human labels" covers the judged part.

## Sources

- tau-bench reward and `pass^k`: [paper](https://arxiv.org/html/2406.12045), [tau2 in inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/tau2/index.html)
- ToolSandbox milestones: [repo](https://github.com/apple/ToolSandbox), [paper](https://arxiv.org/abs/2408.04682)
- AppWorld state-based tests: [repo](https://github.com/StonyBrookNLP/appworld)
- HUD graders and QA: [ops cookbook](https://docs.hud.ai/v6/cookbooks/ops-diagnostics.md), [environment guide](https://docs.hud.ai/v6/guides/creating-an-environment.md), [QA agents](https://docs.hud.ai/platform/qa-agents.md), [evaluations](https://docs.hud.ai/platform/evaluations.md)
- Gymnasium step and seeding: [Env API](https://gymnasium.farama.org/api/env/)
- Potential-based reward shaping and its conditions: Ng, Harada and Russell (1999), "Policy invariance under reward transformations" (cited from memory; not re-read for this note)
- MLE-bench seeds and grading: [repo](https://github.com/openai/mle-bench)
- MLflow scorers: [conversation simulation](https://www.mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/conversation-simulation)
