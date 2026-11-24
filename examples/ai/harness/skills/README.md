# Skill-Trigger Example

A runnable example that measures how well a skill's description triggers that skill, using the skills layer of `velari_harness`
(`Skill`, `SkillRegistry`) with the same core as the [core example](../core/README.md): `Task`, `Candidate`, a scorer, `Harness` and `Result`. It is
offline: no model, no network, no extras.

`skills_cli.py` is a second, separate example: it measures how well a skill's description triggers that skill. It loads
the skills under `_build/agent-skills/packages` into a `SkillRegistry` (a table with one row per skill), gives each of 27 requests to a
candidate, and scores the skill it picks against the skill that should have been used.

- **`skills_cli.py`**: the registry, the candidate (`SkillTrigger`, a keyword baseline that weights each word by how rare it is across the skill
  descriptions), the scorer (`SkillScorer.skill_match`) and the printing, in one script shaped like `harness_cli.py`.
- **`examples/_conf/harness/skills/app_skills.yaml`**: the task name and version, the skills folder, and `min_score`, how strong a match must be
  before a skill is triggered instead of answering `none`.
- **`data/skill_requests.csv`**: the examples, `id,request,expected_skill`; 24 requests with the skill that should trigger and 3 where none should. I wrote
  them from the skill descriptions, so the labels are one author's judgment; edit the file to change the run.

## How the skill classes are used

| Class or method | Where in this example | What for |
|---|---|---|
| `SkillRegistry` | built in `SkillsCli.__init__` from the config's folder | one row per skill |
| `registry.to_dataframe()` | `SkillTrigger.__init__` | reads each skill's name and description to build its keyword index |
| `registry.catalog_text()` | `render()` | prints the size of the catalog a model would be given |
| `Skill`, `SkillValidator` | indirectly, when the registry loads each folder | every row is a validated skill |
| `get(name)`, `instructions()`, `resources()`, `list(**filters)` | not used here | what a model-based candidate would call |

The harness core does not know about skills. The registry is built here and handed to the candidate, and the harness only runs and scores
what the candidate returns: skills belong to the candidate side.

With a model instead of the keyword baseline, the catalog text goes into the model's system prompt. When the model picks a skill, the candidate
reads `registry.get(name).instructions()` and gives it back. `list(**filters)` and `catalog_text(names=...)` limit what the model is shown (the
catalog of all 43 of the author's skills is about 7,000 tokens), and a `skill_triggered` event through `TrialContext` (planned) would record what fired.
The design and what was built are in [stage 2, Increment 16](../../../../docs/packages/velari-harness/plan/stage-2-base-framework.md).

## Prerequisites

The skills come from a separate repository that is not part of this one (`SKILLS_REPO` and `SKILLS_DIR` in `config/runtime/runtime.env`).
From the repository root, clone it into `_build/` (gitignored) and install the dependencies:

```
git clone https://github.com/akamlani/agent-skills _build/agent-skills
uv sync
```

To use other skills, point `skills.root` in `examples/_conf/harness/skills/app_skills.yaml` at a folder of skill folders; the output below is for these.

## Run

```
uv run python examples/ai/harness/skills/skills_cli.py
```

Output (shown at 150 columns; it is deterministic):

```
            skills             
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ skills ┃ catalog characters ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ 24     │ 15,448             │
└────────┴────────────────────┘
                    scores                    
┏━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━┳━━━━━━━━━┓
┃ scorer      ┃ mean ┃ stderr ┃ n  ┃ invalid ┃
┡━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━╇━━━━━━━━━┩
│ skill_match │ 0.81 │ 0.08   │ 27 │ 0       │
└─────────────┴──────┴────────┴────┴─────────┘
                          skill-trigger (version 1, data 4bcad121121c): outcome per request                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ request                                                      ┃ expected          ┃ triggered              ┃ score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Draw a sequence diagram of the checkout flow in Mermaid      │ diagramming       │ diagramming            │ hit   │
│ Make a flowchart of our deployment pipeline                  │ diagramming       │ diagramming            │ hit   │
│ Create an architecture spec for the new billing service      │ architecture-spec │ architecture-spec      │ hit   │
│ Write a system design doc for a notification platform        │ architecture-spec │ architecture-spec      │ hit   │
│ Generate an invoice for Acme Corp for 12 hours of consulting │ write-invoice     │ write-invoice          │ hit   │
│ Prepare an invoice PDF for a client from the XLSX template   │ write-invoice     │ write-invoice          │ hit   │
│ Build a slide deck from this markdown outline                │ marp              │ marp                   │ hit   │
│ Render my notes as a Marp presentation                       │ marp              │ marp                   │ hit   │
│ Review my local changes and list the action items            │ review            │ review                 │ hit   │
│ Do a code review of this change before I merge it            │ review            │ review                 │ hit   │
│ Write unit and integration tests for the parser module       │ testing           │ testing                │ hit   │
│ Fix the flaky end-to-end test in the checkout suite          │ testing           │ testing                │ hit   │
│ Profile this CSV and tell me about missing values            │ profiling         │ profiling              │ hit   │
│ Explore the dataset and describe its columns and quality     │ profiling         │ feature-ideation       │ miss  │
│ Create a data dictionary in YAML for this table              │ schema            │ schema                 │ hit   │
│ Generate a synthetic customer dataset with realistic columns │ synthetic-dataset │ synthetic-dataset      │ hit   │
│ Brainstorm recurring team workflows that AI could automate   │ identify          │ identify               │ hit   │
│ Prioritize these workflows using the 3V framework            │ prioritize        │ prioritize             │ hit   │
│ Write docstrings for this module                             │ write-docstring   │ write-docstring        │ hit   │
│ Refactor this function to remove the duplication             │ refactor          │ refactor               │ hit   │
│ Export my Granola meeting transcripts                        │ granola           │ granola                │ hit   │
│ Summarize this article for me                                │ summarize         │ summarize              │ hit   │
│ Explain how this repository is structured                    │ explain           │ architecture-spec      │ miss  │
│ What will AI look like in 2035?                              │ futurism          │ explain                │ miss  │
│ What is the capital of France?                               │ none              │ none                   │ hit   │
│ Thanks that is all for today                                 │ none              │ identify               │ miss  │
│ Tell me a joke about cats                                    │ none              │ strategic-storytelling │ miss  │
└──────────────────────────────────────────────────────────────┴───────────────────┴────────────────────────┴───────┘
skill_triggered events: 27
```

Reading it: the baseline picks the expected skill for 22 of 27 requests (accuracy 0.81, standard error 0.08). Two misses are arguable
(`Explore the dataset...` could reasonably trigger `feature-ideation`, and `Explain how this repository is structured` sits between
`explain` and `architecture-spec`). The clearer failures are the two requests where no skill should trigger: a keyword matcher always finds
some word in common, so `min_score` is a weak way to say "nothing applies". Replacing `SkillTrigger` with a model that reads
`registry.catalog_text()` would change only that class; the registry, task, scorer and harness stay as they are. The catalog printed in the first table is the
text such a model would be given.
The last line counts the `skill_triggered` events `SkillTrigger` emitted through its `TrialContext` (one per request, including the `none` picks); they are kept on each trial's record.
