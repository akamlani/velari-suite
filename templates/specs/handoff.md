# Session Handoff Template

Reusable template for handing work from one agent session to the next — across a context reset or compaction, a switch of harness (Claude Code, Codex, Cursor, Gemini CLI, …) or model, or a handoff between agents or people.

- **Goal**: the next session can pick up exactly where this one stopped without re-deriving facts, re-litigating decisions, or retrying approaches that already failed.
- **Harness-agnostic**: plain Markdown, no harness-specific commands, paths, or syntax — any agent that can read a file can resume from it.

**How to use**

- Save a copy per session in a harness-neutral, repo-tracked location (e.g. `docs/handoffs/YYYY-MM-DD-slug.md`) or paste it into the next session's first prompt.
- Don't rely on harness-local state (plan-mode files, todo lists, memory, session/transcript IDs) — it won't exist in another harness, so copy anything that matters into this file.
- Replace every `<placeholder>`; write `None` rather than deleting a section.
- Facts over narrative: `path:line`, exact commands and errors, commit SHAs, absolute dates. No secrets.

---

## Metadata

| Field | Value |
|---|---|
| Date | `<YYYY-MM-DD>` |
| Harness / model | `<harness>` / `<model>` |
| Branch @ commit | `<branch>` @ `<sha or uncommitted>` |
| Status | `<in-progress / blocked / review / done>` |
| Links | `<plan, issue, PR>` |

## Goal

`<one-sentence outcome>`

- [ ] Done when:    `<checkable criterion>`
- [ ] Out of scope: `<deferred items>`

## Current State

- **Working:** `<what works, and how it was verified>`
- **Broken:** `<what doesn't, with the symptom>`
- **Checks:** `<test/lint/typecheck command>` → `<e.g. 42 passed, 1 failed>`
- **Uncommitted / external state:** `<unstaged changes, running servers, temp envs to clean up>`

## Active Files

| Path | Role | Notes |
|---|---|---|
| `<path>` | `<edited / to-edit / context>` | `<hot spots as path:line>` |

## Changes Made

- `<change>` — `<path>` — `<why>` — `<verified / unverified>`

## Failed Attempts

| Approach | Result | Lesson |
|---|---|---|
| `<what was tried>` | `<error or outcome>` | `<why it failed / what it rules out>` |

## Next Steps

- [ ] `<first action on resume>` — done when `<verification>`
- [ ] `<next action>` — done when `<verification>`

## Notes

- **Decisions (settled):** `<decision>` — `<rationale>`
- **Open questions / blockers:** `<question>` — owner: `<who>`
- **Read first:** `<project instructions, e.g. AGENTS.md / CLAUDE.md / GEMINI.md>`, `<plan or spec>`, `<key modules>`

## Resume Prompt

```text
Read <path/to/handoff.md> and the files under "Read first". Start with Next Steps #1.
Don't retry anything under "Failed Attempts"; treat "Decisions" as settled.
```
