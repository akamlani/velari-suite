# Harness Examples

Two runnable examples for `velari_harness`, each in its own folder with its own data and README.

- **[`core/`](core/README.md)**: the core only. A `Task`, a rule-based `Candidate`, a scorer function and a `Harness`, validating the rows of a customer CSV file.
  `uv run python examples/ai/harness/core/harness_cli.py`
- **[`skills/`](skills/README.md)**: the skills layer. A `SkillRegistry` of the repository's skills and a keyword candidate, scored on how often it triggers the expected skill.
  `uv run python examples/ai/harness/skills/skills_cli.py`

Both run offline from the repository root after `uv sync`. Their configs are in `examples/_conf/harness/core/` and `examples/_conf/harness/skills/`.
