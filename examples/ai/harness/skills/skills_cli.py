# uv run python examples/ai/harness/skills/skills_cli.py
# Edit examples/_conf/harness/skills/app_skills.yaml (the skills folder and the matching threshold) or data/skill_requests.csv to change the run.
# See examples/ai/harness/skills/README.md for details.

import  logging
import  math
import  pandas as pd
import  re
import  typer
from    collections   import Counter
from    pathlib       import Path
from    typing        import Any, Dict, Optional, Self, Set
from    omegaconf     import DictConfig
from    rich.console  import Console
from    rich.table    import Table
# package modules
from    velari_ai.ai.evals.heuristics import exact_match
from    velari_core.core import read_root_dir
from    velari_core.core.io.partition.hydra import read_hydra
from    velari_harness.core.execution.harness import Harness
from    velari_harness.core.protocols.candidate import Candidate
from    velari_harness.core.schemas.dataset import Dataset, Example, Task
from    velari_harness.core.schemas.response.record import Score
from    velari_harness.core.schemas.response.result import Result
from    velari_harness.core.schemas.skill import SkillRegistry

# Objective: measure how well skill descriptions trigger the right skill. A request goes in, the candidate picks a skill (or "none"),
# and the harness scores the pick against the skill that should have been used.

# Terms used below, each with what it is in this example (full definitions: docs/packages/velari-harness/glossary.md)
#   Skill      a folder with a SKILL.md whose name and description tell an agent when to use it
#              here: the 24 skills under _build/agent-skills/packages, loaded into a SkillRegistry (a table, one row per skill)
#   Candidate  the thing being measured: it takes one input and returns an output
#              here: SkillTrigger, which takes a request and returns the name of the skill whose description matches it best
#   Task       what is measured: a name, a version and a dataset of examples
#              here: "skill-trigger", whose examples are the 27 requests of data/skill_requests.csv, each with the skill it should trigger
#   Scorer     judges one output against the reference (the expected value), without the candidate ever seeing the reference
#              here: SkillScorer.skill_match, which compares the picked skill with the expected_skill column
#   Result     what one harness run returns; its summary gives the accuracy as a mean with its standard error
#              here: the share of requests where the right skill was picked


class SkillTrigger(Candidate):
    """The candidate: a keyword baseline that picks the skill whose name and description best match a request.

    Each word is weighted by how rare it is across the skills (IDF), so "invoice" counts for more than "create". A model that reads
    the skill catalog would replace this class; the harness around it would not change.
    """
    STOPWORDS        = {"the", "and", "for", "with", "this", "that", "from", "when", "use", "into", "your", "their", "are", "can", "how"}
    MIN_TOKEN_LENGTH = 2

    def __init__(self, registry: SkillRegistry, min_score: float) -> None:
        self.min_score = min_score
        df_skills = registry.to_dataframe()
        self.skills: Dict[str, Set[str]] = {
            name: self.tokens(f"{name.replace('-', ' ')} {description}") for name, description in zip(df_skills["name"], df_skills["description"])
        }
        skills_per_token = Counter(token for tokens in self.skills.values() for token in tokens)
        self.idf = {token: math.log(len(self.skills) / count) for token, count in skills_per_token.items()}

    @classmethod
    def tokens(cls, text: str) -> Set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {word.removesuffix("s") for word in words if len(word) > cls.MIN_TOKEN_LENGTH and word not in cls.STOPWORDS}

    def run(self, input: Any) -> Any:
        request = self.tokens(input)
        scores = {name: sum(self.idf[token] for token in request & tokens) for name, tokens in self.skills.items()}
        best = max(scores, key=lambda name: scores[name])
        return best if scores[best] >= self.min_score else "none"


class SkillScorer(object):
    """Scorers for the skill-trigger task; each is a plain function of (output, reference) that only scores."""
    @staticmethod
    def skill_match(output: Any, reference: Any) -> Score:
        # a pick either is the expected skill or it is not
        matched = exact_match(output, reference)
        return Score(value=float(matched), passed=matched, label="hit" if matched else "miss", metadata={"expected": reference})


class SkillsCli(object):
    """Load the skills, run the matcher on every request through a Harness, then print the results. Calls chain: `cli.run().render()`."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # the skills the candidate chooses from: one row per skill, with its name, description and the rest of its frontmatter as columns
        self.registry = SkillRegistry(Path(read_root_dir()) / cfg.skills.root)
        # the Task: each CSV row is a request (the input the candidate sees) and the skill that should trigger (the reference, which it never sees)
        df_requests = pd.read_csv(Path(read_root_dir()) / cfg.task.requests_path, dtype=str, keep_default_na=False)
        examples = [Example(request, reference, uid=uid) for uid, request, reference in zip(df_requests["id"], df_requests["request"], df_requests["expected_skill"])]
        self.task = Task(cfg.task.name, Dataset(Path(cfg.task.requests_path).stem, examples), version=str(cfg.task.version))
        # step 1, create: the harness pairs the candidate and the scorer with the task
        self.harness = Harness.for_task(self.task, SkillTrigger(self.registry, float(cfg.skills.min_score)), scorers=[SkillScorer.skill_match])
        self.result: Optional[Result] = None   # set by run()

    @classmethod
    def from_config(cls, path: Path) -> Self:
        cfg = read_hydra(str(path))
        if cfg is None:
            raise ValueError(f"failed to load config: {path}")
        return cls(cfg)

    def run(self) -> Self:
        # step 2, run: the result holds one trial record per request
        self.result = self.harness.run()
        return self

    def render(self) -> Self:
        # step 3, render: print what the registry holds and what run() produced
        if self.result is None:
            raise RuntimeError("call run() before render()")
        result = self.result

        # table 1: the skills: how many were loaded and how large the catalog is that a model would be given
        skills = Table(title="skills")
        for column in ("skills", "catalog characters"):
            skills.add_column(column)
        skills.add_row(str(len(self.registry)), f"{len(self.registry.catalog_text()):,}")

        # table 2: the scorer's summary over the run: the accuracy (mean), its standard error, and counts of the scores
        summary = Table(title="scores")
        for column in ("scorer", "mean", "stderr", "n", "invalid"):
            summary.add_column(column)
        for name, stats in result.summary.items():
            summary.add_row(name, f"{stats.mean:.2f}" if stats.mean is not None else "",
                            f"{stats.stderr:.2f}" if stats.stderr is not None else "", str(stats.n), str(stats.invalid))

        # table 3: one row per request: the skill that should trigger, the one that did, and whether they match
        requests = Table(title=f"{self.task.name} (version {self.task.version}, data {self.task.dataset.fingerprint}): outcome per request")
        for column in ("request", "expected", "triggered", "score"):
            requests.add_column(column)
        for record in result.records:
            output = record.outcome.output
            score = next(iter(record.scores.values()))
            color = "green" if score.passed else "red"
            requests.add_row(str(record.example.input), str(record.example.reference), f"[{color}]{output.value if output else ''}[/{color}]", str(score.label))

        console = Console()
        console.print(skills)
        console.print(summary)
        console.print(requests)
        return self


if __name__ == "__main__":
    # the Runner logs every failed trial at ERROR level; no trial fails here, but keep the output to the tables
    logging.getLogger("velari_harness").setLevel(logging.CRITICAL)
    # build the command-line app: one command, no options; its name and help text come from the config
    path = Path(read_root_dir()) / "examples" / "_conf" / "harness" / "skills" / "app_skills.yaml"
    cli = SkillsCli.from_config(path)
    app = typer.Typer(rich_markup_mode="rich", add_completion=False)
    app.info.name = cli.cfg.app.info.name
    app.info.help = cli.cfg.app.info.help

    @app.command(help="Measure how well a skill's description triggers it.")
    def run() -> None:
        cli.run().render()

    app()
