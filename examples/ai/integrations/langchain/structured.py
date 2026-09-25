# Structured output for customer support: free-text message -> typed Ticket record.
#
#   uv run --extra agents python examples/ai/integrations/langchain/structured.py analyze "My order #48213 arrived damaged, this is the second time!"
#   uv run --extra agents python examples/ai/integrations/langchain/structured.py analyze --json "I was charged twice for my subscription."
#   uv run --extra agents python examples/ai/integrations/langchain/structured.py batch --file tickets.txt   # one ticket per line
#
# Requires OPENAI_API_KEY in the repo-root .env (default model: openai:gpt-4o-mini).
from __future__ import annotations

import  logging
import  typer
from    pathlib import Path
from    typing import Any

from    rich.console import Console
from    rich.pretty import pprint
from    rich.table import Table

# package modules
from    velari_core.core import read_root_dir, read_env
from    velari_ai.ai.types import ModelConfig
from    velari_ai.ai.schemas.applied.support import Ticket
from    velari_ai.integrations.langchain.models.llm import LLM


logger  = logging.getLogger(__name__)
console = Console()
app     = typer.Typer(rich_markup_mode="rich", add_completion=False)

DEFAULT_MODEL = "openai:gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a customer support assistant. Read the customer's message and fill in every field of the ticket."
)


def _require_ticket(response: Any) -> Ticket:
    if not isinstance(response, Ticket):
        raise TypeError(f"Expected Ticket, got {type(response).__name__}")
    return response


def _build_llm(model: str) -> LLM:
    read_env(str(Path(read_root_dir()) / ".env"))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return LLM(model_config=ModelConfig(model=model))

# JEV: System 1 Model (typesafe.ai)


@app.command()
def analyze(
    message: str  = typer.Argument(..., help="Customer support ticket text."),
    model: str    = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Chat model as provider:model."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw JSON record instead of a pretty view."),
) -> None:
    """Extract a structured Ticket from a single customer message."""
    try:
        result = _build_llm(model).run(SYSTEM_PROMPT, message, response_model=Ticket)
        ticket = _require_ticket(result.response)
    except Exception as exc:
        typer.echo(f"Structured output failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(ticket.model_dump_json(indent=2))
    else:
        pprint(ticket.model_dump(mode="json"))
    logger.info(f"Extracted ticket in {result.metrics.latency_sec}s")


@app.command()
def batch(
    file: Path = typer.Option(..., "--file", "-f", exists=True, dir_okay=False, help="Text file with one ticket per line."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Chat model as provider:model."),
) -> None:
    """Extract a Ticket from every message in a file in parallel and print a summary table."""
    messages = [line.strip() for line in file.read_text().splitlines() if line.strip()]
    try:
        results = _build_llm(model).batch(SYSTEM_PROMPT, messages, response_model=Ticket)
        tickets = [_require_ticket(r.response) for r in results]
    except Exception as exc:
        typer.echo(f"Structured output failed: {exc}", err=True)
        raise typer.Exit(code=1)

    table = Table(title=f"Tickets ({len(tickets)})")
    for column in ("category", "priority", "sentiment", "summary"):
        table.add_column(column)
    for t in tickets:
        table.add_row(t.category, t.priority, t.sentiment, t.summary)
    console.print(table)


if __name__ == "__main__":
    app()
