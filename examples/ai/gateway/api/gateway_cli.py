# uv run --extra gateway python examples/ai/gateway/api/gateway_cli.py --help
# uv run --extra gateway python examples/ai/gateway/api/gateway_cli.py embed "retrieval-augmented generation" "vector similarity search"
# uv run --extra gateway python examples/ai/gateway/api/gateway_cli.py chat "What is retrieval-augmented generation?"
# Provider/model for both commands come from examples/_conf/gateway/gateway.yaml, not the CLI.
# chat requires OPENAI_API_KEY to be set in the environment.

from __future__ import annotations

import  logging
import  os
import  typer
from    pathlib             import Path
from    typing              import Any, List
from    omegaconf           import DictConfig, OmegaConf
from    rich.pretty         import pprint
from    transformers.utils  import logging as hf_logging
# package modules
from    velari_core.core.io.partition.hydra           import read_hydra_compose
from    velari_core.core                              import read_root_dir
from    velari_core.core.experiment                   import Experiment
from    velari_ai.ai.types                            import ModelConfig
from    velari_ai.infra.gateway.llm.gateway           import LLMGateway
from    velari_ai.infra.gateway.llm.schemas.types     import GatewayMessage, Role
from    velari_ai.infra.gateway.llm.schemas.requests  import GatewayRequest
from    velari_ai.infra.gateway.llm.schemas.response  import GatewayEmbeddingResponse, GatewayResponse

logger  = logging.getLogger(__name__)
app     = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _silence_local_model_logging() -> None:
    """Suppress huggingface/sentence-transformers logs, the HTTP request lines, and progress bars."""
    os.environ.update({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"})
    hf_logging.disable_progress_bar()
    for name in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)


_silence_local_model_logging()


def _read_gateway_config() -> DictConfig:
    """Load examples/_conf/gateway/gateway.yaml via Hydra."""
    config_dir = str(Path(read_root_dir()) / "examples" / "_conf" / "gateway")
    cfg, _     = read_hydra_compose(config_dir, "gateway.yaml")
    return cfg


def _read_model_config(section: DictConfig, **parameters: Any) -> ModelConfig:
    """Build a `ModelConfig` from a gateway.yaml section via `ConfigBase.from_config`; `parameters` kwargs are unioned over the section's own."""
    container = OmegaConf.to_container(section, resolve=True)
    entry     = {str(k): v for k, v in container.items()} if isinstance(container, dict) else {}
    return ModelConfig.from_config({**entry, "parameters": entry.get("parameters", {}) | parameters})


@app.command()
def embed(texts: List[str] = typer.Argument(..., help="Texts to embed.")) -> None:
    """Embed texts through the gateway's stf provider, configured via Hydra."""
    cfg          = _read_gateway_config()
    model_config = _read_model_config(cfg.embedding)
    gateway      = LLMGateway()
    result: GatewayEmbeddingResponse = gateway.embed(texts, model_config=model_config)
    pprint(result.model_dump(), max_length=3)  # max_length truncates each 768-float vector


@app.command()
def chat(message: str = typer.Argument(..., help="User message to send to the model.")) -> None:
    """Send a synchronous chat message through the gateway's OpenAI provider, configured via Hydra."""
    cfg          = _read_gateway_config()
    experiment   = Experiment(root_path=read_root_dir())
    model_config = _read_model_config(cfg.chat, seed=experiment.seed)
    gateway      = LLMGateway()
    request      = GatewayRequest(messages=[GatewayMessage(role=Role.USER, content=message)], parameters=model_config.parameters)
    result: GatewayResponse = gateway.chat(request, model_config=model_config)
    pprint(result.model_dump(exclude={"response": {"raw"}}))  # raw is the verbose SDK payload


if __name__ == "__main__":
    app()
