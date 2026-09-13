# uv run --isolated --with mlflow python examples/ai/integrations/mlflow/serve_cli.py
# uv run --isolated --with mlflow python examples/ai/integrations/mlflow/serve_cli.py mlflow.connection.port=5002
# See examples/ai/integrations/mlflow/README.md for details.

import  os
import  logging
from    pathlib import Path

import  hydra
from    omegaconf import DictConfig
# package modules
from    velari_core.core import read_root_dir
from    velari_ai.integrations.mlflow.connection import ConnectorConfig

logger = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3",
    config_path=str(Path(read_root_dir()) / "config" / "tracing"),
    config_name="mlflow",
)
def main(cfg: DictConfig) -> None:
    storage = ConnectorConfig.Storage(
        working_dir = str(Path(read_root_dir()) / cfg.mlflow.storage.working_dir),
        db_name     = cfg.mlflow.storage.db_name,
    )
    artifacts_dir = Path(read_root_dir()) / cfg.mlflow.storage.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "mlflow", "server",
        "--host", str(cfg.mlflow.connection.host),
        "--port", str(cfg.mlflow.connection.port),
        "--backend-store-uri", storage.db_url,
        "--artifacts-destination", str(artifacts_dir),
    ]
    if cfg.mlflow.security.allowed_hosts:
        args += ["--allowed-hosts", ",".join(cfg.mlflow.security.allowed_hosts)]
    if cfg.mlflow.security.cors_allowed_origins:
        args += ["--cors-allowed-origins", ",".join(cfg.mlflow.security.cors_allowed_origins)]

    logger.info(f"Starting MLflow server at http://{cfg.mlflow.connection.host}:{cfg.mlflow.connection.port}")
    os.execvp("mlflow", args)


if __name__ == "__main__":
    main()
