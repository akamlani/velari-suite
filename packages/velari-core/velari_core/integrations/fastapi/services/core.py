# Run locally:
#   uv run uvicorn velari_core.integrations.fastapi.services.core:app --reload --port 8000
#   uv run python -m velari_core.integrations.fastapi.services.core
# Example entrypoint (reuses the server instance built below):
#   uv run python examples/core/services/start_core_service.py
# Note: running this file directly by path (e.g. `python .../services/core.py`) does NOT
# work — its relative imports require being loaded as part of the velari_core package,
# via one of the -m/uvicorn/example forms above.

import  os
import  logging
from    contextlib  import asynccontextmanager
from    fastapi     import FastAPI
# package modules
from    ....core         import read_root_dir
from    ..server         import Server
from    ..routers.route  import router as core_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # perform startup tasks here (e.g. connect to DB, initialize resources)
    logger.info("Performing startup tasks...")
    yield
    # perform shutdown tasks here (e.g. close DB connection, clean up resources)
    logger.info("Performing shutdown tasks...")

_server = Server.from_yaml(
    os.path.join(read_root_dir(), "config", "services", "core_services.yaml"),
    lifespan = _lifespan,
)

_server.register_router(core_router)
app = _server.app


if __name__ == "__main__":
    _server.run()
