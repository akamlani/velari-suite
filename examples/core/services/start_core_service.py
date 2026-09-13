# uv run python examples/core/services/start_core_service.py

from velari_core.integrations.fastapi.services.core import _server

if __name__ == "__main__":
    # reuses the server instance core.py already built from config/services/core_services.yaml —
    # host/port come from that config, not hardcoded here.
    _server.run()
