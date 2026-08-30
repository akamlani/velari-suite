# uv run --extra evals python examples/ai/integrations/arize/serve_cli.py
# uv run --extra evals python examples/ai/integrations/arize/serve_cli.py phoenix.connection.port=6007
# See examples/ai/integrations/arize/README.md for details.

import  os
import  signal
import  threading
import  logging
from    pathlib import Path
from    types import FrameType
from    typing import Optional

import  hydra
import  phoenix as px
from    omegaconf import DictConfig
# package modules
from    velari_core.core import read_root_dir

logger = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3",
    config_path=str(Path(read_root_dir()) / "config" / "tracing"),
    config_name="phoenix",
)
def main(cfg: DictConfig) -> None:
    # due to deprecation of launch_app, we manually set the environment variables Phoenix expects
    os.environ["PHOENIX_HOST"]        = str(cfg.phoenix.connection.host)
    os.environ["PHOENIX_PORT"]        = str(cfg.phoenix.connection.port)
    os.environ["PHOENIX_WORKING_DIR"] = str(Path(read_root_dir()) / cfg.phoenix.storage.working_dir)
    # launch Phoenix with the environment variables set
    session = px.launch_app(use_temp_dir=False, run_in_thread=True)
    if session is None:
        raise RuntimeError("Phoenix failed to start — see the log output above for details.")

    stop_event = threading.Event()

    def _shutdown(signum: int, frame: Optional[FrameType]) -> None:
        logger.info("Stopping Phoenix...")
        stop_event.set()

    # SIGINT covers interactive Ctrl+C; SIGTERM covers `kill`/process managers/backgrounded runs —
    # relying on KeyboardInterrupt alone isn't reliable outside an attached interactive terminal.
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"Phoenix UI available at {session.url} — press Ctrl+C to stop.")
    stop_event.wait()
    session.end()


if __name__ == "__main__":
    main()
