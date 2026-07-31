import argparse
import logging

logger = logging.getLogger(__name__)

try:
    from IPython.core.getipython import get_ipython

    from rich import print_json
    from rich.pretty import pprint as rpprint
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    ipython = get_ipython()
    if ipython:
        logging.info("Running in Jupyter Notebook environment")
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")


except NameError:
    # get_ipython is not defined if not in a Jupyter environment
    logging.error("This script requires a Jupyter Notebook to execute magic commands.")


if __name__ == "__main__":
    # common imports for this aistudio library
    from omegaconf import DictConfig, OmegaConf
    from velari.core.experiment import Experiment
    from velari.core.utils.env_utils import read_env, read_root_dir

    # parse input arguments
    parser = argparse.ArgumentParser(description="Experiment Parameters")
    parser.add_argument("--expname", "-e", type=str, required=True, help="Experiment Name")
    parser.add_argument("--rootdir", "-r", type=str, required=False, help="Root Directory Path to walk")

    args = parser.parse_args()
    expname = args.expname
    rootdir = args.rootdir
    # execute experiment
    exp = Experiment(root_path=read_root_dir() if not rootdir else rootdir)
    seed = exp.seed_init()
    exp_config = exp.create(experiment_name=expname, tags=expname.split("-")).experiment
