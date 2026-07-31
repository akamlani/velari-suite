import  numpy as np
import  datetime
import  logging
import  random
from    pathlib import Path
from    typing import List
from    appdirs import user_cache_dir
from    omegaconf import DictConfig, OmegaConf

from    .io.filesystem import get_username
from    .io.partition.hydra import read_hydra
from    .utils.env_utils import read_env, read_root_dir

logger = logging.getLogger(__name__)


def resolve_tuple(*args):
    """Convert positional arguments into a tuple.

    Intended for registration as an OmegaConf resolver under the key ``as_tuple``,
    enabling tuple literals in YAML configuration files via ``${as_tuple:a,b,c}``.

    Args:
        *args: Arbitrary values to pack into a tuple.

    Returns:
        A tuple containing all provided arguments in order.

    Examples:
        >>> resolve_tuple(1, 2, 3)
        (1, 2, 3)
        >>> OmegaConf.register_new_resolver("as_tuple", resolve_tuple)
    """
    return tuple(args)


class ConfigOmegaConf(object):
    """Manage OmegaConf global resolver registrations."""

    def resolve(self):
        """Register custom OmegaConf resolvers required by the project configuration.

        Registers the ``as_tuple`` resolver only if it has not already been registered,
        preventing duplicate-registration errors on repeated calls or hot-reloads.

        Examples:
            >>> conf = ConfigOmegaConf()
            >>> conf.resolve()

            Register resolvers before composing a config that uses ``as_tuple``:

            >>> from omegaconf import OmegaConf
            >>> conf = ConfigOmegaConf()
            >>> conf.resolve()  # idempotent — safe to call on every startup
            >>> cfg = OmegaConf.create({"dims": "${as_tuple:128,256,512}"})
            >>> OmegaConf.to_container(cfg, resolve=True)
        """
        if not OmegaConf.has_resolver("as_tuple"):
            OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


class Experiment(object):
    """Encapsulate shared experiment state including seed, directory layout, and environment.

    Provides a consistent initialisation point for reproducible machine-learning
    experiments: a fixed random seed, standard directory paths derived from the
    project root, and loaded environment variables.

    Examples:
        >>> exp = Experiment()

        Full bootstrap workflow — initialise, then create and inspect an experiment:

        >>> exp = Experiment()
        >>> cfg = exp.create("bert-finetune", tags=["nlp", "classification", "v1"])
        >>> print(cfg.experiment.install.dir)  # resolved output directory
    """

    def __init__(self, root_path: str | Path = None):
        """Initialise the experiment with a deterministic seed and resolved directory paths.

        Args:
            root_path: Override for the project root directory. When None, the root is
                discovered automatically by traversing parent directories for
                ``pyproject.toml``.

        Examples:
            >>> exp = Experiment()  # auto-discovers root via pyproject.toml

            Override the root when running from a non-standard working directory:

            >>> from pathlib import Path
            >>> exp = Experiment(root_path=Path.home() / "projects" / "aistudio")
            >>> print(exp.data_dir)   # Path(...)/projects/aistudio/data
            >>> print(exp.username)   # current OS username
        """
        # if passing the seed to random_state, it will produce the same results, same RNG is used across all calls to fit
        self.seed = self.seed_init()
        # estimators that share the same RandomState instance will influence each other
        self.rng = np.random.RandomState(self.seed)

        # directories
        self.root_dir = Path(read_root_dir()) if not root_path else Path(root_path)
        self.template_dir = self.root_dir.joinpath("templates")
        self.apps_dir = self.root_dir.joinpath("apps")
        self.conf_dir = self.root_dir.joinpath("config")
        self.data_dir = self.root_dir.joinpath("data")
        self.catalog_dir = self.data_dir.joinpath("catalog")
        self.date_timestamp = f"{datetime.date.today().strftime('%m%d%Y')}"

        self.cache_dir = user_cache_dir()
        # user inforamtion
        self.username = get_username()
        # load environment
        self.env: dict = read_env(path=str(self.root_dir.joinpath(".env")))

    def seed_init(self, seed: int = 42) -> int:
        """Seed Python's ``random`` module and NumPy's global RNG for reproducibility.

        Args:
            seed: Integer seed value to apply globally. Defaults to 42.

        Returns:
            The seed value that was applied, for use with ``np.random.RandomState``.

        Examples:
            >>> exp = Experiment()
            >>> exp.seed_init(0)   # reseed to 0 before a new training fold

            Re-seed mid-workflow to reset randomness for a new experimental run:

            >>> import numpy as np
            >>> exp = Experiment()
            >>> exp.seed_init(seed=2024)
            >>> sample = np.random.randint(0, 100, size=5)  # reproducible draw
        """
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def create(self, experiment_name: str, tags: List[str], **kwargs) -> DictConfig:
        """Create a new experiment configuration and its output directory on disk.

        Loads the base experiment YAML template via OmegaConf, injects the provided
        name, tags, and current author, resolves the install path relative to the
        project root, then creates the project directory.

        Args:
            experiment_name: Human-readable name for the experiment; written into
                both ``experiment.name`` and ``experiment.project.name``.
            tags: String labels attached to ``experiment.tags`` and
                ``experiment.project.tags``.
            **kwargs: Additional keyword arguments reserved for future extension.

        Returns:
            A DictConfig populated with the experiment metadata, resolved paths, and
            a guaranteed-to-exist project output directory.

        Examples:
            >>> exp = Experiment()
            >>> cfg = exp.create("bert-finetune", tags=["nlp", "v1"])

            Create an experiment then use the resolved config to set up MLflow tracking:

            >>> import mlflow
            >>> exp = Experiment()
            >>> cfg = exp.create("gpt2-summarise", tags=["nlp", "summarisation", "v2"])
            >>> mlflow.set_experiment(cfg.experiment.name)
            >>> print(cfg.experiment.install.dir)  # log artefacts here
        """
        clone: DictConfig = read_hydra(self.conf_dir.joinpath("experimentation", "experiment.yaml"))
        clone.experiment.name = experiment_name
        clone.experiment.tags = tags
        clone.experiment.install.author = self.username
        clone.experiment.project.name = experiment_name
        clone.experiment.project.tags = tags
        clone.experiment.install.dir = str(self.root_dir.joinpath(clone.experiment.install.dir))

        # create directories
        Path(clone.experiment.project.path).mkdir(parents=True, exist_ok=True)
        return clone
