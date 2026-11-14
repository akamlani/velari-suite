import  hashlib
from    dataclasses  import dataclass, field
from    typing       import Any


@dataclass(frozen=True)
class Environment(object):
    """The conditions of a run that must not vary between runs: for now, the base seed.

    The seed is not applied anywhere by the core. It only produces a separate, repeatable seed for each
    trial (`seed_for`), which the candidate may use. The core never seeds or reads global random state.

    Args:
        seed (int): Base seed every trial seed is derived from.

    Examples:
        Each input gets its own seed, the same one on every run of the same environment:

        >>> environment = Environment(seed=42)
        >>> [environment.seed_for(index) for index in range(3)]
        [247386905, 2345647074, 368107479]
        >>> environment.seed_for(0) == Environment(seed=42).seed_for(0)
        True

        Applied (illustrative): seed a model driver per fold, so a comparison across candidates is repeatable:

        >>> driver = HDBSCANDriver.from_config(cfg, seed=environment.seed_for("fold-3"))
    """
    seed: int = field(default=42)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"Environment.seed must be an int, got {type(self.seed).__name__}")

    def seed_for(self, key: Any, trial: int = 0) -> int:
        """Derive the seed for one trial from the base seed, a key for the input, and the trial number.

        The same arguments always give the same seed, in any process; different keys or trials give
        different seeds. The result fits in 32 bits, so it works with NumPy and scikit-learn.

        Args:
            key (Any): Identifies the input: its position today, its case id once cases exist.
            trial (int): Which repeat of the input this is.

        Returns:
            int: A seed in `[0, 2**32)`.
        """
        digest = hashlib.sha256(f"{self.seed}:{key}:{trial}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % 2**32
