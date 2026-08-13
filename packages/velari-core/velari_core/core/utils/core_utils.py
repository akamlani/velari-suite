from typing import Any


def unpack_obj_attr(x: Any, key: str, default: Any = None) -> Any:
    """Read a named field from `x`, whether it's a dict or an attribute-based object.

    Args:
        x (Any): A dict (looked up by key) or an object (looked up by attribute).
        key (str): The field name — a dict key or attribute name.
        default (Any): Returned when the key/attribute is missing; defaults to `None`.

    Returns:
        Any: The field's value, or `default` if not present.

    Examples:
        >>> unpack_obj_attr({"name": "churn-2026", "rows": 15000}, "rows")
        15000
        >>> unpack_obj_attr(DatasetSpecInfo(name="churn-2026", rows=15000), "rows")
        15000
    """
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)
