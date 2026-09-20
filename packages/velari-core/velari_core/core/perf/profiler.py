import functools
import inspect
import multiprocessing
import os
import platform
import shutil
import time
from    typing import Any, Callable, Optional
# specific modules
import psutil


def profile_memory() -> dict:
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    ram = psutil.virtual_memory()

    return dict(
        memory=dict(
            total_ram_gb=ram.total / (1024**3),
            available_ram_gb=ram.available / (1024**3),
            used_ram_gb=ram.used / (1024**3),
        ),
        disk_space=dict(
            total_disk_gb=total / (1024**3), free_disk_gb=free / (1024**3), used_disk_gb=(total - free) / (1024**3)
        ),
    )


def profile_system(gb_unit: bool = True) -> dict:
    # convert to either GB or MB
    scale = (1024.0**3) if gb_unit else (1024.0**2)
    memory = psutil.virtual_memory()
    return dict(
        platform=dict(
            system=platform.system(),  # e.g., 'Linux'
            cpu_processor=platform.processor(),  # e.g., 'x86_64'
            architecture=list(platform.architecture()),  # e.g., ('64bit', 'ELF')
            num_cores=multiprocessing.cpu_count(),
        ),
        **profile_memory(),
    )


def profile_time(
    attr: str = "latency_sec", target: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure wall-clock time around a call and set it on the returned/yielded object's `attr`.

    Dispatches once at decoration time over sync functions, async functions, sync generators,
    and async generators, so one decorator covers a client's `fetch()`/`afetch()`/`stream()`/
    `astream()` without duplicating timing boilerplate in each. The wrapped function's return
    value(s) must be mutable objects exposing `attr` as a plain, settable attribute — for a
    generator, `attr` is refreshed on every yield to the elapsed time so far.

    Args:
        attr (str): Attribute name to set — defaults to `"latency_sec"`.
        target (Optional[Callable[[Any], Any]]): Given the returned/yielded object, returns the
            object `attr` should be set on. Defaults to the returned/yielded object itself; pass
            e.g. `target=lambda result: result.metrics.perf` when the timed field lives on a
            nested object instead of the top-level result.

    Examples:
        >>> @dataclass
        ... class QueryResult:
        ...     @dataclass
        ...     class Metrics:
        ...         latency_sec: float = 0.0
        ...     rows: List[dict]
        ...     metrics: Metrics = field(default_factory=Metrics)
        >>> class WarehouseClient:
        ...     @profile_time(target=lambda result: result.metrics)
        ...     def query(self, sql: str) -> QueryResult:
        ...         return QueryResult(rows=self._execute(sql))
        >>> WarehouseClient().query("select * from orders limit 10").metrics.latency_sec
        0.0421
    """
    # Each branch below does the same thing on its own function shape: start a clock, call
    # through to `func`, then write the elapsed seconds onto the result. `setattr(target(item)
    # if target else item, attr, elapsed)` is just `<target object>.<attr> = elapsed` — `target`
    # lets the caller point at a nested object (no string-path parsing involved) when `attr`
    # doesn't live directly on the returned/yielded object.
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _write(item: Any, elapsed: float) -> None:
            setattr(target(item) if target else item, attr, round(elapsed, 3))

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                async for item in func(*args, **kwargs):
                    _write(item, time.perf_counter() - start)
                    yield item
            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                for item in func(*args, **kwargs):
                    _write(item, time.perf_counter() - start)
                    yield item
            return gen_wrapper

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start  = time.perf_counter()
                result = await func(*args, **kwargs)
                _write(result, time.perf_counter() - start)
                return result
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start  = time.perf_counter()
            result = func(*args, **kwargs)
            _write(result, time.perf_counter() - start)
            return result
        return wrapper

    return decorator
