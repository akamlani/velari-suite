"""Tests for velari_core.core.perf.profiler.profile_time."""
import asyncio
import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


@dataclass
class _Result:
    latency_sec: float = 0.0


@dataclass
class _Metrics:
    latency_sec: float = 0.0


@dataclass
class _NestedResult:
    metrics: _Metrics = field(default_factory=_Metrics)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    from velari_core.core.perf import profiler
    fake = _FakeClock()
    monkeypatch.setattr(profiler, "time", SimpleNamespace(perf_counter=fake.perf_counter))
    return fake


async def _collect(async_iterator):
    return [item async for item in async_iterator]


class TestSync:
    def test_sets_elapsed_on_returned_object(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch():
            clock.advance(0.25)
            return _Result()

        assert fetch().latency_sec == 0.25

    def test_returns_same_object_func_returned(self):
        from velari_core.core.perf.profiler import profile_time
        result = _Result()
        assert profile_time()(lambda: result)() is result

    def test_forwards_args_and_kwargs(self):
        from velari_core.core.perf.profiler import profile_time
        seen = {}

        @profile_time()
        def fetch(a, b, *, scale):
            seen.update(a=a, b=b, scale=scale)
            return _Result()

        fetch(1, 2, scale=3)
        assert seen == {"a": 1, "b": 2, "scale": 3}

    def test_rounds_to_three_decimals(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch():
            clock.advance(0.123456)
            return _Result()

        assert fetch().latency_sec == 0.123

    def test_custom_attr(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time(attr="elapsed")
        def fetch():
            clock.advance(0.5)
            return SimpleNamespace()

        assert fetch().elapsed == 0.5

    def test_target_points_at_nested_object(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time(target=lambda result: result.metrics)
        def fetch():
            clock.advance(0.4)
            return _NestedResult()

        assert fetch().metrics.latency_sec == 0.4

    def test_preserves_function_metadata(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch():
            """Fetch docs."""
            return _Result()

        assert fetch.__name__ == "fetch"
        assert fetch.__doc__ == "Fetch docs."

    def test_exception_propagates(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fetch()

    def test_unsettable_attr_raises_attributeerror(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch():
            return 42

        with pytest.raises(AttributeError):
            fetch()

    def test_batch_of_calls_each_timed_independently(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def fetch(seconds):
            clock.advance(seconds)
            return _Result()

        durations = [0.1, 0.5, 0.25]
        assert [fetch(d).latency_sec for d in durations] == durations

    def test_method_on_instance(self, clock):
        from velari_core.core.perf.profiler import profile_time

        class Client(object):
            @profile_time(target=lambda result: result.metrics)
            def query(self, seconds):
                clock.advance(seconds)
                return _NestedResult()

        assert Client().query(0.75).metrics.latency_sec == 0.75


class TestAsync:
    def test_sets_elapsed_on_awaited_result(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def afetch():
            clock.advance(0.25)
            return _Result()

        assert asyncio.run(afetch()).latency_sec == 0.25

    def test_target_points_at_nested_object(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time(target=lambda result: result.metrics)
        async def afetch():
            clock.advance(0.4)
            return _NestedResult()

        assert asyncio.run(afetch()).metrics.latency_sec == 0.4

    def test_wrapper_is_still_coroutine_function(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def afetch():
            return _Result()

        assert inspect.iscoroutinefunction(afetch)

    def test_exception_propagates(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def afetch():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            asyncio.run(afetch())

    def test_batch_gather_each_timed_independently(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def afetch(seconds):
            await asyncio.sleep(0)
            clock.advance(seconds)
            return _Result()

        async def run():
            return await asyncio.gather(afetch(0.1), afetch(0.5))

        # the fake clock is shared, so the overlapping call also absorbs the other's advance:
        # call 1 starts at 0, call 2 starts at 0, call 1 advances to 0.1 (elapsed 0.1),
        # call 2 advances to 0.6 (elapsed 0.6)
        assert [r.latency_sec for r in asyncio.run(run())] == [0.1, 0.6]


class TestGenerator:
    def test_refreshes_elapsed_on_every_yield(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def stream():
            for _ in range(3):
                clock.advance(0.1)
                yield _Result()

        assert [item.latency_sec for item in stream()] == [0.1, 0.2, 0.3]

    def test_yields_items_in_order_unchanged(self):
        from velari_core.core.perf.profiler import profile_time
        items = [_Result(), _Result()]

        @profile_time()
        def stream():
            yield from items

        yielded = list(stream())
        assert len(yielded) == len(items)
        assert all(a is b for a, b in zip(yielded, items))

    def test_target_points_at_nested_object(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time(target=lambda result: result.metrics)
        def stream():
            clock.advance(0.2)
            yield _NestedResult()

        assert next(stream()).metrics.latency_sec == 0.2

    def test_empty_generator_yields_nothing(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def stream():
            yield from ()

        assert list(stream()) == []

    def test_wrapper_is_still_generator_function(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def stream():
            yield _Result()

        assert inspect.isgeneratorfunction(stream)

    def test_clock_starts_on_first_next_not_at_call(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        def stream():
            clock.advance(0.1)
            yield _Result()

        gen = stream()
        clock.advance(5.0)
        assert next(gen).latency_sec == 0.1


class TestAsyncGenerator:
    def test_refreshes_elapsed_on_every_yield(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def astream():
            for _ in range(3):
                clock.advance(0.1)
                yield _Result()

        items = asyncio.run(_collect(astream()))
        assert [item.latency_sec for item in items] == [0.1, 0.2, 0.3]

    def test_target_points_at_nested_object(self, clock):
        from velari_core.core.perf.profiler import profile_time

        @profile_time(target=lambda result: result.metrics)
        async def astream():
            clock.advance(0.2)
            yield _NestedResult()

        items = asyncio.run(_collect(astream()))
        assert items[0].metrics.latency_sec == 0.2

    def test_wrapper_is_still_async_generator_function(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def astream():
            yield _Result()

        assert inspect.isasyncgenfunction(astream)

    def test_early_aclose_closes_inner_generator_immediately(self):
        from velari_core.core.perf.profiler import profile_time
        closed = []

        @profile_time()
        async def astream():
            try:
                while True:
                    yield _Result()
            finally:
                closed.append(True)

        async def run():
            stream = astream()
            await anext(stream)
            await stream.aclose()
            return list(closed)

        assert asyncio.run(run()) == [True]

    def test_empty_async_generator_yields_nothing(self):
        from velari_core.core.perf.profiler import profile_time

        @profile_time()
        async def astream():
            for item in ():
                yield item

        assert asyncio.run(_collect(astream())) == []
