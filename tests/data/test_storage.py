"""Tests for velari_data.storage."""

import pytest
from dataclasses import dataclass


class TestLocalKeyValueStore:
    def test_put_get(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore()
        store.put("a", 1)
        assert store.get("a") == 1
        assert store.get("missing") is None

    def test_initial_data_has_and_len(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore(data={"a": 1, "b": 2})
        assert store.has("a")
        assert not store.has("z")
        assert len(store) == 2

    def test_delete_noop_when_missing(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore(data={"a": 1})
        store.delete("a")
        assert not store.has("a")
        store.delete("missing")

    def test_mapping_protocol(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore()
        store["a"] = 1
        assert store["a"] == 1
        assert "a" in store
        assert "z" not in store
        del store["a"]
        assert "a" not in store

    def test_getitem_missing_raises_keyerror(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore()
        with pytest.raises(KeyError):
            store["missing"]

    def test_delitem_missing_raises_keyerror(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore()
        with pytest.raises(KeyError):
            del store["missing"]

    def test_filter(self):
        from velari_data.storage import LocalKeyValueStore

        store = LocalKeyValueStore(data={"a": 1, "b": 2, "c": 3})
        evens = store.filter(lambda _, v: v % 2 == 0)
        assert evens == {"b": 2}


class TestLocalDocumentStore:
    def test_upsert_and_get_dict(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}])
        assert store.get(["a"]) == [{"id": "a", "x": 1}]

    def test_upsert_without_id_generates_one(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"x": 1}])
        assert len(store) == 1
        assert store.ids()[0]

    def test_upsert_pydantic_model(self):
        from pydantic import BaseModel
        from velari_data.storage import LocalDocumentStore

        class Doc(BaseModel):
            id: str
            x: int

        store = LocalDocumentStore()
        store.upsert([Doc(id="a", x=1)])
        assert store.get(["a"]) == [{"id": "a", "x": 1}]

    def test_upsert_dataclass(self):
        from velari_data.storage import LocalDocumentStore

        @dataclass
        class Doc:
            id: str
            x: int

        store = LocalDocumentStore()
        store.upsert([Doc(id="a", x=1)])
        assert store.get(["a"]) == [{"id": "a", "x": 1}]

    def test_upsert_unsupported_type_raises_typeerror(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        with pytest.raises(TypeError):
            store.upsert([42])

    def test_get_with_schema(self):
        from pydantic import BaseModel
        from velari_data.storage import LocalDocumentStore

        class Doc(BaseModel):
            id: str
            x: int

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}])
        result = store.get(["a"], schema=Doc)
        assert result == [Doc(id="a", x=1)]

    def test_get_filters_missing_ids(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}])
        assert store.get(["a", "missing"]) == [{"id": "a", "x": 1}]

    def test_delete_and_has(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}])
        assert store.has("a")
        store.delete(["a"])
        assert not store.has("a")

    def test_contains(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}])
        assert "a" in store
        assert "z" not in store

    def test_ids_and_len(self):
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}, {"id": "b", "x": 2}])
        assert set(store.ids()) == {"a", "b"}
        assert len(store) == 2

    def test_to_frame(self):
        import pandas as pd
        from velari_data.storage import LocalDocumentStore

        store = LocalDocumentStore()
        store.upsert([{"id": "a", "x": 1}, {"id": "b", "x": 2}])
        df = store.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df["x"]) == {1, 2}
