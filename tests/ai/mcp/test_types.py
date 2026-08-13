"""Tests for velari_ai.integrations.fastmcp.types."""

import functools

from omegaconf import OmegaConf


class TestToolDoc:
    def test_from_function_parses_summary_and_raises(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        def get_time() -> str:
            """Return the current time.

            Raises:
                RuntimeError: if the system clock is unavailable.
            """
            raise NotImplementedError

        doc = ToolDoc.from_function(get_time)

        assert doc.summary == "Return the current time."
        assert doc.parameters == []
        assert doc.returns is None
        assert doc.raises == [ToolDoc.Raise(exception="RuntimeError", description="if the system clock is unavailable.")]

    def test_from_function_parses_parameters_returns_raises_examples(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        def square(x: int) -> int:
            """Square a number.

            Args:
                x (int): The number to square.

            Returns:
                int: The squared value.

            Raises:
                ValueError: if x is negative.

            Examples:
                >>> square(3)
                9
            """
            raise NotImplementedError

        doc = ToolDoc.from_function(square)

        assert doc.summary == "Square a number."
        assert doc.parameters == [ToolDoc.Parameter(name="x", annotation="int", description="The number to square.")]
        assert doc.returns == ToolDoc.Returns(annotation="int", description="The squared value.")
        assert doc.raises == [ToolDoc.Raise(exception="ValueError", description="if x is negative.")]
        assert doc.examples == [">>> square(3)\n9"]

    def test_from_function_no_docstring_returns_empty(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        def no_doc(x):
            pass

        doc = ToolDoc.from_function(no_doc)

        assert doc == ToolDoc()

    def test_from_function_bare_summary_only(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        def bare() -> None:
            """Just a summary, no sections."""

        doc = ToolDoc.from_function(bare)

        assert doc.summary == "Just a summary, no sections."
        assert doc.parameters == []
        assert doc.raises == []

    def test_render_description_appends_raises_block(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        doc = ToolDoc(
            summary="Return the current time.",
            raises=[ToolDoc.Raise(exception="RuntimeError", description="if the system clock is unavailable.")],
        )

        assert doc.render_description() == (
            "Return the current time.\n\nRaises:\n    RuntimeError: if the system clock is unavailable."
        )

    def test_render_description_no_raises_returns_summary_only(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        doc = ToolDoc(summary="Return the current time.")

        assert doc.render_description() == "Return the current time."

    def test_to_meta_omits_empty_sections(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        doc = ToolDoc(
            summary="Return the current time.",
            raises=[ToolDoc.Raise(exception="RuntimeError", description="if the system clock is unavailable.")],
        )

        assert doc.to_meta() == {
            "raises": [{"exception": "RuntimeError", "description": "if the system clock is unavailable."}],
        }

    def test_to_meta_includes_all_sections_when_present(self):
        from velari_ai.integrations.fastmcp.types import ToolDoc

        doc = ToolDoc(
            summary="Square a number.",
            parameters=[ToolDoc.Parameter(name="x", annotation="int", description="The number to square.")],
            returns=ToolDoc.Returns(annotation="int", description="The squared value."),
            raises=[ToolDoc.Raise(exception="ValueError", description="if x is negative.")],
            examples=[">>> square(3)\n9"],
        )

        assert doc.to_meta() == {
            "parameters": [{"name": "x", "annotation": "int", "description": "The number to square."}],
            "returns": {"annotation": "int", "description": "The squared value."},
            "raises": [{"exception": "ValueError", "description": "if x is negative."}],
            "examples": [">>> square(3)\n9"],
        }


class TestResourceSpec:
    def test_to_kwargs_excludes_uri_and_fn(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        spec = ResourceSpec(uri="config://settings", fn=lambda: {}, name="read_config")

        kwargs = spec.to_kwargs()

        assert "uri" not in kwargs
        assert "fn" not in kwargs
        assert kwargs == {"name": "read_config"}

    def test_to_kwargs_omits_unset_optional_fields(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        spec = ResourceSpec(
            uri="config://settings",
            fn=lambda: {},
            name="read_config",
            mime_type="application/json",
            tags={"config", "server"},
            meta={"source": "hydra"},
        )

        assert spec.to_kwargs() == {
            "name": "read_config",
            "mime_type": "application/json",
            "tags": {"config", "server"},
            "meta": {"source": "hydra"},
        }

    def test_from_config_resolves_handler_and_uri(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        def read_config():
            """Read config."""
            return {}

        entry = OmegaConf.create({"handler": "read_config", "uri": "config://settings"})
        handlers = {"read_config": read_config}

        spec = ResourceSpec.from_config(entry, handlers)

        assert spec.uri == "config://settings"
        assert spec.fn is read_config

    def test_from_config_derives_description_from_handler_docstring_when_unset(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        def read_config(path):
            """Read the config file."""
            return {}

        # functools.partial has no __doc__ of its own — from_config must unwrap to the
        # real function so the description reflects read_config's docstring, not
        # partial()'s own generic docstring.
        handlers = {"read_config": functools.partial(read_config, "/tmp/config.yaml")}
        entry = OmegaConf.create({"handler": "read_config", "uri": "config://settings"})

        spec = ResourceSpec.from_config(entry, handlers)

        assert spec.description == "Read the config file."

    def test_from_config_uses_explicit_description_when_provided(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        def read_config():
            """Read config."""
            return {}

        entry = OmegaConf.create({
            "handler": "read_config",
            "uri": "config://settings",
            "description": "Custom description.",
        })

        spec = ResourceSpec.from_config(entry, {"read_config": read_config})

        assert spec.description == "Custom description."

    def test_from_config_resolves_meta_interpolation(self):
        from velari_ai.integrations.fastmcp.types import ResourceSpec

        def read_config():
            """Read config."""
            return {}

        cfg = OmegaConf.create({
            "author": "Ari Kamlani",
            "resources": [{
                "handler": "read_config",
                "uri": "config://settings",
                "meta": {"source": "hydra", "author": "${author}"},
            }],
        })
        entry = cfg.resources[0]

        spec = ResourceSpec.from_config(entry, {"read_config": read_config})

        assert spec.meta == {"source": "hydra", "author": "Ari Kamlani"}
