"""Tests for velari_ai.ai.tracing.trace."""

import pytest


def test_arize_connector_implements_tracing_connector():
    from velari_ai.integrations.arize.connection import Connector
    from velari_ai.ai.tracing.trace import TracingConnector

    assert issubclass(Connector, TracingConnector)


def test_mlflow_connector_implements_tracing_connector():
    pytest.importorskip("mlflow")
    from velari_ai.integrations.mlflow.connection import Connector
    from velari_ai.ai.tracing.trace import TracingConnector

    assert issubclass(Connector, TracingConnector)
