"""Tests for the velari_ai package."""

import logging


def test_import_velari_ai():
    import velari_ai

    assert velari_ai is not None


def test_version():
    from velari_ai.version import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_logger_available():
    from velari_ai import logger

    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_logger_info(caplog):
    from velari_ai import logger

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("test log message")
    assert "test log message" in caplog.text
