"""Tests for the velari_core package."""

import logging


def test_import_velari_core():
    import velari_core

    assert velari_core is not None


def test_version():
    from velari_core.version import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_logger_available():
    from velari_core import logger

    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_logger_info(caplog):
    from velari_core import logger

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("test log message")
    assert "test log message" in caplog.text
