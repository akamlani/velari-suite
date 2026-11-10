"""Tests for the velari_harness package."""

import logging


def test_import_package_succeeds():
    import velari_harness

    assert velari_harness is not None


def test_version_is_string_on_import():
    from velari_harness import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_logger_is_logging_logger_on_import():
    from velari_harness import logger

    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_logger_info_message_is_captured(caplog):
    from velari_harness import logger

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("test log message")
    assert "test log message" in caplog.text
