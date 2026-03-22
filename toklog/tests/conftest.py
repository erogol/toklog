"""Redirect toklog log dir to a temp directory for the entire test session."""

import tempfile
import shutil
import toklog.logger as _logger

import pytest


@pytest.fixture(scope="session", autouse=True)
def _redirect_log_dir(tmp_path_factory):
    tmp_dir = tempfile.mkdtemp()
    original_log_dir = _logger._LOG_DIR
    original_dir_ensured = _logger._dir_ensured

    _logger._LOG_DIR = tmp_dir
    _logger._dir_ensured = False

    yield

    _logger._LOG_DIR = original_log_dir
    _logger._dir_ensured = original_dir_ensured
    shutil.rmtree(tmp_dir, ignore_errors=True)
