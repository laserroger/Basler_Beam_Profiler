"""Keep the test suite away from the user's real fit_config.json.

`fitconfig` loads that file at import, so without this every test would run
against whatever the last GUI session left behind - and would overwrite it.
"""

import pytest

from beam_profiler import fitconfig


@pytest.fixture(autouse=True)
def isolated_fit_config(tmp_path_factory, monkeypatch):
    """Every test starts from the defaults and writes only to a temp file."""
    path = tmp_path_factory.mktemp("fitcfg") / "fit_config.json"
    monkeypatch.setattr(fitconfig, "CONFIG_PATH", str(path))
    fitconfig.set_active(fitconfig.FitConfig(), save=False)
    yield
    fitconfig.set_active(fitconfig.FitConfig(), save=False)
