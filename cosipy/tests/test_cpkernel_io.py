import shutil
from pathlib import Path

import pytest
import xarray as xr

from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.cpkernel.io import IOClass

class TestIOClass:
    """Test the IOClass class."""

    Config()

    @pytest.fixture(autouse=True)
    def fixture_IO(self):
        """Test the init function."""
        Config()
        Constants()
        Config.data_path = Path.cwd() / "cosipy" / "tests" / "data"

    def test_restart_file_does_not_exist(self):
        """Test the load_restart_file function with a non-existent file."""
        IO = IOClass()
        with pytest.raises(FileNotFoundError):
            IO.load_restart_file()
        assert not hasattr(IO, "restart_date")
        assert not hasattr(IO, "GRID_RESTART")

    def test_timestamps_are_equal(self):
        IO = IOClass()
        Config.time_start = "2009-01-10T00:00"
        Config.time_end = "2009-01-10T00:00"

        with pytest.raises(IndexError):
            IO.load_restart_file()
        assert not hasattr(IO, "restart_date")
        assert not hasattr(IO, "GRID_RESTART")

    def test_load_restart_file(self):
        """Test the load_restart_file function."""
        Config.time_start = "2009-01-10T00:00"
        Config.time_end = "2009-01-11T00:00"

        IO = IOClass()
        IO.load_restart_file()
        assert hasattr(IO, "GRID_RESTART")
        assert isinstance(IO.GRID_RESTART, xr.Dataset)
        assert list(IO.GRID_RESTART.variables) == [
            "time",
            "lat",
            "lon",
            "layer",
            "NLAYERS",
            "NEWSNOWHEIGHT",
            "NEWSNOWTIMESTAMP",
            "OLDSNOWTIMESTAMP",
            "LAYER_HEIGHT",
            "LAYER_RHO",
            "LAYER_T",
            "LAYER_LWC",
            "LAYER_IF",
        ]
