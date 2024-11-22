.. _documentation:

===============
Getting started
===============

.. _requirements:

Requirements
============

COSIPY is compatible with Python 3.9-3.12 on Linux and MacOS.
If you think your specific Python version or operating system causes an issue with COSIPY, please create a topic in the forum.
The model is tested and developed on:

 * Python 3.9-3.12 on Ubuntu 22.04 / Debian 12
 * Anaconda distribution on macOS
 * Anaconda 3 64-bit (Python 3.9) on CentOS Linux 7.4
 * High-Performance Cluster Erlangen-Nuremberg University 

.. warning::
    COSIPY 2.0 is not backwards-compatible with older versions of COSIPY.
    Please :ref:`read the instructions <upgrading>` on upgrading from an older version, and using the :ref:`new configuration system <configuration>`.

.. _installation:

Installation
============

For Python 3.11 and above, the recommended environment manangers are conda/mamba.

Pre-requisites
--------------

Install GDAL:

.. code-block:: bash

    sudo apt-get install gdal-bin libgdal-dev
    pip install --upgrade gdal==`gdal-config --version` pybind11  # with pip
    conda install gdal  # with conda/mamba

If you are installing dependencies with conda/mamba, use ``-c conda-forge`` if it does not already have the highest channel priority.

.. note:: If you are installing with **pip** and Python 3.11+, you will need to `compile and install richdem`_.
    This is not necessary when using conda/mamba.

.. _`compile and install richdem`: https://github.com/r-barnes/richdem?tab=readme-ov-file#compilation

The ``icc_rt`` package may provide a performance boost on some systems:

.. code-block:: bash

    pip install icc-rt             # with pip
    conda install icc_rt -c numba  # with conda/mamba

Installation with pip
---------------------

Installing COSIPY as a package allows it to run from any directory:

.. code-block:: bash

    pip install cosipymodel
    conda install richdem  # if you are using conda/mamba and python 3.11+
    cosipy-setup  # generate sample configuration files
    cosipy-help   # view help

This is the recommended installation method if you do not plan to modify the source code.

Installation from Source
------------------------

Activate your preferred python environment, then run:

.. code-block:: bash

    git clone https://github.com/cryotools/cosipy.git
    pip install -r requirements.txt              # install default environment
    pip install -r dev_requirements.txt          # install dev environment
    conda install --file conda_requirements.txt  # install using conda/mamba
    python3 COSIPY.py -h

Installation as an Editable
---------------------------

Installing COSIPY as an editable allows it to run from any directory.

.. code-block:: bash

    git clone https://github.com/cryotools/cosipy.git
    cd cosipy
    pip install -e .
    pip install -e .[tests] # install with dependencies for tests
    pip install -e .[docs]  # install with dependencies for documentation
    pip install -e .[dev]   # install with dependencies for development
    cosipy-setup            # generate sample configuration files
    cosipy-help             # view help

.. _upgrading:

Upgrading from an Older Version of COSIPY
-----------------------------------------

COSIPY 2.0 is not backwards-compatible with previous versions of COSIPY.
If you have written your own modules that import from ``constants.py``, ``config.py``, or use Slurm, these will break.

Navigate to COSIPY's root directory and convert your existing configuration files:

.. code-block:: bash

    pip install toml
    git fetch --all
    git checkout master -- convert_config.py
    python convert_config.py  # convert .toml files

This works on any branch regardless of local changes.
Alternatively you can copy and run ``convert_config.py`` into any older COSIPY source tree.
This will preserve your configuration for ``config.py``, ``constants.py``, ``aws2cosipyConfig.py`` and ``wrf2cosipyConfig.py``.

.. warning::
    Parameters for ``create_static`` must still be added manually to the generated ``utilities_config.toml``.
    Custom configuration variables that do not appear in the main branch must also be added manually.

Checkout a new branch with a clean version of COSIPY and merge your modifications.

.. code-block:: bash

    git checkout master
    git pull
    git checkout -b <new-branch-name>
    git merge --no-ff <old-branch-name>  # Good luck!

You can also merge the new version of COSIPY into an existing branch, but this creates even more merge conflicts.
Most conflicts involve importing configuration parameters or constants.
In most cases you simply need to prepend ``Config.`` or ``Constants.`` to a variable.
Please read the documentation for the :ref:`new configuration system <configuration>`.

After updating to the latest version of COSIPY, run ``python COSIPY.py --help`` to see how to specify paths to configuration files.
COSIPY will default to ``./config.toml``, ``./constants.toml``, ``./slurm_config.toml``, ``./utilities_config.toml`` in the current working directory.
**You no longer need to hardcode different simulation parameters into a single file.**

.. _tutorial:

Tutorial
========

For this tutorial, download or copy the sample ``data`` folder and place it in your COSIPY working directory.
If you have installed COSIPY as a package, you can use the entry point ``setup-cosipy`` to generate the sample configuration files.
Otherwise, run ``python -m cosipy.utilities.setup_cosipy.setup_cosipy``.

Pre-Processing
--------------

COSIPY requires a file with the corresponding meteorological and static input data.
Various tools are available to create the file from simple text or geotiff files.

.. _static_tutorial:

Create the static file
~~~~~~~~~~~~~~~~~~~~~~~

In the first step, topographic parameters are derived from a Digital Terrain Model (DEM) and written to a netCDF file.
A shape file is also required to delimit the glaciated areas.
The DEM and the shapefile should be in lat/lon WGS84 (EPSG:4326) projection.

.. note:: The DEM can be reprojected to EPSG:4326 using gdal:

    .. code-block:: bash

        gdalwarp -t_srs EPSG:4326 dgm_hintereisferner.tif dgm_hintereisferner-lat_lon.tif


COSIPY comes with the script ``create_static_file.py`` located in the utilities folder.
This script runs some gdal routines in the command line.
At the moment this is only compatible with UNIX and MacOS.
The script creates some intermediate netCDF files (dem.nc, aspect.nc, mask.nc and slope.nc) that are automatically deleted after the static file is created.

Open ``utilities_config.toml``.
Under ``create_static.paths``, check the paths point to the DEM **n30_e090_3arc_v2.tif** (SRTM) and the shapefile **Zhadang_RGI6.shp** provided in the ``./data/static/`` folder.

The static file is created using either:

.. code-block:: bash

    python -m cosipy.utilities.createStatic.create_static_file  # from source
    cosipy-create-static  # from entry point

The command creates a new file **Zhadang_static.nc** in the ``./data/static/`` folder.

.. _input_tutorial:

Create the COSIPY input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating the input file requires the static information file from :ref:`the previous section<static_tutorial>`.
To convert the data from an automatic weather station (AWS) use the conversion script ``aws2cosipy.py``, located in the folder ``./utilities/aws2cosipy/``.
A sample configuration is available in ``utilities_config.toml`` which defines the structure of the AWS file and other user-defined parameters.
Since the input file provides point information, the data is interpolated via lapse rates for two-dimensional runs.
The solar radiation fields are based on a model by `Wohlfahrt et al. (2016)`_.
Other variables like wind velocity and cloud cover fraction are assumed constant over the domain.

.. _`Wohlfahrt et al. (2016)`: https://doi.org/10.1016/j.agrformet.2016.05.012

.. note:: The script ``aws2cosipy.py`` is only an illustration of how data can be prepared for COSIPY.
    For most applications it is recommended to develop your own data interpolation routines.

The script is executed with:

.. code-block:: bash

    # from source
    python -m cosipy.utilities.aws2cosipy.aws2cosipy \
        -i ./data/input/Zhadang/Zhadang_ERA5_2009_2018.csv \
        -o ./data/input/Zhadang/Zhadang_ERA5_2009.nc \
        -s ./data/static/Zhadang_static.nc \
        -b 20090101 -e 20091231

    # from entry point
    cosipy-aws2cosipy \
        -i ./data/input/Zhadang/Zhadang_ERA5_2009_2018.csv \
        -o ./data/input/Zhadang/Zhadang_ERA5_2009.nc \
        -s ./data/static/Zhadang_static.nc \
        -b 20090101 -e 20091231

If the script executes successfully it will create the file ``./data/input/Zhadang/Zhadang_ERA5_2009.nc``.

**Usage:**

.. code-block:: bash

    cosipy.utilities.aws2cosipy [-h] [-u <path>] -c <path> -o <path> -s <path> [-b <str>] [-e <str>] [-xl <float>] [-xr <float>] [-yl <float>] [-yu <float>]

Required arguments:
    -i, --csv_file <path>       Path to .csv file with meteorological data.
    -o, --cosipy_file <path>    Path to the resulting COSIPY netCDF file.
    -s, --static_file <path>    Path to static file with DEM, slope etc.

Optional arguments:
    -u, --utilities <path>      Relative path to utilities' configuration file.
    -b, --start_date <str>      Start date.
    -e, --end_date <str>        End date.
    --xl <float>                Left longitude value of the subset.
    --xr <float>                Right longitude value of the subset.
    --yl <float>                Lower latitude value of the subset.
    --yu <float>                Upper latitude value of the subset.

.. _run_model:

Run the COSIPY model
--------------------

To run COSIPY, run the following command in the root directory:

.. code-block:: bash

    python COSIPY.py  # from source
    run-cosipy        # from package

The example should take less than a minute on a workstation with 4 cores.

.. _run_usage:

Running COSIPY
==============

COSIPY accepts arguments specifying paths to configuration files.

**Usage:**

.. code-block:: bash

    python COSIPY.py [-h] [-c <path>] [-x <path>] [-s <path>]  # from source
    run-cosipy [-h] [-c <path>] [-x <path>] [-s <path>]  # from package

Optional arguments:
    -c <path>, --config <path>      Relative path to configuration file.
    -x <path>, --constants <path>   Relative path to constants file.
    -s <path>, --slurm <path>       Relative path to Slurm configuration file.

.. _entry_points:

Entry Points
------------

If installed as an editable or package, COSIPY provides several entry points to speed up common operations.
These entry points accept python arguments (such as ``--help``).

Available shortcuts:
    :cosipy-help:           Display help for running COSIPY.
    :cosipy-shortcuts:      Display available entry points.
    :cosipy-setup:          Setup missing configuration files.
    :cosipy-run:            Run COSIPY. Accepts python arguments.
    :cosipy-aws2cosipy:     Convert AWS data to netCDF.
    :cosipy-create-static:  Create static file.
    :cosipy-wrf2cosipy:     Convert WRF data to netCDF.
    :cosipy-plot-field:     Generate field plots.
    :cosipy-plot-profile:   Generate profile plots.
    :cosipy-plot-vtk:       Generate 3D plots.
    :help-cosipy:           Alias for ``cosipy-help``.
    :run-cosipy:            Alias for ``cosipy-run``.
    :setup-cosipy:          Alias for ``cosipy-setup``.

.. _configuration:

Configuration
=============

.. note:: Configure parameters/constants in ``config.toml``, ``constants.toml``, ``slurm_config.toml`` and ``utilities_config.toml``.

All user configuration is done with .toml files.
If COSIPY is installed as a package, generate sample configuration files using ``setup-cosipy``.
Configuration is split into four parts: model configuration, constants, utilities, and Slurm configuration.
You can keep multiple configuration files for different simulations in the same (or indeed any working directory).

If you are using a cluster, you can also chain multiple simulations with a single batch script:

.. code-block:: bash

    python $WORK/COSIPY.py -c config_01.toml -x constants_01.toml -s slurm_01.toml
    python $WORK/COSIPY.py -c config_02.toml -x constants_02.toml -s slurm_02.toml

Select which output variables are saved to disk under ``[OUTPUT_VARIABLES]`` in ``config.toml``.
This can prevent out-of-memory errors when working with very large datasets.
This replaces the ``cosipy/output`` and ``cosipy/output.full`` files.

Import configuration parameters or constants into a module.
These are read-only to avoid namespace collisions.

.. code-block:: python

    from cosipy.config import Config
    from cosipy.constants import Constants

    foo = Config.foo  # declare at module level if used in an njitted function

    @njit
    def get_foo_njit(...):
        """Njitted functions cannot reference the imported parameters directly."""
        return foo

    def get_foo_nopython(...):
        """Non-compiled functions can reference the parameters directly."""
        return Config.foo
