.. image:: https://cryo-tools.org/wp-content/uploads/2019/11/COSIPY-logo-2500px.png

The coupled snowpack and ice surface energy and mass balance model in Python COSIPY solves the energy balance at the surface and is coupled to an adaptive vertical multi-layer subsurface module.

Documentation
-------------
The documentation for COSIPY is available at the following link:
https://cosipy.readthedocs.io/en/latest/

**Confused about migrating to the new .toml configuration system?**
The documentation contains an in-depth tutorial and a guide on upgrading.

Convert your existing configuration files before merging the latest update:

.. code-block:: console

    pip install toml
    git fetch --all
    git checkout master -- convert_config.py
    python convert_config.py  # generate .toml files

This works on any branch regardless of local changes.
Alternatively you can copy and run ``convert_config.py`` into any older COSIPY source tree.

This will preserve your configuration for ``config.py``, ``constants.py``, ``aws2cosipyConfig.py`` and ``wrf2cosipyConfig.py``.
Parameters for ``create_static`` must still be added manually to the generated ``utilities_config.toml``.

Checkout a new branch with a clean version of COSIPY and merge your modifications.
This minimises the number of merge conflicts.
After updating to the latest version of COSIPY, run ``python COSIPY.py --help`` to see how to specify paths to configuration files.
COSIPY will default to ``./config.toml``, ``./constants.toml``, ``./slurm_config.toml``, ``./utilities_config.toml`` in the current working directory.

Installation
------------

Install GDAL:
.. code-block:: console

    sudo apt-get install gdal-bin libgdal-dev
    pip install --upgrade gdal==`gdal-config --version` pybind11  # with pip

Install COSIPY with pip (for general use):
.. code-block:: console

    pip install cosipymodel
    cosipy-setup  # generate template configuration files
    cosipy-help   # view help

Install COSIPY from source (for development):
.. code-block:: console

    git clone https://github.com/cryotools/cosipy.git
    pip install -r requirements.txt              # install default environment
    pip install -r dev_requirements.txt          # install dev environment
    conda install --file conda_requirements.txt  # install using conda/mamba
    python3 COSIPY.py -h

Communication and Support
-------------------------
We are using the groupware slack for communication (inform about new releases, bugs, features, ...) and support:
https://cosipy.slack.com

About
-----

:Tests:
    .. image:: https://github.com/cryotools/cosipy/actions/workflows/python-app.yml/badge.svg?branch=master
        :target: https://github.com/cryotools/cosipy/actions/workflows/python-app.yml

    .. image:: https://readthedocs.org/projects/cosipy/badge/?version=latest
        :target: https://cosipy.readthedocs.io/en/latest/

    .. image:: http://www.repostatus.org/badges/latest/active.svg
        :target: http://www.repostatus.org/#active

    .. image:: https://travis-ci.org/cryotools/cosipy.svg?branch=master
        :target: https://travis-ci.org/cryotools/cosipy

    .. image:: https://codecov.io/gh/cryotools/cosipy/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/cryotools/cosipy

:Citation:
    .. image:: https://img.shields.io/badge/Citation-GMD%20paper-orange.svg
        :target: https://gmd.copernicus.org/articles/13/5645/2020/

    .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3902191.svg
        :target: https://doi.org/10.5281/zenodo.2579668

:License:
    .. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
        :target: http://www.gnu.org/licenses/gpl-3.0.en.html
