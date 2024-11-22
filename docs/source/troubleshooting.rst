.. _troubleshooting:

===============
Troubleshooting
===============

.. _common_issues:

Common Issues
=============

Conda fails to solve dependencies when installing
-------------------------------------------------

Ensure conda-forge has the highest channel priority, and channel priority is set to strict.

.. code-block:: bash

    conda install --file conda_requirements.txt -c conda-forge

The ``richdem`` package won't install or cannot be found when creating a static file
------------------------------------------------------------------------------------

The ``richdem`` package is not available on PyPI for Python 3.11+.
If you are using a pip venv, you will need to `compile richdem yourself`_.

.. _`compile richdem yourself`: https://github.com/r-barnes/richdem?tab=readme-ov-file#compilation

Consider using conda/mamba to install dependencies:

.. code-block:: bash

    conda install richdem

COSIPY does not work on Windows
-------------------------------

Windows is not currently supported. Consider installing `Windows Subsystem for Linux`_.

.. _`Windows Subsystem for Linux`: https://learn.microsoft.com/en-us/windows/wsl/install

Input data is out of range
--------------------------

Please check your settings in the utilities configuration for ``aws2cosipy`` and ``wrf2cosipy``.
By default, the ``in_K`` parameter is set to true.

Data in the output file do not match the input file
---------------------------------------------------

Some input variables are modified when loaded into COSIPY, e.g. negative values in incoming solar radiation are set to zero.

``interp_subT()`` raises an IndexError
--------------------------------------

The glacier has melted away completely at a particular grid point.
Consider increasing the initial glacier height.

Parameters set in the Slurm configuration file are not passed on to Slurm
-------------------------------------------------------------------------

Ensure parameters between your Slurm configuration file are identical to those in your sbatch script.

.. _developer_issues:

Developer Issues
================

Error when referencing Config.<X> or Constants.<X> inside an njitted function
-----------------------------------------------------------------------------

Variables must be declared at the module level if you wish to use them in an njitted function:

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

Dask workers run out of memory when saving the output
-----------------------------------------------------

This is a known issue.
Consider selecting fewer output variables under ``[OUTPUT_VARIABLES]`` in ``config.toml``.

Cannot set values of imported ``Config`` or ``Constants`` attributes
--------------------------------------------------------------------

Parameters parsed from configuration files are read-only.
Some users work around this by passing parameters as a ``dict`` argument into ``cosipy_core``.
