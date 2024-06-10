"""Copyright 2023 COSIPY Contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

=====

Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the
documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# pylint: skip-file
# flake8: noqa: F541

# -- General configuration ---------------------------------------------------

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'COSIPY'
copyright = f"2019-{date.today().year}, COSIPY Contributors"
author = 'COSIPY Contributors'
version = 'v1.4'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx_rtd_theme',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
]

# Default processing flags for sphinx
#autoclass_content = 'class'
#autodoc_member_order = 'bysource'
autodoc_default_flags = [
  'members',
  'undoc-members',
  'show-inheritance',
  ]


autosummary_generate = True

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

napoleon_use_param = True
napoleon_use_rtype = True

pygments_style = 'sphinx'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

master_doc = 'index'
html_show_sphinx = False
trim_footnote_reference_space = True
html_static_path = ['_static']
highlight_language = 'python'
html_title = 'COSIPY'
html_logo = './_static/cosipy_logo_transparent.png'

