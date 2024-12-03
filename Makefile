#!/usr/bin/make
# 2024 Nicolas Gampierakis

LIBNAME=cosipy

ifeq (, $(shell which python ))
  $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif
PYTHON=$(shell which python)
PYTHON_VERSION=$(shell $(PYTHON) -c "import sys;\
	version='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));\
	sys.stdout.write(version)")
PYTHON_CHECK_MAJOR=$(shell $(PYTHON) -c 'import sys;\
  	print(int(float("%d"% sys.version_info.major) >= 3))')
PYTHON_CHECK_MINOR=$(shell $(PYTHON) -c 'import sys;\
  	print(int(12 >= float("%d"% sys.version_info.minor) >= 9))' )
PYTHON_SOURCES=src/$(LIBNAME)/[a-z]*.py

TEST_SOURCES=tests/[a-z]*.py
DOCS_SOURCES=docs

LOG_DATE=$(shell date +%Y%m%d_%H%M%S)

.PHONY:	help
help:	## Display this help screen
		@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY:	install
install:	install-conda-env	## Install editable package using conda/mamba
		@echo "\nInstalling editable..."
		@pip install -e .

install-conda-env:	--check-python --hook-manager	## Install conda/mamba dependencies
		@echo "\nInstalling dependencies (core)..."
		@$(pkg-manager) install --file conda_requirements.txt -c conda-forge

install-pip:	## Install editable package using pip
		@echo "\nInstalling editable..."
		$(PYTHON) -m pip install --upgrade gdal==`gdal-config --version` pybind11
		$(PYTHON) -m pip install -e .

install-pip-tests:	install-pip	## Install editable package with tests using pip
		@echo "\nInstalling editable with tests..."
		$(PYTHON) -m pip install --upgrade gdal==`gdal-config --version` pybind11
		@pip install -e .[tests]

install-pip-docs:	## Install editable package with local documentation using pip
		@echo "\nInstalling editable with documentation..."
		$(PYTHON) -m pip install --upgrade gdal==`gdal-config --version` pybind11
		@pip install -e .[docs]
		@make docs

install-pip-all:	## Install editable package with tests & documentation using pip
		@echo "\nInstalling editable with tests & documentation..."
		$(PYTHON) -m pip install --upgrade gdal==`gdal-config --version` pybind11
		@pip install -e .[tests,docs]
		@make docs

install-pip-dev:	--check-python --hook-manager	## Install editable package in development mode using pip
		@echo "\nInstalling editable in development mode..."
		$(PYTHON) -m pip install --upgrade gdal==`gdal-config --version` pybind11
		@pip install -e .[dev]

.PHONY:	tests
tests:	flake8 coverage pylint	## Run tests

.PHONY:	commit
commit:	tests	## Test, then commit
		@echo "\nCommitting..."
		@git commit

.PHONY: docs
docs:	## Build documentation
		@echo "\nBuilding documentation..."
		@cd $(DOCS_SOURCES); make clean && make html

format:	isort black	## Format all python files

setup-cosipy:	## Generate COSIPY configuration files
		@$(PYTHON) -m cosipy.utilities.setup_cosipy.setup_cosipy

create-static:
		@$(PYTHON) -m cosipy.utilities.createStatic.create_static_file

commands:	## Display help for COSIPY
		@$(PYTHON) -m COSIPY.py -h

.PHONY: run
run:	commands	## Alias for `make commands`

flake8:	## Lint with flake8
		@echo "\nLinting with flake8..."
		@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
		@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

.PHONY:	coverage
coverage:	## Run pytest with coverage
		@echo "\nRunning tests..."
		@mkdir -p "./logs/coverage"
		@coverage run --rcfile .coveragerc -m pytest && coverage html

.PHONY: pylint
pylint:	## Lint with Pylint
		@echo "\nLinting with pylint..."
		@pylint --rcfile .pylintrc **/*.py

black:	## Format all python files with black
		@echo "\nFormatting..."
		@black $(PYTHON_SOURCES) --line-length=80
		@black $(TEST_SOURCES) --line-length=80

isort:	## Optimise python imports
		@isort $(PYTHON_SOURCES)
		@isort $(TEST_SOURCES)

.PHONY:	pkg
pkg:	tests docs build	## Run tests, build documentation, build package

.PHONY: build
build:	--install-build-deps	## Build COSIPY package
		$(PYTHON) -m build
		@twine check dist/*

bump-version:
		@bash bump_version.sh

.PHONY:
upload-pypi:	# Private: upload COSIPY package
		@twine check dist/*
		@twine upload dist/*


--install-pip-deps:	--check-python  # Private: install core dependencies with pip
		@echo "\nInstalling dependencies with pip..."
		$(PYTHON) -m pip install --r requirements.txt
		
--install-build-deps:	--check-python	--hook-manager	# Private: install build dependencies
		@echo "\nInstalling build dependencies..."
		$(PYTHON) -m pip install --upgrade build hatchling twine

--check-python:	# Private: check Python is >=3.9
ifeq ($(PYTHON_CHECK_MAJOR),0)
	$(error "Python version is $(PYTHON_VERSION). Requires Python >= 3.9")
else ifeq ($(PYTHON_CHECK_MINOR),0)
	$(error "Python version is $(PYTHON_VERSION). Requires Python >= 3.9")
endif

--hook-manager:	# Private: hook package manager
ifneq (,$(findstring mamba, ${CONDA_EXE}))
pkg-manager := @mamba
else ifneq (,$(findstring miniforge, ${CONDA_EXE}))
pkg-manager := @mamba
else ifeq (,$(findstring conda, ${CONDA_EXE}))
pkg-manager := @conda
else
	$(error "No conda/mamba installation found. Try pip install -e . instead")
endif


