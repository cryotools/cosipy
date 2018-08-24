
# IN PROGRESS

# Introduction ##

### Contact
Tobias Sauter, tobias.sauter@fau.de
Anselm Arndt, anselm.arndt@geo.hu-berlin.de

# Requirements
Libraries

## Input
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
| Air Pressure| PRES | hPa| |
| Cloud cover | N | - | |
| Relative humidity | RH2 | %/100 | |
| Snowfall | SNOWFALL | m | optional |
| Solar radiation | G | W m<sup>-2</sup> | |
| Air temperature | T2 | K | |
| Wind speed | U2 | m s<sup>-1</sup> | |
| Incoming shortwave radiation | LWin | W m<sup>-2</sup> | optional |

# Quick tutorial

## Create input file

```bash
gdal_warp asdfasdfa
```

COSPY provides some utilities which can be used to create the input file 

## Run model

## Evaluation

## Restart


# Model Structure

## Directories

| Directory | Files | Content |
|---|---|---|
|   | COSIPY.py | Main program |
|---|---|---|
|cpkernel | core_cosipy.py | Core of the model (time loop) |
|         | grid.py | Grid structure, consists of a list of layer nodes (vertical snow profile) |
|         | node.py | Node class handles the nformation of each layer |
|         | init.py | Initialization of the snow cover |
|         | io.py | Contains all input/output functions |
|---|---|---|
| modules | albedo.py | Albedo parametrization |
|         | densification.py | Snowpack densification |
|         | heatEquation.py | Solves the heat equation |
|         | penetratingRadiation.py | Parametrization of the penetrating shortwave radiation |
|         | percolation_incl_refreezing.py | Liquid water percolation and refreezing |
|         | radCor.py | Radiation model (topographic shading) |
|         | roughness.py | Update of the roughness length |
|         | surfaceTemperature.py | Solves the energy balance at the surface and calculates the surface temperature |


## Output

# Config options

## Open issues:

* densification
* subsurface melt (penetrating radiation)

In future model should work with total precipitation (separeted with linear allocation between zero degree and 5 degree)
with liquid and solid precipitation (solid used as snowfall, liquid add to surface melt for percolation)



REWRITE AND DELETE THE FOLLOWING!!!!
These are basic information about the python version of
the 'COupled Snowpack and Ice surface energy and MAss balance glacier
model' (COSIMA). The model is originally written and developed in
Matlab code by Huintjes et al., (2015) and is available on [https://bitbucket.org/glaciermodel/cosima/src](https://bitbucket.org/glaciermodel/cosima/src) or @glaciermodel.

The Python translation and model improvement of COSIMA was done by
@tsauter and @bjoesa under the umbrella of the Institute of
Geography, Friedrich-Alexander-University Erlangen-Nuernberg.

The python version of the model is subsequently called >> COSIPY <<.

The model is written in Python 3.6 and is tested on Anaconda2-4.4.0 64-bit
distribution with additional packages.

For more information about the model physics please read:

Huintjes, E., Sauter, T., Schröter, B., Maussion, F., Yang, W.,
 Kropáček, J., Buchroithner, M., Scherer, D., Kang, S. and
 Schneider, C.: Evaluation of a Coupled Snow and Energy Balance Model
 for Zhadang Glacier, Tibetan Plateau, Using Glaciological Measurements
 and Time-Lapse Photography, Arctic, Antarctic, and Alpine Research,
 47(3), 573–590, doi:10.1657/AAAR0014-073, 2015.
  
Current version: 0.1 (Feb 2017)

Current status: development

### Structure and setup ###

**input_COSIMA-example.mat** test data from Huintjes et al. (2015)

**COSIPY.py** main model file -- this file has to be executed in your terminal:

```
#!python

python COSIPY.py
```
Hint: Have a look inside COSIPY.py, you can easily catch the model structure.

**config.py** This is the only file where you make your individual adaptions:

* path to the input data file (currently .mat; will change to NetCDF)

* time index to start

* time index to stop, default: length of the time series

* length of the time step

* information/debug level, default: 0

* layer merging level, default: 0 (no merging)

* Minimal height of layers, thin layers rise computing time

* more variables ...

**inputData.py** contains the routine to read the model forcing as variables

**Constants.py** contains all variables which store applied constants

**Grid.py** contains the functions to setup, read and modify the data Grid

**Node.py** contains the functions to set, get and modify nodes in the Grid

### PHYSICAL MODULES ###

**albedo.py** updates the Albedo

**roughness.py** updates the Roughness

**heatEquationLagrange.py** solves the Heat Equation

**surfaceTemperature.py** updates the Surface Temperature

**penetratingRadiation.py** calculates the penetrating radiation into the
                           snowpack

**percofreeze.py** updates the liquid water content of each layer and
                  calculates refreezing and runoff

### Modules currently in development: ###

percolation.py (percofreeze.py)

densification.py

results2nc.py

### Model forcing ###

* u2 = Wind speed (magnitude) [m/s]

* G = Solar radiation at each time step [W m-2]

* T2 = Air temperature (2m over ground) [K]

* rH2 = Relative humidity (2m over ground)[%]

* snowfall = Snowfall per time step [m]

* p = Air Pressure [hPa]

* N = Cloud cover [%/100]

* sh = Initial snow height [m]

### Contribution guidelines ###

Please branch or fork your version, do not change the master.

You are allowed to use and modify this code in a noncommercial manner and by
appropriately citing the above mentioned developers.

### Who do I talk to? ###

Master maintainance: @bjoesa, bjoern.sass[at]fau.de
