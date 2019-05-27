# IN PROGRESS
### ToDos in README:
* Finish Quick tutorial
* Finish Model structure
* describe config options
* describe plot routines
# Introduction ##
COSIPY solves the energy balance at the surface and is coupled to an adaptive vertical multilayer snow and ice module.
### Contact
Tobias Sauter, tobias.sauter@fau.de <br>
Anselm Arndt, anselm.arndt@geo.hu-berlin.de

# Requirements
## Packages and libraries
####Python 3
Any Python 3 version on any operating system should work. If you think the reason for a problem might be your specific Python 3 version or your 
operating system, please create a topic in the forum. $LINK$ <br> Model is tested and developed on:
  * Anaconda Distribution on max OS 
  * Python 3.6.5 on Ubuntu 18.04
  * Anaconda 3 64-bit (Python 3.6.3) on CentOS Linux 7.4
  * Cluster Innsbruck

#### Needed Python modules (with an Anaconda installation, they might be already installed):
* xarray
* dask-jobqueue
* netcdf4
* numpy             (included in Anaconda)
* scipy             (included in Anaconda)
* distributed       (included in Anaconda) 

## Input
Some variables are optinal and for ussage it has to be specified in the config file.
### Dynamic 2D fields: 
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
| Air pressure| PRES | hPa| |
| Air temperature | T2 | K | |
| Cloud cover | N | - | |
| Relative humidity | RH2 | %/100 | |
| Solar radiation | G | W m<sup>-2</sup> | |
| Total precipitation | RRR | mm |  |
| Wind speed | U2 | m s<sup>-1</sup> | |
| Snowfall | SNOWFALL | m | optional, replaces RRR |
| Incoming longwave radiation | LWin | W m<sup>-2</sup> | optional, replaces N |
### Static 2D fields:
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
|Glacier mask|MASK|Boolean||
|Elevation|HGT|m a.s.l.||

# Quick tutorial
## Preprocessing
COSiPY provides some utilities which can be used to create the required input file for the core run.
### Create needed combined static input file
The following is the example in the "data/static/" folder. If the procedure does not work for your study area, please try it first
with the example.
#### Required packages and libaries:
* gdal (e.g. in Debian-based Linux distributions package called gdal-bin)
* climate date operators (e.g. in Debian-based Linux distributions package called cdo)
* netCDF Operators (e.g. in Debian-based Linux distritutions package called nco)
#### Needed input files
* Digital elevation model (WGS84 - EPSG:4326)
* Shapefile of the glacier (WGS84 - EPSG:4326)

#### Procedure:
In the utilities folder, there is the script create_static_file_command_line.py. This script runs some commands in the command line.
That's is the reason, that we can provide this script only for UNIX and MAC users at the moment. We are working on a version where no UNIX command
line is needed.
(create_static_file.py).<br>
The intermediate files 'dem.nc', 'aspect.nc', 'mask.nc' and 'slope.nc' are deleted automatically. First, try to run the script
and create the 'static.nc' file with the example 'n30_e090_3arc_v2.tif' (SRTM) and 'Zhadang_RGI6.shp'. If this works, try to change to
your DEM and shapefile and adjust the area to which you want to shrink the DEM. The input data have to be in Lat/Lon
WGS84-EPSG:4326 projection with the units degrees, that the script works correctly. <br>
Run the script with:
```
python create_static_files_with_command_line.py
```
Hint: to reproject the DEM to EPSG 4326 (not needed for the example in the source code):
```
gdalwarp -t_srs EPSG:4326 dgm_hintereisferner.tif dgm_hintereisferner-lat_lon.tif
```
### Create input file with all needed static and dynamic 2D fields
#### Needed files and parameters
* static.nc file, created in step above
* 1D fields of all required dynamic input files
#### Procedure:
There are two different preprocessing scripts in the utilities folder to create the needed gridded input data. One is especially desingned for the
usage of csv file from a datalogger of an AWS station. This file is called aws_logger2cosipy.py with the corresponding configuration file 
'aws_logger2cosipyConfig.py'.<br> 
The 'csv2cosipy.py' script with the corresponding configuration file 'csv2cosipyConfig.py' is a more general file.<br>
Very important: For the aws_logger2cosipy.py version the temperature has to be in degree Celsius.<br> For the following example you have to use 
the csv2cosipy.py file.<br>
For the solar radiation, a model after Wohlfahrt et al. (2016; doi: 10.1016/j.agrformet.2016.05.012) is used. <br>
For air temperature, relative humidity and precipitation constant lapse rates, which have to be set, are used. <br>
Wind speed and cloud cover fraction kept constant for all gridpoint at on time step.<br><br>
The script needs:
* the input file; for example a Campbell Scientific logger file with all required dynamic input fields
* the file path (including the name) for the resulting COSIPY file, which will be used as input file for the core run
* the path to the static file, created in the step above
* the start and end date of the timespan

In the csv2cosipyConfig.py one has to define how the input variables are called in the CS_FILE. <br> 
For the radiation module, one has to set the timezone and the zenit threshold. <br> Furthermore, the station name has to be set, the altitude of the station, and the lapse rates for temperature, relative humidity and precipitation.<br>
If everything is set, configured and prepared, run the script:
```bash
py csv2cosipy.py -c ../../data/input/Zhadang_ERA5_2009_2018.csv -o ../../data/input/Zhadang_ERA5_2009.nc -s ../../data/static/static.nc -b 20090101 -e 20091231
```
The script takes all input timestamps which are in the -c input file. If you want only a specific period. The -b and -e option are optional:
```
python csv2cosipy.py -c ../data/input/Zhadang_ERA5_2009_2018.csv -o ../data/input/Zhadang_ERA5_2009_2018.nc -s ../data/static/static.nc

```
## Core run
### Changes config.py and set everything for your specific need. See in config options.
For the example just run:
```
python COSIPY.py
```
in the root folder of the source code. The example execute the model run for January 2009 and should take less than 10 minutes (approx 3 minutes or 1 minute with 4 cores).
## Evaluation

## Restart

## Postprocessing
Need the following python modules:
* 
In folder postprocessing. 
```bash
python plot_cosipy_fields.py -f $OUTPUT_FILE -d $POINT_IN_TIME_OF_INTEREST -t 1
python plot_cosipy_fields.py -h #for help
```
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
Input variables are stored in output dataset as well.
### Dynamic 2D fields: 
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
Air temperature at 2 m|T2|K
Relative humidity at 2 m|RH2|%| 
|Wind velocity at 2 m|U2| m s<sup>-1</sup>|
Liquid precipitation|RAIN| mm|
Snowfall|SNOWFALL| m| 
Atmospheric pressure|PRES| hPa| 
Cloud fraction|N| -| 
Incoming shortwave radiation|G| W m<sup>-2</sup>| 
Incoming longwave radiation|LWin| W m<sup>-2</sup>| 
Outgoing longwave radiation|LWout|W m<sup>-2</sup>| 
Sensible heat flux|H|W m<sup>-2</sup>| 
Latent heat flux|LE|W m<sup>-2</sup>| 
Ground heat flux|B|W m<sup>-2</sup>| 
Available melt energy|ME|W m<sup>-2</sup>| 
Total mass balance|MB| m w.e.| 
Surface mass balance|surfMB| m w.e.| 
Internal mass balance|intMB| m w.e.| 
Evaporation|EVAPORATION| m w.e.| 
Sublimation|SUBLIMATION| m w.e.| 
Condensation|CONDENSATION| m w.e.| 
Depostion|DEPOSITION| m w.e.| 
Surface melt|surfM| m w.e.| 
Subsurface melt|subM| m w.e.| 
Runoff|Q| m w.e.| 
Refreezing|REFREEZE| m w.e.| 
Snowheight|SNOWHEIGHT| m|
Total domain height|TOTALHEIGHT| m|
Surface temperature|TS| K| 
Roughness length|Z0| m| 
Albedo|ALBEDO| -| 
Number of layers|NLAYERS| -|
### Static 2D fields:
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
|Glacier mask|Mask|Boolean||
|Elevation|HGT|m a.s.l.||

# Config options
### describe everything possible!

# Open issues:
$TODO:LINK TO ISSUE SECTION WOULD BE BETTER$
* densification
* subsurface melt (penetrating radiation)

# Planed parameterisations and included processes:
* add heat flux from liquid precipitation
* add liquid precipitation to surface melt
* superimposed ice

# 
Please branch or fork your version, do not change the master.

You are allowed to use and modify this code in a noncommercial manner and by
appropriately citing the above-mentioned developers.
