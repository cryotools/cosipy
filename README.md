# IN PROGRESS
### ToDos in README:
* Finish Quick tutorial
* Finish Model strucute
* describe config options
* describe plot routines

# Introduction ##

### Contact
Tobias Sauter, tobias.sauter@fau.de <br>
Anselm Arndt, anselm.arndt@geo.hu-berlin.de

# Requirements
## Packages and libaries
* Python 3, any Python 3 version on any Operating System should and must work. If you think the reason of a problem might be your specific Python 3 version or your Operating System, please create a topic in the forum. <br> Model is tested and developed on:
  * Anaconda Distribution on max OS 
  * Python 3.6.5 on Ubuntu 18.04
  * Anaconda 3 64-bit (Python 3.6.3) on CentOS Linux 7.4
  * CLUSTER Innsbruck

## Needed Python modules (with an Anaconda installation, they might be already installed):
* numpy
* xarray
* netcdf4
* scipy
* distributed

## Input
Some variables are optinal. If they exist and the users wnat to include them, one has to set theese options in the config.py file.
### Dynamic 2D fields: 
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
| Air Pressure| PRES | hPa| |
| Cloud cover | N | - | |
| Relative humidity | RH2 | %/100 | |
| Snowfall | SNOWFALL | m | optional |
| Total precipitation | RRR | mm |  |
| Solar radiation | G | W m<sup>-2</sup> | |
| Air temperature | T2 | K | |
| Wind speed | U2 | m s<sup>-1</sup> | |
| Incoming longwave radiation | LWin | W m<sup>-2</sup> | optional |
### Static 2D fields:
|Variable|Short Name|Unit|Comment|
|---|---|---|---|
|Glacier mask|Mask|Boolean||
|Elevation|hgt|m a.s.l.||



# Quick tutorial
## Preprocessing
COSPY provides some utilities which can be used to create the required input file for the core run.
### Create needed combined static input file
#### Required packages and libaries:
* gdal (e.g. in Debian-based Linux distributions package called gdal-bin)
* cliamte date operators (e.g. in Debian-based Linux distributions package called cdo)
* netCDF Operators (e.g. in Debian-based Linux distritutions package called nco)
#### Needed static input files
* Digital elevation model
* Shapefile of glacier
### Procedure:
Convert digital elevation model (DEM) to lat lon:
```bash
gdalwrap -t_srs EPSG:4326 input.tif output.tif
```
Subset your studyarea from the DEM with upper left x (longitude) and y (latitude) value and lower right x and y value:
```bash
gdal_translate -r cubicspline -projwin ulx uly lrx lry input.tif output.tif
#example; small area of Hintereisferner
gdal_translate -r cubicspline -projwin 10.74 46.794 10.76 46.79 dem1.tif dem_small.tif #
```
Calculate aspect and slope:
```bash
gdaldem slope dem_small.tif slope.tif
gdaldem aspect dem_small.tif aspect.tif
```
Create glacier mask with shapefile:
```bash
gdalwarp -cutline shapefile.shp DEM.tif mask.tif   
```
Create NC-files from geoTiffs:
 ```bash
 gdal_translate -of NETCDF input.tif output.nc
 ```
Rename variables in netCDF files:
```bash
ncrename -v Band1,HGT dem_small.nc  # example if elevation is called Band1
ncrename -v Band1,ASPECT aspect.nc  # example if aspect is called Band1
ncrename -v Band1,SLOPE slope.nc    # example if slope is called Band1
ncrename -v Band1,MASK mask.nc      # example if boolean mask is called Band1
```
Combine created netCDF files:
```bash
cdo merge *.nc static.nc
```
### Create input file with all needed static and dynamic 2D fields
#### Needed files and parameters
* static.nc file, created in step above
* 1D fields of all required dynamic input files
### Run script
In the utilities folder there is a python script called cs2cosipy.py. This file has a configuration file called cs2cosipyConfig.py. The script can be uses to create 2D fiels from 1D fiels. <br>
For the solar radiation a model after Wohlfahrt et al. (2016; doi: 10.1016/j.agrformet.2016.05.012) is used. <br>
For air temperature, relative humidity and precipitation constant lapse rates, which have to be set, are used. <br>
Wind speed and cloud cover fraction kept constant for all gridpoint at on time stept.<br><br>
The script needes:
* the input file; for example a Campbell Scientific logger file (csv file) with all required dynamic input fiels
* the file path (including the name) for the resulting COSIPY file, which will be used as input file for the core run
* the path to the static file, created in the step above
* the start and end date of the timespan
In the cs2cosipyConfig.py one has to define how the input variables are called in the CS_FILE. <br> 
For the radiation module one has to set the timezone and the zenit threshold. <br> Furthermore, the station name has to be set, the altitude of the station, and the laps rates for temperature, relative humidity and precipitation.<br>
If everything is set, configured and prepared, run the script:
```bash
python cs2cosipy.py -c data/input_1D.dat -o input_core_run.nc -s ../data/static/static.nc -b 2010-01-01T00:00 -e 2010-12-31T23:00
```
## Core run
### Changes config.py and set everything for your specific need. See in config options.

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

# Config options
### describe everything possible!

# Open issues:

* densification
* subsurface melt (penetrating radiation)

# Planed parameterisations and included processes:
* add heat flux from liquid precipitation
* add liquid precipitation to surface melt
* superimposed ice

# 
Please branch or fork your version, do not change the master.

You are allowed to use and modify this code in a noncommercial manner and by
appropriately citing the above mentioned developers.
