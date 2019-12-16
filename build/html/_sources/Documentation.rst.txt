.. _Documentation:


***************
Getting started
***************

.. _requirements:

Requirements
============

Packages and libraries
----------------------

COSIPY should run with any Python 3 version on any operating system. If you think the
reason for a problem might be your specific Python 3 version or your operating
system, please create a topic in the forum. The model is tested and
developed on:

 * Anaconda Distribution on max OS
 * Python 3.6.5 on Ubuntu 18.04
 * Anaconda 3 64-bit (Python 3.6.3) on CentOS Linux 7.4
 * High-Performance Cluster Erlangen-Nuremberg University 

The model requires the following libraries:

 * xarray
 * dask-jobqueue
 * netcdf4
 * numpy (included in Anaconda)
 * scipy (included in Anaconda)
 * distributed (included in Anaconda)


Additional packages (optional):

 * gdal (e.g. in Debian-based Linux distributions package called gdal-bin)
 * climate date operators (e.g. in Debian-based Linux distributions package called cdo)
 * netCDF Operators (e.g. in Debian-based Linux distritutions package called nco)


.. _tutorial:

Quick tutorial
==============

Pre-processing
--------------

COSIPY requires a file with the corresponding meteorological and static input
data. Various tools are available to create the file from simple text or
geotiff files.


.. _static:

Create the static file
~~~~~~~~~~~~~~~~~~~~~~~

In the first step, topographic parameters are derived from the Digital Terrain
Model (DEM) and written to a NetCDF file. A shape file is also required to
delimit the glaciated areas. The DEM and the shapefile should be in lat/lon
WGS84 (EPSG:4326) projection.

.. note:: The DEM can be reprojected to EPSG:4326 using gdal::

           > gdalwarp -t_srs EPSG:4326 dgm_hintereisferner.tif dgm_hintereisferner-lat_lon.tif 


COSIPY comes with the script create_static_file.py located in the utilities folder.
This script runs some gdal routines in the command line. That's is the reason that
we can provide this script only for UNIX and MAC users at the moment.
The script creates some intermediate NetCDF files (dem.nc, aspect.nc,
mask.nc and slope.nc) that are automatically deleted after the static file is created. 

Here we use the DEM **n30_e090_3arc_v2.tif** (SRTM) and the shapefile
**Zhadang_RGI6.shp** provided in the /data/static folder. The static file is
created using::

        python create_static_file.py

The command creates a new file **Zhadang_static.nc** in the /data/static folder.
The file names and paths can be simply changed in the python script.


.. _input:

Create the COSIPY input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The creation of the input file requires the static information (file) from
:ref:`section <static>`. To convert the data from an automatic weather station
(AWS) we use the conversion script aws2cosipy.py located in the folder
/utilities/aws2cosipy. The script comes with a configuration file
aws2cosipyConfig.py which defines the structure of the AWS file and other
user-defined parameter. Since the input file provides point information, the
data is interpolated via lapse rates for two-dimensional runs.  The solar
radiation fields is based on a model suggested by Wohlfahrt et al.  (2016; doi:
10.1016/j.agrformet.2016.05.012).  Other variables as wind velocity and cloud
cover fraction are assumed to be constant over the domain.

.. note:: The script aws2cosipy.py only serves to illustrate how data can be
          prepared for COSIPY. For most applications it is recommended to develop your
          own routine for data interpolation.

The script is executed with

::

        > python aws2cosipy.py / 
          -c ../../data/input/Zhadang/Zhadang_ERA5_2009_2018.csv / 
          -o ../../data/input/Zhadang/Zhadang_ERA5_2009.nc /
          -s ../../data/static/Zhadang_static.nc /
          -b 20090101 -e 20091231

+-----------+-------------+
| Argument  | Description |
+-----------+-------------+
| -c        | meteo file  |
+-----------+-------------+
| -o        | output file |
+-----------+-------------+
| -s        | static file |
+-----------+-------------+
| -b        | start date  |
+-----------+-------------+
| -e        | end date    |
+-----------+-------------+

If the script was executed successfully, the file
/data/input/Zhadang/Zhadang_ERA5_2009_2018.nc should have been created.

.. _run:

Execute the COSIPY model:
~~~~~~~~~~~~~~~~~~~~~~~~~

To run Cosipy, run the following command in the root directory::

        > python COSIPY.py

The example should take about 3-5 minutes on a workstation with 4 cores.

.. note:: **The configuration and definition of the parameters/constants is done
          in config.py and constants.py.**


Visualization
--------------

     
