#!/bin/bash -l

# The batch system should use the current directory as working directory.
#SBATCH --job-name=REAL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=03:00:00

module load intel64 netcdf 

export KMP_STACKSIZE=64000000
export OMP_NUM_THREADS=1
ulimit -s unlimited

python aws2cosipy.py -c ../../data/input/Peru/data_aws_peru.csv -o Peru_input.nc -s ../../data/static/Peru_static.nc -xl -77.64709 -xr -77.6098 -yl -8.979938 -yu -8.948105


