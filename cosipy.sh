#!/bin/bash -l

# The batch system should use the current directory as working directory.
#SBATCH --job-name=Test_run
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=20
#SBATCH --time=00:10:00

unset SLURM_EXPORT_ENV

module load intel64 netcdf 

/home/woody/gwgk/gwgk01/envs/karoshi/bin/python COSIPY.py

# Clean dask worker files
rm -r worker-* *.lock
