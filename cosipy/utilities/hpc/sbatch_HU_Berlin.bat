#!/bin/bash

## Please replace $WORKING_DIR with the absolute path to your current COSIPY folder and add the name of your accounting project

#SBATCH --job-name="Master"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=$WORKING_DIR
#SBATCH --account=
#SBATCH --error=Control_master.err
##SBATCH --partition=computehm
#SBATCH --output=Control_master.out
##SBATCH --mail-type=ALL

echo $SLURM_CPUS_ON_NODE

module load anaconda/2019.07
python -u $WORKING_DIR/COSIPY.py
