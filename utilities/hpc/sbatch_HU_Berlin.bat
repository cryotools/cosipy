#!/bin/bash

#SBATCH --job-name="MaEraHal"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --workdir=$WORKING_DIR
#SBATCH --account=prime
#SBATCH --error=Control_master.err
#SBATCH --partition=computehm
#SBATCH --output=Control_master.out
#SBATCH --mail-type=ALL

echo $SLURM_CPUS_ON_NODE

export PATH="/nfsdata/programs/anaconda3_201812/bin:$PATH"
python -u $WORKING_DIR/COSIPY.py
