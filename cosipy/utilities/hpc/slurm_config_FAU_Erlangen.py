
"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here.
  """

port = 8786
cores = 1                                   # One grid point per core, do not change
processes = 20                                # grid points submitted in one sbatch script
memory = '30GB'                               # memory per processes in GB
project = 'Peru'                         # equivalent to slurm parameter --account
name = 'Peru',                                # equivalent to slurm parameter --job-name
queue = 'work'
nodes = 1                                   # processes multiplied by the number of workers
shebang = '/bin/bash -l'
slurm_parameters = ['--nodes=1','--error=slurm.err','--output=slurm.out','--time=00:10:00', '-p work']
