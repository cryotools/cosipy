
"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here.
  """

port = 8786
cores = 20                                   # One grid point per core, do not change
processes = 10                                 # grid points submitted in one sbatch script
memory = '32GB'                               # memory per processes in GB
project = 'Greenland'                         # equivalent to slurm parameter --account
name = 'Test',                                # equivalent to slurm parameter --job-name
queue = 'work'
slurm_scale = 200   # processes multiplied by the number of workers
shebang = '/bin/bash -l'
slurm_parameters = [  '--nodes=1', '--time=02:00:00']
