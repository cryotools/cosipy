
"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here.
  """

port = 8786                                 # with this port, the monitoring webpage can be accessed
processes = 20                              # grid points submitted in one sbatch script
nodes = 20                                  # processes multiplied by the number of workers
cores = 1                                   # One grid point per core, do not change

### example HU Berlin
name = ''                                   # equivalent to slurm parameter --job-name
memory = '3'                                # memory per processes in GB
extra_slurm_parameters = [
                            '--qos=short',                      # Slurm quality of service
                            '--output=Output_test.output',      # Path slurm output file
                            '--error=Error_test.err',           # Path slurm error file
                            '--time=1-00:00:00',                # Time limit for job
                            '--account='                        # equivalent to slurm paarameter --account
                            ]


### example Universität Erlangen
memory = '3'                                  # memory per processes in GB
name = 'Greeland',                            # equivalent to slurm parameter --job-name
queue = 'work'
slurm_parameters = ['--nodes=1','--error=slurm.err','--output=slurm.out','--time=04:00:00', '--shebang=/bin/bash -l']
