
"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here.
  """

port = 8786                                                     # with this port, the monitoring webpage can be accessed
processes = 20                                                  # grid points submitted in one sbatch script
nodes = 1                                                       # processes multiplied by the number of workers
cores = 1                                                       # One grid point per core, do not change
name = 'WoEraHal'                                               # equivalent to slurm parameter --job-name
memory_per_process = 3
memory = memory=str(memory_per_process * processes) + 'GB'      # memory per processes in GB
queue = 'work'
shebang = '/bin/bash -l'
slurm_parameters = [
                    '--qos=short',                              # Slurm quality of service
                    '--output=Output_nodes.output',	            # Path slurm output file
                    '--error=Error_nodes.err',                  # Path slurm error file
                    '--time=1-00:00:00',                        # Time limit for job
                    '--account='                                # equivalent to slurm parameter --account
                    ]
