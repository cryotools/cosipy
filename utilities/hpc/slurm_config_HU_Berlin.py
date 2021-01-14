"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here. Please add the name of your SLURM accounting project
"""

port = 8786                                                     # With this port, the monitoring webpage can be accessed
processes = 20                                                  # Grid points submitted in one sbatch script
nodes = 1                                                       # Processes multiplied by the number of workers
cores = 1                                                       # One grid point per core, do not change
name = 'Worker'                                                 # Equivalent to slurm parameter --job-name
memory_per_process = 3
memory = memory=str(memory_per_process * processes) + 'GB'      # Memory per processes in GB
queue = 'work'
shebang = '/bin/bash -l'
slurm_parameters = [
                    '--qos=short',                              # Slurm quality of service
                    '--output=Output_nodes.output',	        # Path slurm output file
                    '--error=Error_nodes.err',                  # Path slurm error file
                    '--time=1-00:00:00',                        # Time limit for job
                    '--account='                                # Equivalent to slurm parameter --account
                    ]
