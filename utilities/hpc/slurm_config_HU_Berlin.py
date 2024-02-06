"""
 This is the configuration file for usage of the slurm job scheduler.
  Please make your changes here. Please add the name of your SLURM accounting project
"""

name = 'Worker'                                                 # Equivalent to slurm parameter --job-name
cores = 20                                                      # One grid point per core, do not change
nodes = 1							# Grid points submitted in one sbatch script
account=''							# Equivalent to slurm parameter --account; Slurm account/group
memory_per_process = 3
#memory_per_process = 6
memory = memory=str(cores * memory_per_process) + 'GB'      	# Total memory per submited slurm job
slurm_parameters = [
                    '--qos=short',                              # Slurm quality of service
                    '--output=Output_nodes.output',	        # Path slurm output file
                    '--error=Error_nodes.err',                  # Path slurm error file
                    '--time=1-00:00:00',                        # Time limit for job
                    #'--reservation=COSIPY',                    # Time limit for job
		    #'--partition=computehm'
                    ]
