"""
 This is the configuration file for usage of the slurm job scheduler.
 Please make your changes here.
"""

processes = 20	          				            # grid points submitted in one sbatch script
nodes = 1					                    # how many sbatach scripts are submitted
memory_per_process = 3					            # memory per processes in GB
project = ''                            			    # equivalent to slurm parameter --account
name = 'Test',                                                      # equivalent to slurm parameter --job-name
extra_slurm_parameters = [ '--qos=short', 		            # Slurm quality of service
                    	   '--output=Output_test.out',	            # Path slurm output file
                    	   '--error=Error_test.err',	            # Path slurm error file
                           '--time=1-00:00:00', 		    # Time limit for job
                   	 ]
port_monitoring = 8786				                    # port for monitoring
cores = 1                 			                    # One grid point per core, do not change
