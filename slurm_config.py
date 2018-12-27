"""
 This is the configuration file for usage of the slurm job scheduler.
 Please make your changes here.
"""

port_monitoring = 8786

cores = 1                  ### 1 gridpoint on one core is one job

#################################
###please adapt the following ###
#################################

### to submit every gridpoint as single sbatch job
processes = 1             ### 1 process pro core
memory = '3GB'
min_slurm_workers = 1     ### minimum jobs
max_slurm_workers = 5     ### maximum jobs

### to use only one node with 20 processes
#processes = 20            ### 1 process pro core
#memory = '60GB'
#min_slurm_workers = 1     ### minimum jobs
#max_slurm_workers = 1     ### maximum jobs

slurm_parameters = ['--qos=',
                    '--job-name="Test"',
                    '--account=',
                    '--output=output.out',
                    '--error=error.err',
                    #'--partition=computehm',
		            '--time=7-00:00:00'
                    ]