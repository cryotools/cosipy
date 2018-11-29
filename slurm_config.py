"""
 This is the configuration file for usage of the slurm job scheduler.
 Please make your changes here.
"""

port_monitoring = 8786
cores = 1                 ### 1 gridpoint on one core is one job
processes = 1             ### 1 process pro core

### please adapt the following ###

memory = '2GB'            ### memoroy per job
min_slurm_workers = 1     ### minimum jobs
max_slurm_workers = 20    ### maximum jobs
slurm_parameters = ['--qos=', '--job-name="Hintereisfener"','--account=',
                    '--output=',
                    '--error='
                    ]
