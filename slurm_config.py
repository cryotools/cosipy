"""
 This is the configuration file for usage of the slurm job scheduler.
 Please make your changes here.
"""

port_monitoring = 8786
cores = 1                 ### 1 gridpoint on one core is one job
processes = 1             ### 1 process pro core
memory = '2GB'            ### memoroy per job
slurm_parameters = ['--qos=short', '--job-name="Argog"','--account=prime',
                    '--output=/data/scratch/arndtans/logs/1_COSIPY_Argog.out',
                    '--error=/data/scratch/arndtans/logs/1_COSIPY_Argog.err'
                    ]
slurm_scale = 100
