"""
 This file run the 2cosipy scripts from without the command line
"""
from utilities.aws_logger2cosipy import create_input

create_input('../data/input/008_station_hintereis_lf_toa5_cr3000_a_small.dat', '../data/input/Hintereisferner_input.nc',
             '../data/static/static.nc',None,None)
