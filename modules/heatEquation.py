import numpy as np
from numpy.distutils.command.install_clib import install_clib
import logging

from scipy import sparse
from scipy.sparse.linalg import spsolve

from constants import *
from config import *

import sys

def solveHeatEquation(GRID, T2, t):
    """ Solves the heat equation on a non-uniform grid using

    dt  ::  integration time
    
    """
    # start module logging
    logger = logging.getLogger(__name__)

    curr_t = 0
    Tnew = 0

    nl = GRID.get_number_layers()

    # Calculate thermal conductivity [W m-1 K-1] from mean density
    lam = 0.021 + 2.5 * (np.asarray(GRID.get_density())/1000.0)**2.0

    # Calculate thermal diffusivity [m2 s-1]
    K = lam / (np.asarray(GRID.get_density()) * np.asarray(GRID.get_specific_heat()))
    
    # Get snow layer heights    
    hlayers = np.asarray(GRID.get_height())

    diff = ((hlayers[0:nl-1]/2.0)+(hlayers[1:nl]/2.0))
    hk = diff[0:nl-2] 
    hk1 = diff[1:nl-1]

    # Lagrange coeffiecients
    ak = 2.0 / (hk*(hk+hk1))
    bk = -2.0 / (hk*hk1)
    ck = 2.0 / (hk1*(hk+hk1))

    # Introduce C for matrix
    C1 = np.zeros(nl-1)
    C2 = np.ones(nl)
    C3 = np.zeros(nl-1)

    #C1[0:nl-2] = K[1:nl-1]*(t/ak)
    #C2[1:nl-1] = 1+K[1:nl-1]*(t/bk)
    #C3[1:nl-1] = K[1:nl-1]*(t/ck)
    ak = 2.0 / (hk*(hk+hk1))
    bk = 2.0 / (hk*hk1)
    ck = 2.0 / (hk1*(hk+hk1))
    C1[0:nl-2] = -(2*K[1:nl-1]*t)/(hk*(hk+hk1))
    C2[1:nl-1] = 1+(2*K[1:nl-1]*t)/(hk*hk1)
    C3[1:nl-1] = -(2*K[1:nl-1]*t)/(hk1*(hk+hk1))

    # Create tridiagonal matrix
    A = sparse.diags([C1, C2, C3], [-1,0,1], format='csc')
   
    # Correct matrix for BC
    #A[0,1] = 0; A[0,0] = 1
    #A[nl-1,nl-2] = 0; A[nl-1,nl-1] = 1

    # Initial vector
    b = np.asarray(GRID.get_temperature())

    # Solve equation
    x = spsolve(A,b)

    #print('----------------------------')
    #print(T2)
    #print(x)
    #print(GRID.get_temperature())      
    #GRID.set_temperature(x)
    #print('----------------------------\n')



    
