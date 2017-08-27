#!/usr/bin/env python

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model' (COSIMA). The model is originally written
    and developed in Matlab code by Huintjes et al. (2015).

    The Python translation and model improvement of COSIMA was done by
    [in allhabetical order]
    Anselm Arndt
    David Loibl
    Bjoern Sass
    Tobias Sauter

    The python version of the model is subsequently called COSIPY.

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on bitbucket. For more information read the README.

    The model is written in Python 3.6 and is tested on Anaconda2-4.4.0 64-bit.

    Correspondence: anselm.arndt@geo.hu-berlin.de
"""

from core.cosima import cosima


def main():
    cosima()

''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()