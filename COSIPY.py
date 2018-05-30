#!/usr/bin/env python

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model in Python' (COSIPY). The model is orientated and inspired
    by the COupled Snowpack and Ice surface energy and MAss balance model (COSIMA)
    which was written in Matlab by Huintjes et al. (2015).

    The Python translation and model improvement of COSIMA was done by
    [in allhabetical order]
    Anselm Arndt
    David Loibl
    Bjoern Sass
    Tobias Sauter

    The python version of the model is subsequently called COSIPY.

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on github. https://github.com/cryotools/cosipy

    For more information read the README and see https://cryo-tools.org/

    The model is written in Python 3.6.3 and is tested on Anaconda3-4.4.7 64-bit.

    Correspondence: anselm.arndt@geo.hu-berlin.de
"""

from core.check_1D_or_distributed_and_run import check_1D_or_distributed_and_run

def main():
    check_1D_or_distributed_and_run()

''' MODEL EXECUTION '''
if __name__ == "__main__":
    main()
