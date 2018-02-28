import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
from pathlib import Path

def plots_fluxes(DATA,timestr):
    albedo=DATA.albedo
    condensation=DATA.condensation
    deposition=DATA.deposition
    evaporation=DATA.evaporation
    b=DATA.ground_heat_flux
    lw_in=DATA.longwave_in
    lw_out=DATA.longwave_out
    l=DATA.latent_heat_flux
    mb=DATA.mass_balance
    melt_height=DATA.melt_height
    num_labers=DATA.number_layers
    h=DATA.sensible_heat_flux
    refreezing=DATA.refreezing
    snow_height=DATA.snowHeight
    sw_net=DATA.shortwave_net
    sublimation=DATA.sublimation
    subsurface_melt=DATA.subsurface_melt
    surf_melt=DATA.surface_melt
    surf_temperature=DATA.surface_temperature
    u2=DATA.u2
    sw_in=DATA.sw_in
    T2=DATA.T2
    rH2=DATA.rH2
    snowfall=DATA.snowfall
    pressure=DATA.pressure
    cloud=DATA.cloud
    sh=DATA.sh
    rho=DATA.rho
    Lv=DATA.Lv
    Cs=DATA.Cs
    q0=DATA.q0
    q2=DATA.q2
    qdiff=DATA.qdiff
    phi=DATA.phi
    cpi=DATA.cpi
    names=['albedo','condensation','deposition','evaporation','ground_heat_flux','longwave_in', 'longwave_out', \
     'latent_heat_flux', 'mass_balance', 'melt_heigt', 'number_layers', 'refreezing', 'sensible_heat_flux', 'snow_height', \
     'shortwave_net', 'sublimation', 'subsurface_melt', 'surface_melt', 'surface_temperature', 'wind_speed', 'shortwave_in', 'air_temperature', \
     'relative_humitidy', 'snowfall', 'air_pressure', 'cloud_cover', 'inital_snow_height', 'air_density', \
     'latent_heat_fusion', 'bulk_transfer_coef', 'mixing_ratio_surface', 'mixing_ratio_2m', 'mixing_difference', \
     'stability_parameter', 'specific_heat_ice']
    variables=[albedo, condensation, deposition, evaporation, b, lw_in, lw_out, l, mb, melt_height, num_labers, refreezing, \
        h, snow_height, sw_net, sublimation, subsurface_melt, surf_melt, surf_temperature, u2, sw_in, T2, rH2, \
        snowfall, pressure, cloud, sh, rho, Lv, Cs, q0, q2, qdiff, phi, cpi]
    os.makedirs('output/plots/'+timestr)
    for i in range(0,len(names),1):
        name=names[i]
        variable=variables[i]
        plt_directory='output/plots/'+timestr+'/'
        plot_file=plt_directory+str(name)+'.png'
        plt.plot(variable,linestyle='',marker='o')
        plt.xlabel('iteration (h)')
        plt.ylabel(name)
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()


# # max(melt)
# # idx_meltex=np.where(melt>0.003) ###SWnet+Li+Lo-B-H-L)
# # print(swi[idx_meltex])
# # print(lwi[idx_meltex]+lwo[idx_meltex])
# # print(h[idx_meltex])
# # print(l[idx_meltex])
# # print(b[idx_meltex])
# # print(tsfc[idx_meltex])
# # #plt.show()
# # print(idx_l)
# # print(len(idx_l))
# # print(DATA.L[519])
# # print(DATA)
#
# # plt.plot(DATA.L,linestyle='',marker='o')
# # plt.show()
#
# #from Tkinter import Tk
# #from tkFileDialog import askopenfilename
# #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# #filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# #print(filename)
