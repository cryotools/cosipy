import math
import numpy as np
import metpy.calc
from metpy.units import units

# The following functions are needed for the radiation method Wohlfahrt 2016
def solarFParallel(lat, lon, timezone_lon, day, hour):
    """ Calculate solar elevation, zenith and azimuth angles
    
    Inputs:
        
        lat             ::  latitude (decimal degree)
        lon             ::  longitude (decimal degree)
        timezone_lon    ::  longitude of standard meridian (decimal degree)
        doy             ::  day of the year (1-366)
        hour            ::  hour of the day (decimal, e.g. 12:30 = 12.5
        
    Outputs:
        
        beta            ::  solar elevation angle (radians)
        zeni            ::  zenith angle (radians)
        azi             ::  solar azimuth angle (radians)
    """

    # Calculate conversion factor degree to radians
    FAC = math.pi / 180.0

    # Solar declinations (radians)
    dec = math.asin(0.39785 * math.sin((278.97 + 0.9856 * day + 1.9165 *
        math.sin((356.6 + 0.9856 * day) * FAC)) * FAC))

    # Day length in hours
    length = math.acos(-1.0 * (math.sin(lat * FAC) * math.sin(dec)) /
        (math.cos(lat * FAC) * math.cos(dec))) / FAC * 2.0/15.0

    # Teta (radians), time equation (hours)
    teta = (279.575 + 0.9856 * day) * FAC
    timeEq = (-104.7 * math.sin(teta) + 596.2 * math.sin(2.0 * teta) + 4.3 *
        math.sin(3.0 * teta) - 12.7 * math.sin(4.0 * teta) - 429.3 *
        math.cos(teta) - 2.0 * math.cos(2.0 * teta) + 19.3 * math.cos(3.0 * teta)) / 3600.0

    # Longitude correction (hours)
    LC = (timezone_lon - lon) / 15.0

    # Solar noon (hours) / solar time (hours)
    solarnoon = 12.0 - LC - timeEq 
    solartime = hour - LC - timeEq

    # Solar elevation
    beta = math.asin(math.sin(lat * FAC) * math.sin(dec) + math.cos(lat * FAC) * 
        math.cos(dec) * math.cos(15.0 * FAC * (solartime-solarnoon)))

    # Zenith angle (radians)
    zeni = math.pi/2.0 - beta
    
    # Azimuth angle (radians)
    azi = math.acos((math.sin(lat * FAC) * math.cos(zeni) - math.sin(dec)))/math.cos(lat*FAC)*math.sin(zeni)
    
    if (solartime < solarnoon):
        azi = azi * -1.0

    return beta, zeni, azi


def Fdif_Neustift(doy, zeni, Rg):
    """ Estimate fraction of diffuse radiation
     from Wohlfahrt et al. (2016) doi: 10.1016/j.agrformet.2016.05.012
    
     the basic logic here is that we use the ratio between incident global
     and potential shortwave radiation to define a clearness index and use
     this to calibrate an empirical function of the diffuse radiation
     fraction, in this case data from Neustift were used for calibration
    
     doy .... day of year (-)
     zeni ... zenith angle of sun (rad)
     Rg ..... incident global (shortwave) radiation (W/m2)
     
     last edit: 23.04.2018, Georg
    """
    So = 1367.0 * (1 + 0.033 * math.cos(2.0 * math.pi * doy / 366.0)) * math.cos(zeni)
    
    CI = Rg / So
    
    # empirical parameters from Wohlfahrt et al. (2016) (Appendix C)
    p1 = 0.1001
    p2 = 4.7930
    p3 = 9.4758
    p4 = 0.2465

    if (CI>50):
        Fdif = p4
    else:
        # Eq. C1 in Wohlfahrt et al. (2016)
        Fdif = math.exp(-math.exp(p1 - (p2 - p3 * CI))) * (1.0 - p4) + p4
    return Fdif


def radCor2D(doy, zeni, azi, angslo, azislo, Rm, zeni_thld):
    """ Correct solar radiation measured horizontally for slope and aspect of
    underlying surface
 
     here first estimate fraction of diffuse radiation and then correct beam
     radiation component for slope and aspect of underlying surface based on
     Ham (2005)
    
     Ham, J.M. (2005) Useful equations and tables in micrometeorology. In: Hatfield, J.L.,
     Baker, J.M., Viney, M.K. (Eds.), Micrometeorology in Agricultural Systems.
     American Society of Agronomy Inc.; Crop Science Society of America Inc.; Soil
     Science Society of America Inc., Madison, Wisconsin, USA, pp. 533560.
    
    last edit: 24.03.2018, Georg
    
    the following inputs are scalars
    doy .... day of year
    zeni ... solar zenith angle (from vertical, rad)
    azi .... solar azimuth angle (rad)
    Rm ..... solar radiation measured horizontally (W/m2)
    zeni_thld ... zenith threshold (deg)
    
    the following inputs/outputs represent spatial data (arrays y and x direction)
    angslo ... slope of pixel 
    azislo ... azimuth of pixel
    Rc ... corrected solar radiation (W/m2)
    """
    
    # Calculate conversion factor degree to radians
    FAC = math.pi / 180.0

    # Derive fraction of diffuse radiation
    Fdif = Fdif_Neustift(doy, zeni, Rm)
    if (zeni > math.radians(zeni_thld)):
        Fdif = 1.0
    
    # Split measured global radiation into beam and diffuse part
    Rb = Rm * (1.0 - Fdif)  # Beam radiation
    Rd = Rm * Fdif          # Diffuse radiation

    # Correct beam component for angle and azimuth of pixels
    cf = (math.cos(zeni) * math.cos(angslo*FAC) + math.sin(zeni) * math.sin(angslo*FAC) * \
            math.cos(azi-(azislo*FAC))) / math.cos(zeni)

    Rc = Rb * cf + Rd
    
    return Rc


def correctRadiation(lat, lon, timezone_lon, doy, hour, angslo, azislo, Rm, zeni_thld):

    beta, zeni, azi = solarFParallel(lat, lon, timezone_lon, doy, hour)
    Rc = radCor2D(doy, zeni, azi, angslo, azislo, Rm, zeni_thld)

    return Rc


# The following functions are needed for radiation method Moelg2009
def solpars(lat):
    """ Calculate time correction due to orbital forcing (Becker 2001)
     and solar parameters that vary on daily basis (Mölg et al. 2003)
     0: day angle (rad); 1: in (deg); 2: eccentricity correction factor;
     3: solar declination (rad); 4: in (deg); 5: sunrise hour angle; 6: day length
    """

    timecorr = np.zeros((366, 4))
    solpars = np.zeros((366, 7))

    for j in np.arange(0, 365):
        # Time correction
        x = 0.9856 * (j + 1) - 2.72
        T2 = -7.66 * math.sin(math.radians(x)) - 9.87 * math.sin(
            2 * math.radians(x) + math.radians(24.99) + math.radians(3.83) * math.sin(math.radians(x)))
        timecorr[j, 0] = j + 1  # Julian Day
        timecorr[j, 1] = x
        timecorr[j, 2] = T2  # Time difference between True Local Time (TLT) and Average Local Time (ALT)
        timecorr[j, 3] = T2 * 15 / 60  # Time difference in deg (15°/h)

        # Solar parameters
        tau = 2 * math.pi * (j) / 365
        solpars[j, 0] = tau
        solpars[j, 1] = tau * 180 / math.pi
        solpars[j, 2] = 1.00011 + 0.034221 * math.cos(tau) + 0.00128 * math.sin(tau) + 0.000719 * math.cos(2*tau) + 0.000077 * math.sin(2 * tau)
        solpars[j, 3] = 0.006918 - 0.399912 * math.cos(tau) + 0.070257 * math.sin(tau) - 0.006758 * math.cos(2*tau) + 0.000907 * math.sin(2 * tau) - 0.002697 * math.cos(3 * tau) + 0.00148 * math.sin(3 * tau)
        solpars[j, 4] = solpars[j, 3] * 180 / math.pi
        solpars[j, 5] = math.acos(-math.tan(lat * math.pi / 180) * math.tan(solpars[j, 3])) * 180 / math.pi
        solpars[j, 6] = 2 / 15 * solpars[j, 5]

    # Duplicate line 365 for years with 366 days
    solpars[365, :] = solpars[364, :]
    timecorr[365, :] = timecorr[364, :]

    return solpars, timecorr


def haversine(lat1, lon1, lat2, lon2):
    """ This function calculates the distance between two points given their longitudes and latitudes
     based on the haversine formula. """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = ((math.sin(delta_lat / 2)) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lon / 2)) ** 2) ** 0.5
    d = 2 * 6371000 * math.asin(a)
    return d


def relshad(dem, mask, lats, lons, solh, sdirfn):
    """ This function calculates the topographic shading based on Mölg et al. 2009

     Input:
           dem:    A DEM of the study region, that also includes surrounding terrain
           mask:   A glacier mask
           lats:   The latitudes
           lons:   The longitudes
           solh:   The solar elevation (degrees)
           sdirfn: The illumination direction (in degrees from north)
     Output:
           illu:   Grid illu containing 0 = shaded, 1 = in sun
    """

    z = dem
    illu = dem * 0.0  # create grid that will be filled
    illu[:, :] = np.nan

    # Define maximum radius (of DEM area) in degreees lat/lon
    rmax = ((np.linalg.norm(np.max(lats) - np.min(lats))) ** 2 + (np.linalg.norm(np.max(lons) - np.min(lons))) ** 2) ** 0.5
    nums = int(rmax * len(lats) / (lats[0] - lats[-1]))

    # Calculate direction to sun
    beta = math.radians(90 - sdirfn)
    dy = math.sin(beta) * rmax  # walk into sun direction (y) as far as rmax
    dx = math.cos(beta) * rmax  # walk into sun direction (x) as far as rmax

    # Extract profile to sun from each (glacier) grid point
    for ilat in np.arange(1, len(lats) - 1, 1):
        for ilon in np.arange(1, len(lons) - 1, 1):
            if mask[ilat, ilon] == 1:
                start = (lats[ilat], lons[ilon])
                targ = (start[0] + dy, start[1] + dx)  # find target position

                # Points along profile (lat/lon)
                lat_list = np.linspace(start[0], targ[0], nums)  # equally spread points along profile
                lon_list = np.linspace(start[1], targ[1], nums)  # equally spread points along profile

                # Don't walk outside DEM boundaries
                lat_list_short = lat_list[(lat_list < max(lats)) & (lat_list > min(lats))]
                lon_list_short = lon_list[(lon_list < max(lons)) & (lon_list > min(lons))]

                # Cut to same extent
                if (len(lat_list_short) > len(lon_list_short)):
                    lat_list_short = lat_list_short[0:len(lon_list_short)]
                if (len(lon_list_short) > len(lat_list_short)):
                    lon_list_short = lon_list_short[0:len(lat_list_short)]

                # Find indices (instead of lat/lon) at closets gridpoint
                idy = (ilat, (np.abs(lats - lat_list_short[-1])).argmin())
                idx = (ilon, (np.abs(lons - lon_list_short[-1])).argmin())

                # Points along profile (indices)
                y_list = np.round(np.linspace(idy[0], idy[1], len(lat_list_short)))
                x_list = np.round(np.linspace(idx[0], idx[1], len(lon_list_short)))

                # Calculate ALTITUDE along profile
                zi = z[y_list.astype(np.int), x_list.astype(np.int)]

                # Calclulate DISTANCE along profile
                d_list = []
                for j in range(len(lat_list_short)):
                    lat_p = lat_list_short[j]
                    lon_p = lon_list_short[j]
                    dp = haversine(start[0], start[1], lat_p, lon_p)
                    d_list.append(dp)
                distance = np.array(d_list)

                # Topography angle
                Hang = np.degrees(np.arctan((zi[1:len(zi)] - zi[0]) / distance[1:len(distance)]))

                if np.max(Hang) > solh:
                    illu[idy[0], idx[0]] = 0
                else:
                    illu[idy[0], idx[0]] = 1

    return illu


def LUTshad(solpars, timecorr, lat, elvgrid, maskgrid, lats, lons, STEP, TCART):
    """ This function calculates the look-up-table for topographic shading for one year.

     Input:
           solpars:   Solar parameters
           timecorr:  Time correction due to orbital forcing
           lat:       Latitude at AWS
           elvgrid:   DEM
           maksgrid:  Glacier mask
           lats:      Latitudes
           lons:      Longitudes
           STEP:      Time step (s)
           TCART:     Time correction due to difference MLT - TLT
     Output:
           shad1yr:   Look-up-table for topographic shading for 1 year
    """

    hour = np.arange(1, 25, 1)
    shad1yr = np.zeros((int(366 * (3600 / STEP) * 24), len(lats), len(lons)))  # Array (time,lat,lon)
    shad1yr[:, :, :] = np.nan

    # Go through days of year
    for doy in np.arange(0, 366, 1):

        soldec = solpars[doy, 3]  # solar declination (rad)
        eccorr = solpars[doy, 2]  # eccenctricy correction factor
        tcorr = timecorr[doy, 3]  # time correction factor (deg)

        # Go through hours of day
        for hod in np.arange(0, 24, int(STEP / 3600)):

            # calculate solar geometries
            stime = 180 + (15 / 2) - hod * 15 - tcorr + TCART
            sin_h = math.sin(soldec) * math.sin(lat * math.pi / 180) + math.cos(soldec) * math.cos(lat * math.pi / 180) * \
                    math.cos(stime * math.pi / 180)
            cos_sol_azi = (sin_h * math.sin(lat * math.pi / 180) - math.sin(soldec)) / math.cos(math.asin(sin_h)) / \
                    math.cos(lat * math.pi / 180)

            if stime > 0:
                solar_az = math.acos(cos_sol_azi) * 180 / math.pi
            else:
                solar_az = math.acos(cos_sol_azi) * 180 / math.pi * (-1)

            solar_h = math.asin(sin_h) * 180 / math.pi

            sdirfn = 180 - solar_az

            # Calculation (1 = in sun, 0 = shaded, -1 = night)
            if sin_h > 0.01:
                illu = relshad(elvgrid, maskgrid, lats, lons, solar_h, sdirfn)
                shad1yr[round(doy * (3600 / STEP) * 24 + (hod * 3600 / STEP)), maskgrid == 1] = illu[maskgrid == 1]
            else:
                shad1yr[round(doy * (3600 / STEP) * 24 + (hod * 3600 / STEP)), maskgrid == 1] = -1.0

    return shad1yr


def LUTsvf(elvgrid, maskgrid, slopegrid, aspectgrid, lats, lons):
    """ This function calculates the look-up-table for the sky-view-factor for one year.

     Input:
           elvgrid:   DEM
           maksgrid:  Glacier mask
           slopegrid: Slope
           aspectgrid:Aspect
           lats:      Latitudes
           lons:      Longitudes
    """

    slo = np.radians(slopegrid)
    asp = np.radians(aspectgrid)
    res = elvgrid * 0
    count = 0

    # Go through all directions (0-360°)
    for azi in np.arange(10, 370, 10):

        # Go through all elevations (0-90°)
        for el in np.arange(2, 90, 2):
            illu = relshad(elvgrid, maskgrid, lats, lons, el, azi)
            a = ((math.cos(np.radians(el)) * np.sin(slo) * np.cos(asp - np.radians(azi))) + (np.sin(np.radians(el)) * np.cos(slo)))
            a[a < 0] = 0
            a[a > 0] = 1
            a[illu == 0] = 0
            res = res + a
            count = count + 1

    vsky = elvgrid * 0
    vsky[:, :] = np.nan
    vsky[maskgrid == 1] = res[maskgrid == 1] / (36 * 44)

    return vsky


def calcRad(solPars, timecorr, doy, hour, lat, tempgrid, pgrid, rhgrid, cldgrid, elvgrid, maskgrid, slopegrid,
            aspectgrid, shad1yr, gridsvf, STEP, TCART):
    """ This function computes the actual calculation of solar Radiation (direct + diffuse)
     including corrections for topographic shading and self-shading, based on Mölg et al. 2009, Iqbal 1983, Hastenrath 1984.

     Input:
           solpars:   Solar parameters
           timecorr:  Time correction due to orbital forcing
           doy:       Day of year
           hour:      Hour of day
           lat:       Latitude at AWSi
           tempgrid:  Air Temperature
           pgrid:     Air Pressure
           rhgrid:    Relative Humidity
           cldgrid:   Cloud fraction
           elvgrid:   DEM
           maksgrid:  Glacier mask
           slopegrid: Slope
           aspectgrid:Aspect
           shad1yr:   LUT topographic shading
           gridsvf:   LUT Sky-view-factor
           STEP:      Time step (s)
           TCART:     Time correction due to difference MLT - TLT

     Output:
           swiasky:   All-sky shortwave radiation
"""

    # Constants
    Sol0 = 1367          # Solar constant (W/m2)
    aesc1 = 0.87764      # Transmissivity due to aerosols at sea level
    aesc2 = 2.4845e-5    # Increase of aerosol transmissivity per meter altitude
    alphss = 0.9         # Aerosol single scattering albedo (Zhao & Li JGR112), unity (zero) -> all particle extinction is due to scattering (absorption)
    dirovc = 0.00        # Direct solar radiation at overcast conditions (as fraction of clear-sky dir. sol. rad, e.g. 10% = 0.1)
    dif1 = 4.6           # Diffuse radiation as percentage of potenial clear-sky GR at cld = 0
    difra = 0.66         # Diffuse radiation constant
    Cf = 0.65            # Constant that governs cloud impact

    soldec = solPars[doy - 1, 3]  # Solar declination (rad)
    eccorr = solPars[doy - 1, 2]  # Cccenctricy correction factor
    tcorr = timecorr[doy - 1, 3]  # Time correction factor (deg)

    # Output files
    swiasky = elvgrid.copy() + np.nan
    swidiff = elvgrid.copy() + np.nan

    # Mixing ratio from RH and Pres
    mixing_interp = metpy.calc.mixing_ratio_from_relative_humidity(rhgrid * units.percent, tempgrid * units.kelvin, pgrid * units.hPa)
    vp_interp = np.array(metpy.calc.vapor_pressure(pgrid * units.hPa, mixing_interp))

    # Solar geometries
    stime = 180 + (STEP / 3600 * 15 / 2) - hour * 15 - tcorr + TCART
    sin_h = math.sin(soldec) * math.sin(lat * math.pi / 180) + math.cos(soldec) * math.cos(
        lat * math.pi / 180) * math.cos(stime * math.pi / 180)
    mopt = 35 * (1224 * (sin_h) ** 2 + 1) ** (-0.5)
    if sin_h < 0:
        mopt = np.nan

    if sin_h > 0.01:  # Calculations are only performed when sun is there

        # Direct & diffuse radiation under clear-sky conditions
        TOAR = Sol0 * eccorr * sin_h
        TAUr = np.exp((-0.09030 * ((pgrid / 1013.25 * mopt) ** 0.84)) * (
                    1.0 + (pgrid / 1013.25 * mopt) - ((pgrid / 1013.25 * mopt) ** 1.01)))
        TAUg = np.exp(-0.0127 * mopt ** 0.26)
        k_aes = aesc2 * elvgrid + aesc1
        k_aes[k_aes > 1.0] = 1.0  # Aerosol factor: cannot be > 1
        TAUa = k_aes ** (mopt)
        TAUaa = 1.0 - (1.0 - alphss) * (1 - pgrid / 1013.25 * mopt + (pgrid / 1013.25 * mopt) ** (1.06)) * (1.0 - TAUa)
        TAUw = 1.0 - 2.4959 * mopt * (46.5 * vp_interp / tempgrid) / ((1.0 + 79.034 * mopt * (46.5 * vp_interp / tempgrid)) **
                                                                      0.6828 + 6.385 * mopt * (46.5 * vp_interp / tempgrid))
        taucs = TAUr * TAUg * TAUa * TAUw

        sdir = Sol0 * eccorr * sin_h * taucs  # Direct solar radiation on horizontal surface, clear-sky
        Dcs = difra * Sol0 * eccorr * sin_h * TAUg * TAUw * TAUaa * (1 - TAUr * TAUa / TAUaa) / (
                    1 - pgrid / 1013.25 * mopt + (pgrid / 1013.25 * mopt) ** (1.02))  # Diffuse solar radiation, clear sky
        grcs = sdir + Dcs  # Potential clear-sky global radiation

        # Correction for slope and aspect (Iqbal 1983)
        cos_zetap1 = (np.cos(np.radians(slopegrid)) * np.sin(np.radians(lat)) - np.cos(np.radians(lat)) * np.cos(np.radians(180 - aspectgrid)) *
                      np.sin(np.radians(slopegrid))) * np.sin(soldec)
        cos_zetap2 = (np.sin(np.radians(lat)) * np.cos(np.radians(180 - aspectgrid)) * np.sin(np.radians(slopegrid)) +
                      np.cos(np.radians(slopegrid)) * np.cos(math.radians(lat))) * np.cos(soldec) * np.cos(stime * np.pi / 180)
        cos_zetap3 = np.sin(np.radians(180 - aspectgrid)) * np.sin(np.radians(slopegrid)) * np.cos(soldec) * np.sin(stime * np.pi / 180)
        cos_zetap = cos_zetap1 + cos_zetap2 + cos_zetap3

        # Clear-sky direct solar radiation at surface (aspect & slope corrected)
        swidir0 = Sol0 * eccorr * cos_zetap * taucs
        swidir0[cos_zetap < 0.0] = 0.0  # self-shaded cells set to 0
        illu = elvgrid * 0.0
        illu = shad1yr[np.int(((doy - 1) * (86400 / STEP)) + (hour / (STEP / 3600))), :, :]
        swidir0[illu == 0.0] = 0.0
        sdir[illu == 0.0] = 0.0

        # Correction for cloud fraction
        swidiff[cldgrid > 0.0] = grcs[cldgrid > 0.0] * (((100 - Cf * 100) - dif1) / 100 * cldgrid[cldgrid > 0.0] +
                                    (dif1 / 100)) * gridsvf[cldgrid > 0.0]  # diffuse amount as percentage of direct rad.
        swidiff[cldgrid == 0.0] = Dcs[cldgrid == 0.0] * gridsvf[cldgrid == 0.0]
        swiasky[:, :] = swidir0 * (1 - (1 - dirovc) * cldgrid) + swidiff  # all-sky solar radiation at surface

    else:
        TOAR = 0.0
        swiasky[maskgrid == 1] = 0 * elvgrid[maskgrid == 1]
        illu = 0.0 * elvgrid - 1

    swiasky_ud = swiasky[::-1, :]
    return swiasky_ud
