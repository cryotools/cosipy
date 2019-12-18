import math

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

