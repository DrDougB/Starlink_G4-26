# -----------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023/2024: Douglas J. Buettner, PhD. GPL-3.0 license
# specific terms of this GPL-3.0 license can be found here:
# https://github.com/DrDougB/Starlink_G4-26/blob/main/LICENSE 
#
#  primary functions: 
#  calculate_heading_from_velocity, twoecef2razel, cockpitview, and twoecef2enu
#
#  calculate_heading_from_velocity calculates the heading in radians from the ECEF velocity  
#
#  twoecef2razel converts two ECEF position and velocity vectors into range, 
#    azimuth, elevation, and rates.  notice the value of small as it can 
#    affect the rate term calculations. the solution uses the velocity vector 
#    to find the singular cases. also, the elevation and azimuth rate terms 
#    are not observable unless the acceleration vector is available.
#
#  cockpitview is similar to twoecef2razel but corrects for the heading
#
#  twoecef2enu converts two ECEF position and velocity vectors into observer relative ENU 
#    coordinates
#
#  orig. author  : david vallado         719-573-2600   22 jun 2002
#  Python rev    : doug buettner                        29 jan 2024
#
#  revisions
#    vallado     - add terms for ast calculation        30 sep 2002
#    vallado     - update for site fixes                 2 feb 2004 
#    buettner    - Converted to Python, included body 
#    relative ECEF 2 ENU modifications and included  
#    heading corrections for a cockpit view              4 feb 2024
#
#  inputs          description                          units
#    r1ecef      - ECEF position vector for body1        km
#    v1ecef      - ECEF velocity vector for body1        km/s
#    r2ecef      - ECEF position vector for body2        km
#    v2ecef      - ECEF velocity vector for body2        km/s
#    latgd       - geodetic latitude for body1          -pi/2 to pi/2 rad
#    lon         - longitude for body1                  -2pi to 2pi rad
#
#  outputs       : are function dependent see each function
#
#  references    :
#    vallado       2007, 268-269, alg 27
#    vallado       2013, 914-915
#
# [rho,az,el]   = twoecef2razel (r1ecef, v1ecef, r2ecef, v2ecef, lat, lon);
# [rho,look,el] =   cockpitview (r1ecef, v1ecef, r2ecef, v2ecef, lat, lon);
#
#
# Change History: Version 1.1, DJB (3/30/24):
#                 Added explicit copyright statements in this revision. Users should consider this 
#                 retroactive to the first version, prior version only had copyright applied at the GitHub
#                 level. This change brings that copyright notice into this code.
#
#                 Version 1.0: Initial version
#
# -----------------------------------------------------------------------------------------------------------------

import numpy as np
import rot as rot
from math import tau

# Set to True if you want verbose function print statements enabled
verbose = False


# Vector dot product - used to calculate the sun grazing angle
def dot_product(a, b):
    if verbose:
       print(f"dot_product...")

    # Compute dot product of two vectors.
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# Vector magnitude - used to calculate the sun grazing angle
def magnitude(v):
    if verbose:
       print(f"magnitude..")

    # Compute magnitude (or norm) of a vector.
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

# DJB: This was the original function I converted and started with
# 
# Input Parameters
# 
# r1ecef tuple: Oberserver at x, y, z in ECEF at time t in (km)
# r2ecef tuple: Target     at x, y, z in ECEF at time t in (km)
# v1ecef tuple: Oberserver at Vx, Vy, Vz in ECEF at time t in (km)
# v2ecef tuple: Target     at Vx, Vy, Vz in ECEF at time t in (km)
# lat Observer geodetic latitude
# lon Observer geodetic longitude
#
# 
# Results
# 
# azimuth   heading adjusted azimuth to target (deg)
# elevation elevation to target                (deg)
# srange    slant range to target              (km)
def twoecef2razel(r1ecef, v1ecef, r2ecef, v2ecef, lat, lon):
    if verbose:
       print(f"twoecef2razel...")

    pi     = np.pi    # 180 degrees
    halfpi = pi * 0.5 #  90 degrees
    twopi  = 2 * pi   # 360 degrees
    small  = 0.00000001

    if lon < -pi:
        lon += twopi
    elif lon > pi:
        lon -= twopi

    # Convert these lists into vector/arrays 
    r2ecef = np.array(r2ecef)
    r1ecef = np.array(r1ecef)
    v2ecef = np.array(v2ecef)
    v1ecef = np.array(v1ecef)

    # Find ECEF range and velocity vectors from "body1" to "body2" 
    rhoecef  = r2ecef - r1ecef
    drhoecef = v2ecef - v1ecef

    rho = np.linalg.norm(rhoecef)

    # Convert to SEZ for calculations
    tempvec = rot.rot3(rhoecef, lon)
    rhosez  = rot.rot2(tempvec, halfpi - lat)

    tempvec = rot.rot3(drhoecef, lon)
    drhosez = rot.rot2(tempvec, halfpi - lat)

    # Calculate azimuth and elevation
    temp = np.sqrt(rhosez[0] * rhosez[0] + rhosez[1] * rhosez[1])

    if temp < small:
        el = (np.sign(rhosez[2]) * halfpi)
    else:
        magrhosez = np.linalg.norm(rhosez)
        el    = np.arcsin(rhosez[2] / magrhosez)
    
    eldeg = el * (180 / pi)

    if temp < small:
        az = np.arctan2(drhosez[1], -drhosez[0]) * (180 / pi)
    else:
        az = np.arctan2(rhosez[1], -rhosez[0]) * (180 / pi)

    return rho, az, eldeg 


# Input Parameters
# 
# vecef  tuple: Oberserver's velocity x, y, z in ECEF at time t in (km)
# lat Observer geodetic latitude
# lon Observer geodetic longitude
# 
# Results
# 
# heading   heading (rad)
#
def ecef2enu(ecef, lat, lon):
    if verbose:
       print(f"ecef2enu...")

    # The vector components in ECEF
    u = ecef[0] 
    v = ecef[1] 
    w = ecef[2] 

    # Convert the UVW coordinates to ENU
    temp  =  np.cos(lon) * u    + np.sin(lon) * v

    east  = -np.sin(lon) * u    + np.cos(lon) * v
    up    =  np.cos(lat) * temp + np.sin(lat) * w
    north = -np.sin(lat) * temp + np.cos(lat) * w

    return east, north, up

# Input Parameters
# 
# vecef  tuple: Oberserver's velocity x, y, z in ECEF at time t in (km)
# lat Observer geodetic latitude
# lon Observer geodetic longitude
# 
# Results
# 
# heading   heading (rad)
#
def calculate_heading_from_velocity(vecef, lat, lon):
    if verbose:
       print(f"calculate_heading_from_velocity...")

    # The velocity components in ECEF
    u = vecef[0] 
    v = vecef[1] 
    w = vecef[2] 

    # Convert the ECEF coordinates to ENU
    temp  =  np.cos(lon) * u + np.sin(lon) * v

    v_east  = -np.sin(lon) * u    + np.cos(lon) * v
    v_north = -np.sin(lat) * temp + np.cos(lat) * w

    # Calculate the heading angle in radians clockwise from north
    heading_radians = np.arctan2(v_east, v_north)

    return heading_radians


# Input Parameters
# 
# acecef  tuple: Oberserver at x, y, z in ECEF at time t in (km)
# satecef tuple: Target     at x, y, z in ECEF at time t in (km)
#
# u  observer to target x ECEF coordinate (km)
# v  observer to target y ECEF coordinate (km)
# w  observer to target z ECEF coordinate (km)
# 
# Results
# 
# East  target east ENU coordinate  (km)
# North target north ENU coordinate (km)
# Up    target up ENU coordinate    (km)
# 
# Adapted from: pymap3d's ecef.py: uvw2enu/ecef2enuv
# 
def twoecef2enu(acecef, satecef, lat, lon):
    if verbose:
       print(f"twoecef2enu...")

    # Convert the ECEF vectors into Aircraft Relative UVW cartesian coordinates
    u = satecef[0] - acecef[0]
    v = satecef[1] - acecef[1]
    w = satecef[2] - acecef[2]

    # Convert the UVW coordinates to ENU
    temp  =  np.cos(lon) * u    + np.sin(lon) * v

    east  = -np.sin(lon) * u    + np.cos(lon) * v
    up    =  np.cos(lat) * temp + np.sin(lat) * w
    north = -np.sin(lat) * temp + np.cos(lat) * w

    return east, north, up


# Input Parameters
# 
# acecef  tuple: Oberserver at x, y, z in ECEF at time t in (km)
# satecef tuple: Target     at x, y, z in ECEF at time t in (km)
# lat Observer geodetic latitude
# lon Observer geodetic longitude
# 
# Results
# 
# azimuth   azimuth to target     (deg)
# elevation elevation to target   (deg)
# srange    slant range to target (km)
# 
# Adapted from: pymap3d's ecef.py: uvw2enu
# 
def twoecef2aer(acecef, satecef, lat, lon):
    if verbose:
       print(f"twoecef2aer...")

    e, n, u = twoecef2enu(acecef, satecef, lat, lon)

    try:
        e[np.abs(e) < 1e-3] = 0.0
        n[np.abs(n) < 1e-3] = 0.0
        u[np.abs(u) < 1e-3] = 0.0
    except TypeError:
        if np.abs(e) < 1e-3:
            e = 0.0
        if np.abs(n) < 1e-3:
            n = 0.0
        if np.abs(u) < 1e-3:
            u = 0.0

    r          = np.hypot(e, n)
    slantRange = np.hypot(r, u)

    elev = np.arctan2(u, r)
    az   = np.arctan2(e, n) % tau

    az   = np.degrees(az)
    elev = np.degrees(elev)

    return slantRange, az, elev

# Input Parameters
# 
# r1ecef tuple: Aircraft oberserver at x,  y,  z in ECEF at time t in (km)
# r2ecef tuple: Observed satellite  at x,  y,  z in ECEF at time t in (km)
# v1ecef tuple: Oberserver's velocity Vx, Vy, Vz in ECEF at time t in (km)
# lat Observer geodetic latitude
# lon Observer geodetic longitude
# 
# Results
# 
# range     slant range to target              (km)
# look      heading adjusted azimuth to target (deg)
# elevation elevation to target                (deg)
# 
# Adapted from: pymap3d's ecef.py: uvw2enu to include the heading adjustment
# 
def cockpitview(r1ecef, v1ecef, r2ecef, lat, lon):
    if verbose:
       print(f"cockpitview...")

     # Convert ECEF locations into a relative ENU vector from "body1" to "body2" 
    pos_enu = twoecef2enu(r1ecef, r2ecef, lat, lon)
    e = pos_enu[0] # East  relative from from "aircraft" to "satellite" 
    n = pos_enu[1] # North relative from from "aircraft" to "satellite" 
    u = pos_enu[2] # Up    relative from from "aircraft" to "satellite" 

    # Heading angle for the aircraft's direction of travel (degrees from North) in radians
    heading_radians = calculate_heading_from_velocity(v1ecef, lat, lon)

    # Rotate the ENU vector to account for the direction of travel of the aircraft.
    # The negative value takes into account our desire for the coordinates of the object
    # to be negative counterclockwise from the pilot's cockpit perspective.  
    e_prime =  e * np.cos(-heading_radians) + n * np.sin(-heading_radians)
    n_prime = -e * np.sin(-heading_radians) + n * np.cos(-heading_radians)
    u_prime = u 
    range   = magnitude([e_prime,n_prime,u_prime])
    horiz_range = magnitude([e_prime,n_prime, 0])  # Used to calculate the elevation angle

    # Now calculate azimuth and elevation adjusted for the aircraft's cockpit perspective
    
    # The cockpit "look angle" relative to the aircraft's heading is ArcTan2 of E'/N'
    look = np.arctan2(e_prime, n_prime)
    lookdeg = np.degrees(look)
    
    # El angle relative to the aircraft's local ENU plane is the ArcTan2 of Up/Horizontal Range
    el = np.arctan2(u_prime, horiz_range)
    eldeg = np.degrees(el)

    return range, lookdeg, eldeg 
