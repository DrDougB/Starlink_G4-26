# Python code to process comma delimited files output from csvout.py
# This file contains functions to transform Earth-Centered Earth-Fixed (ECEF) into Earth-North-Up (ENU)
# coordinates of aircraft relative views of satellites, launch debris and the sun at specific UTC 
# times photographs were taken of potential Unidentified Aerospace Phenomena (UAP).
# 
# The code also calculates the grazing angle off the satellite from the sun to an aircraft observer. 
# It determines the relative distances between each of the satellites to identify the apparent angular 
# length at a specific UTC time.
# 
# Dr. Doug Buettner generated this code in part with GPT-4, OpenAI’s large-scale language-generation
# model. Upon generating draft language, the author reviewed, edited, and revised the content 
# to his own liking (for example fixing errors) and takes ultimate responsibility for the content.
# 
# OpenAI's Terms of Use statement dated March 14, 2023 (last visited Aug 8th, 2023) is 
# available here: https://openai.com/policies/terms-of-use)
# 
# States the following:
#   (a) Your Content. You may provide input to the Services (“Input”), 
#   and receive output generated and returned by the Services based on the Input 
#   (“Output”). Input and Output are collectively “Content.” As between the parties 
#   and to the extent permitted by applicable law, you own all Input. Subject to your 
#   compliance with these Terms, OpenAI hereby assigns to you all its right, title and 
#   interest in and to Output. This means you can use Content for any purpose, including 
#   commercial purposes such as sale or publication, if you comply with these Terms. OpenAI 
#   may use Content to provide and maintain the Services, comply with applicable law, and 
#   enforce our policies. You are responsible for Content, including for ensuring that it 
#   does not violate any applicable law or these Terms.
#
# Uses code adopted from Michael Hirsch, Ph.D. (pymap3d): https://pypi.org/project/pymap3d/
#
# Change History: Version 1.0 clean code.
#

import numpy as np
import csv
import os
import math

# Add imports from the cockpitview functions here
from cockpitview import cockpitview, twoecef2enu, calculate_heading_from_velocity, ecef2enu

# Change to True if you want to turn on print statements
verbose = False

# Function to extract a value from a given line of the CSV
# Input: line (string)
# Output: float value or None
def extract_value_from_line(line):
    if verbose:
        print(f"extract_value_from_line...")
    try:
        return float(line.split(",")[1].strip())
    except:
        return None


# Function to extract data from a given CSV file
# Apply the log_variables decorator to this function
def extract_data_from_file(file_path):
    if verbose:
        print(f"extract_data_from_file...")
    data = {"X":    None, "Y":    None, "Z":    None, "VX":    None, "VY":    None, "VZ":    None,
            "Xeci": None, "Yeci": None, "Zeci": None, "VXeci": None, "VYeci": None, "VZeci": None,
            "LAT": None, "LON": None, "ALT": None}
    with open(file_path, 'r') as f:
        for line in f:
            if "BCR_POSITION_X" in line:
                data["X"] = extract_value_from_line(line)
            elif "BCR_POSITION_Y" in line:
                data["Y"] = extract_value_from_line(line)
            elif "BCR_POSITION_Z" in line:
                data["Z"] = extract_value_from_line(line)
            elif "BCR_VELOCITY_X" in line:
                data["VX"] = extract_value_from_line(line)
            elif "BCR_VELOCITY_Y" in line:
                data["VY"] = extract_value_from_line(line)
            elif "BCR_VELOCITY_Z" in line:
                data["VZ"] = extract_value_from_line(line)
            elif "BCI_POSITION_X" in line:
                data["Xeci"] = extract_value_from_line(line)
            elif "BCI_POSITION_Y" in line:
                data["Yeci"] = extract_value_from_line(line)
            elif "BCI_POSITION_Z" in line:
                data["Zeci"] = extract_value_from_line(line)
            elif "BCI_VELOCITY_X" in line:
                data["VXeci"] = extract_value_from_line(line)
            elif "BCI_VELOCITY_Y" in line:
                data["VYeci"] = extract_value_from_line(line)
            elif "BCI_VELOCITY_Z" in line:
                data["VZeci"] = extract_value_from_line(line)
            elif "LATITUDE" in line:
                data["LAT"] = extract_value_from_line(line)
            elif "LONGITUDE" in line:
                data["LON"] = extract_value_from_line(line)
            elif "EARTH_ALT_GEODETIC" in line:
                data["ALT"] = extract_value_from_line(line)
    return data

# Function to convert degrees to radians
def deg_to_rad(degrees):
    if verbose:
        print(f"deg_to_rad...")
    return degrees * math.pi / 180.0

# Vector dot product - used to calculate the sun grazing angle
def dot_product(a, b):
    if verbose:
        print(f"dot_product...")
    # Compute dot product of two vectors.
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# Vector cross product - used to calculate the sun grazing angle
def cross_product(a, b):
    if verbose:
        print(f"cross_product...")
    # Compute cross product of two vectors.
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

# Vector magnitude - used to calculate the sun grazing angle
def magnitude(v):
    if verbose:
        print(f"magnitude...")
    # Compute magnitude (or norm) of a vector.
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

# Vector normalization - used to calculate the sun grazing angle
def normalize(v):
    if verbose:
        print(f"normalize...")
    # Normalize a vector, first compute the vector's magnitude
    mag = magnitude(v)
    return [v[0] / mag, v[1] / mag, v[2] / mag]

# Vector transform - used to calculate the sun grazing angle
def transform_vector(matrix, vector):
    if verbose:
        print(f"transform_vector...")
    # Normalize a vector, first compute the vector's magnitude
    # Transform a vector using the tranformation matrix.
    return [
        dot_product(matrix[0], vector),
        dot_product(matrix[1], vector),
        dot_product(matrix[2], vector)
    ]

# 
# Convert Cartesian coordinates to spherical coordinates (rho, theta, phi).
# Parameters:
# r_cartesian (list): A 3-element list representing the Cartesian RSW coordinates [R, S, W].
# Returns:
# tuple: A tuple representing the spherical coordinates (rho, theta, phi).
#        (distance in "km", azimuth-"yaw" in degrees, elevation-"pitch" in degrees)
def cartesian_to_spherical(cartesian):
    if verbose:
        print(f"cartesian_to_spherical...")
    X, Y, Z = cartesian

    # Calculate rho (radial distance)
    rho = math.sqrt(X**2 + Y**2 + Z**2)

    # Calculate theta (azimuth-"yaw" angle)
    # Theta is measured in the XYZ plane from the Y vector towards the X vector
    theta = math.atan2(X, Y)  # atan2 handles division by zero

    # Convert theta from radians to degrees
    theta = math.degrees(theta)

    # Calculate phi (elevation-"pitch" angle)
    # Phi is measured from the Y vector towards the Z vector
    if rho == 0:
        phi = 0  # Avoid division by zero
    else:
        phi = math.asin(Z / rho)

    # Convert phi from radians to degrees
    phi = math.degrees(phi)

    return rho, theta, phi

# Helper function to compute the distance between two 3D points  
# - used to calculate the sun grazing angle
def distance(point1, point2):
    if verbose:
        print(f"distance...")
    return math.sqrt(sum([(point2[i] - point1[i])**2 for i in range(3)]))

# Transforms Earth Centered Inertial coordinates (at a specific time) 
# into the Satellite's NTW (Nadir-Track-Wing) coordinate system where N-axis lies 
# in the orbital plane, T is tangential to the orbit, and W is normal to the
# orbital plane  
# Used to calculate the sun grazing angle in the NTW basis as well as for
# aircraft relative calculations - see below for this basis comparison

def ecef_to_ntw_matrix(r, v):
    if verbose:
        print(f"ecef_to_ntw_matrix...")
    # Compute the ECEF to NTW transformation matrix.
    n_hat = normalize(cross_product(r, v))
    t_hat = normalize(v)
    w_hat = cross_product(t_hat, n_hat)
    return [n_hat, t_hat, w_hat]

# Calculate the angle between two vectors in degrees

def angle_between_vectors(A, B):
    if verbose:
        print(f"angle_between_vectors...")
    # Compute the angle between two vectors in degrees.
    cos_theta = dot_product(A, B) / (magnitude(A) * magnitude(B))
    # Ensure that the value of cos_theta lies between -1 and 1
    # due to potential numerical inaccuracies
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = 180.0 * math.acos(cos_theta)/math.pi
    return theta

# Calculate projection of the satellite vector A in the direction of the plane's velocity B

def parallel_component(A, B):
    if verbose:
        print(f"parallel_component...")
    # Compute the angle between two vectors in degrees.
    scalar = dot_product(A, B) / (magnitude(B) * magnitude(B))
    C = [scalar * B[0],scalar * B[1],scalar * B[2]]
    return C

# Helper routine to transform a vector into a different different basis, e.g. ECEF into RSW or NTW
def transform_to_basis(vector, basis_matrix):
    if verbose:
        print(f"transform_to_basis...")
    return [sum(vector[i] * basis_matrix[j][i] for i in range(3)) for j in range(3)]

# Function to calculate relative values between the aircraft, and the satellite
def calculate_relative_values(aircraft_data, satellite_data, lat, lon):
    if verbose:
        print(f"calculate_relative_values...")
    r_ac_ecef = [aircraft_data["X"], aircraft_data["Y"], aircraft_data["Z"]]
    v_ac_ecef = [aircraft_data["VX"], aircraft_data["VY"], aircraft_data["VZ"]]

    # Transform satellite coordinates to the aircraft's relative ECEF to the NTW frame
    r_satellite = [satellite_data["X"], satellite_data["Y"], satellite_data["Z"]]
    r_satellite_relative = [r_satellite[i] - r_ac_ecef[i] for i in range(3)]
    r_satellite_enu = twoecef2enu(r_ac_ecef, r_satellite, lat, lon)

    sat_enu_rho, sat_enu_theta, sat_enu_phi = cartesian_to_spherical(r_satellite_enu)

    # Transform satellite's velocity to the aircraft's ECEF and ENU frames
    v_satellite = [satellite_data["VX"], satellite_data["VY"], satellite_data["VZ"]]
    v_satellite_relative = [v_satellite[i] - v_ac_ecef[i] for i in range(3)]
    v_satellite_enu = twoecef2enu(v_ac_ecef, v_satellite, lat, lon)

    # Calculating range magnitude
    range_magnitude = magnitude(r_satellite_relative)
    range_mag_enu = magnitude(r_satellite_enu)

    # Calculating velocity magnitude
    velocity_magnitude = magnitude(v_satellite_relative)
    velocity_mag_enu = magnitude(v_satellite_enu)

    return (r_satellite_relative, range_magnitude, r_satellite_enu, range_mag_enu,
            v_satellite_relative, velocity_magnitude, v_satellite_enu, velocity_mag_enu, 
            sat_enu_rho, sat_enu_theta, sat_enu_phi)

# Function to parse through a list of satellites with ECEF values to find the apparent 
# length from the viewpoint of an aircraft observer with ECEF coordinates from the same time
#
#      ^                                  r2->*s8
#      |                                .     *s7
#      |                           .         *s6
#      |                      .           *
#      |                  .              *
#      |              .         
#      |          .                     *
#      |      .                      *s3
#      | .    )Q                    *s2
# (ac)---------------------------->*s1          
#      |                          r1
#      |                 
#
# s1 ... s8 are the hypothetical ECEF coordinates of 9 satellites 
# ac is the aircraft
# Q  is the length angle in degrees between s1 and s6 from the ac pt of view
# H  is heading of the ac, assuming RPY angles are all zero
#
def calculate_apparent_length(satellite_positions, r_ac_ecef, lat, lon):
    if verbose:
        print(f"calculate_apparent_length...")
    max_length_degrees_ecef = 0
    max_length_degrees_enu = 0
    furthest_satellites = {"sat1": None, "sat2": None, "distance": 0, "midpoint": None, "midpoint_sat": 0}

    # Consolidated loop for identifying furthest satellites and calculating angular lengths
    for i in range(len(satellite_positions)):
        for j in range(i + 1, len(satellite_positions)):
            dist = distance(satellite_positions[i]["position"], satellite_positions[j]["position"])
            
            # Update furthest satellites if a new max distance is found
            if dist > furthest_satellites["distance"]:
                furthest_satellites["distance"] = dist
                furthest_satellites["sat1"] = i
                furthest_satellites["sat2"] = j
                furthest_satellites["midpoint"] = [(satellite_positions[i]["position"][k] + satellite_positions[j]["position"][k]) / 2 for k in range(3)]

            vec_a_ecef = [satellite_positions[i]["position"][k] - r_ac_ecef[k] for k in range(3)]
            vec_b_ecef = [satellite_positions[j]["position"][k] - r_ac_ecef[k] for k in range(3)]

            # Transform to ENU coordinates
            vec_a_enu = ecef2enu(vec_a_ecef, lat, lon)
            vec_b_enu = ecef2enu(vec_b_ecef, lat, lon)

            dot_prod_ecef = dot_product(vec_a_ecef, vec_b_ecef)
            magnitude_a_ecef = magnitude(vec_a_ecef)
            magnitude_b_ecef = magnitude(vec_b_ecef)

            dot_prod_enu = dot_product(vec_a_enu, vec_b_enu)
            magnitude_a_enu = magnitude(vec_a_enu)
            magnitude_b_enu = magnitude(vec_b_enu)

            # Before calling acos, clamp the value to be within -1 to 1
            if magnitude_a_ecef * magnitude_b_ecef != 0:
                cos_theta_ecef = dot_prod_ecef / (magnitude_a_ecef * magnitude_b_ecef)
                cos_theta_ecef_clamped = max(min(cos_theta_ecef, 1), -1)  # Clamping
                theta_ecef = math.acos(cos_theta_ecef_clamped)
                length_degrees_ecef = theta_ecef * (180 / math.pi)
                max_length_degrees_ecef = max(max_length_degrees_ecef, length_degrees_ecef)

            if magnitude_a_enu * magnitude_b_enu != 0:
                cos_theta_enu = dot_prod_enu / (magnitude_a_enu * magnitude_b_enu)
                cos_theta_enu_clamped = max(min(cos_theta_enu, 1), -1)  # Clamping
                theta_enu = math.acos(cos_theta_enu_clamped)
                length_degrees_enu = theta_enu * (180 / math.pi)
                max_length_degrees_enu = max(max_length_degrees_enu, length_degrees_enu)

    # After identifying the furthest satellites, find the satellite closest to the midpoint
    closest_distance = float('inf')
    for idx, satellite in enumerate(satellite_positions):
        dist = distance(satellite["position"], furthest_satellites["midpoint"])
        if dist < closest_distance:
            closest_distance = dist
            furthest_satellites["midpoint_sat"] = idx

    # Check if sat1 and sat2 are not None
    if furthest_satellites["sat1"] is not None and furthest_satellites["sat2"] is not None:
        # Calculate the distances from the aircraft to the furthest satellites
        distance_aircraft_to_sat1 = distance(satellite_positions[furthest_satellites["sat1"]]["position"], r_ac_ecef)
        distance_aircraft_to_sat2 = distance(satellite_positions[furthest_satellites["sat2"]]["position"], r_ac_ecef)

        # Determine the length in kilometers as observed by the aircraft
        distanceDiff_kms = abs(distance_aircraft_to_sat1 - distance_aircraft_to_sat2)

        # Calculate the distances from the furthest satellites to the midpoint satellite
        distance_sat1_to_mid = distance(satellite_positions[furthest_satellites["sat1"]]["position"], satellite_positions[furthest_satellites["midpoint_sat"]]["position"])
        distance_sat2_to_mid = distance(satellite_positions[furthest_satellites["sat2"]]["position"], satellite_positions[furthest_satellites["midpoint_sat"]]["position"])

        return max_length_degrees_ecef, max_length_degrees_enu, distanceDiff_kms, distance_sat1_to_mid, distance_sat2_to_mid, furthest_satellites
    else:
        return None, None, None, None, None, None

# Function to calculate the sun's grazing angle with respect to the aircraft observer
#
#         +J            (sat) is the location of the satellite (coordinate origin)
# *        ^             *    is the sun
#  \       |             ac   is the aircraft
#   \      |      ac     Q    is theta - is the angle between * and ac from the sat pt of view (POV)
#    \     |     /       a    is alpha - is grazing angle between the -I axis and * from the sat POV
#     \   .|. Q /        q1   is the angle between *  and +J axis (this is the incident  BRDF light source angle)
#      \.  |  ./         q2   is the angle between ac and +J axis (this is the reflected BRDF angle)
#   a  .\q1|q2/          
#     .  \ | /           I and J maps into either RSW or NTW depending on the transformation  
# <--------+-------->+I  NOTE: Geometry here assumes I and J DO NOT necessarily exactly aligns  
#        (sat)                 with the NTW axis, hence the computation for the grazing angle
#                              uses a=(180-Q)/2.

def calculate_sun_grazing_angle(r_ac_ecef, v_ac_ecef, r_sun_ecef, v_sun_ecef, r_satellite, v_satellite):
    if verbose:
        print(f"calculate_sun_grazing_angle...")
    # Calculate the tranformation matrix to NTW coordinates
    ecef_to_ntw = ecef_to_ntw_matrix(r_satellite, v_satellite)

    # Transform coordinates of both objects to satellite's NTW frame
    position_sun_NTW      = transform_vector(ecef_to_ntw, r_sun_ecef)
    position_aircraft_NTW = transform_vector(ecef_to_ntw, r_ac_ecef)

    # Compute the angle between the two objects as viewed from the satellite in NTW frames
    theta_NTW = angle_between_vectors(position_sun_NTW, position_aircraft_NTW)
    
    # Return the sun's grazing angle in degrees from both coordinate systems
    # These "should" be about the same (see Note above on this calculation).
    graze_angle_NTW = (180-theta_NTW)/2

    return graze_angle_NTW 


# Main function to execute the process
def main():

    print(f"Beginning processing csv files...")

    # Walk the subdirectories to find those with csv files matching our saved SOAP data
    for subdir, _, files in os.walk('.'):
        output_data = []
        length_data = []
        if verbose:
            print(f"Processing files in directory: {subdir}")

        aircrafts  = [file for file in files if file.startswith("ACA") and file.endswith(".csv")]
        satellites = [file for file in files if file[0].isdigit() and file.endswith(".csv")]
        # Identify the sun file in the directory (if it exists)
        sun_file = next((file for file in files if "sun" in file.lower() and file.endswith(".csv")), None)

        # For each file aircraft identified in the directory, extract the data from it
        for aircraft in aircrafts:
            if verbose:
                print(f"Processing aircraft data from: {aircraft}")

            # Extract this aircraft's position and velocity vectors in ECEF and ECI
            data_aircraft = extract_data_from_file(os.path.join(subdir, aircraft))
            r_ac_ecef = [data_aircraft["X"], data_aircraft["Y"], data_aircraft["Z"]]             # BCR is ECR is ECEF
            v_ac_ecef = [data_aircraft["VX"], data_aircraft["VY"], data_aircraft["VZ"]]          # BCR is ECR is ECEF
            r_ac_eci  = [data_aircraft["Xeci"], data_aircraft["Yeci"], data_aircraft["Zeci"]]    # BCI is ECI
            v_ac_eci  = [data_aircraft["VXeci"], data_aircraft["VYeci"], data_aircraft["VZeci"]] # BCI is ECI
            # Extract and Convert latitude and longitude from degrees to radians
            lat_rad = deg_to_rad(data_aircraft["LAT"])
            lon_rad = deg_to_rad(data_aircraft["LON"])
            # Extract altitude (km)
            alt = data_aircraft["ALT"]

            heading_radians = calculate_heading_from_velocity(v_ac_ecef, lat_rad, lon_rad)
            heading = np.degrees(heading_radians)

            satellite_positions = []

            # Initialize variables for sun data
            sun_rho, sun_theta, sun_phi = None, None, None

            # Check if sun data is available. Extract sun's BCI position data from the file containing it 
            # and then also calculate sun's pitch and yaw angles in NTW
            if sun_file:
                if verbose:
                    print(f"Processing sun data from: {sun_file}")
                sun_data = extract_data_from_file(os.path.join(subdir, sun_file))
                r_sun_ecef = [sun_data["X"],  sun_data["Y"],  sun_data["Z"]]           # Extract sun's ECEF position
                v_sun_ecef = [sun_data["VX"], sun_data["VY"], sun_data["VZ"]]          # Extract sun's ECEF velocity
                r_sun_eci  = [sun_data["Xeci"],  sun_data["Yeci"],  sun_data["Zeci"]]  # Extract sun's ECI position
                v_sun_eci  = [sun_data["VXeci"], sun_data["VYeci"], sun_data["VZeci"]] # Extract sun's ECI velocity

                rel_pos_sun_from_aircraft = np.array(r_sun_ecef) - np.array(r_ac_ecef)

                # Append sun data to output_data
                output_data.append([
                    aircraft, "Sun",
                    "Relative position ECEF-XYZ (km)", *rel_pos_sun_from_aircraft, 
                ])

            # For each file satellite or debris file identified in the directory, extract the data from it
            # This for loop structure is not terribly efficient for directories with more than one aircraft
            # If we add in other aircraft in the future, we can move this loop outside of the aircraft loop 
            for satellite in satellites:
                if verbose:
                    print(f"Processing satellite data from: {aircraft, satellite}")

                data_satellite = extract_data_from_file(os.path.join(subdir, satellite))
                # Call the calculate_relative_values function
                (r_sat_ecef_rel, range_mag_ecef, r_sat_enu, range_mag_enu, 
                v_sat_ecef_rel, vel_mag_ecef, v_sat_enu, vel_mag_enu, 
                sat_enu_rho, sat_enu_theta, sat_enu_phi) = calculate_relative_values(data_aircraft, data_satellite, lat_rad, lon_rad)

                # Extract out the ECEF coordinates for this satellite
                r_sat_ecef = [data_satellite["X"],  data_satellite["Y"],  data_satellite["Z"]]
                v_sat_ecef = [data_satellite["VX"], data_satellite["VY"], data_satellite["VZ"]]

                graze_angle_NTW = calculate_sun_grazing_angle(r_ac_ecef, v_ac_ecef, r_sun_ecef, v_sun_ecef, r_sat_ecef, v_sat_ecef)

                # Call the cockpitview function (r1ecef, v1ecef, r2ecef, lat, lon)
                result_cockpitview = cockpitview(r_ac_ecef, v_ac_ecef, r_sat_ecef, lat_rad, lon_rad)

                if verbose:
                    print(f"Appending satellite to satellite_positions data: {satellite}")

                satellite_info = {
                    "filename": satellite,
                    "position": r_sat_ecef
                }
                satellite_positions.append(satellite_info)

                # Append new data to output_data
                output_data.append([
                    aircraft, satellite, graze_angle_NTW, heading, 
                    *r_sat_enu, range_mag_enu,  
                    *v_sat_enu, vel_mag_enu,  
                    *result_cockpitview
                ])

            # Now for this aircraft, calculate the apparent size and identify the satellites
            if satellite_positions:
                if verbose:
                    print(f"Calculating apparent length for the satellite train")

                length_ecef_deg, length_enu_deg, distanceDiff_kms, distance_sat1_to_mid, distance_sat2_to_mid, furthest_satellites_data = calculate_apparent_length(satellite_positions, r_ac_ecef, lat_rad, lon_rad)

                if furthest_satellites_data is not None:
                    length_entry = {
                        "aircraft": aircraft,
                        "furthest_satellites": (
                            satellite_positions[furthest_satellites_data["sat1"]]["filename"],
                            satellite_positions[furthest_satellites_data["sat2"]]["filename"]
                        ),
                        "midpoint_satellite":   satellite_positions[furthest_satellites_data["midpoint_sat"]]["filename"],
                        "length_ecef_degrees":  length_ecef_deg, 
                        "length_enu_degrees":   length_enu_deg,
                        "distanceDiff_kms":     distanceDiff_kms,
                        "distance_sat1_to_mid": distance_sat1_to_mid,
                        "distance_sat2_to_mid": distance_sat2_to_mid
                    }
                    length_data.append(length_entry)
                else:
                    length_entry = {
                    "aircraft": aircraft,
                    "furthest_satellites": (None, None),
                    "midpoint_satellite": None,
                    "length_ecef_degrees": None, 
                    "length_enu_degrees": None,
                    "distanceDiff_kms": None,
                    "distance_sat1_to_mid": None,
                    "distance_sat2_to_mid": None
                    }
                    length_data.append(length_entry)

            else:
                print(f"NO satellite positions to append.")
                length_entry = {
                    "aircraft": aircraft,
                    "furthest_satellites": (None, None),
                    "midpoint_satellite": None,
                    "length_ecef_degrees": None, 
                    "length_enu_degrees": None,
                    "distanceDiff_kms": None,
                    "distance_sat1_to_mid": None,
                    "distance_sat2_to_mid": None
                }
                length_data.append(length_entry)

        # ONLY write output.csv files in the subdirectories
        if output_data or length_data:
            # Writing the results to the output file
            with open(os.path.join(subdir, 'output.csv'), 'w', newline='') as csvfile:
                print(f"Writing output.csv file: {csvfile.name}")
                writer = csv.writer(csvfile)

                # Write headers for length data
                writer.writerow([
                    "Aircraft", "Satellite 1", "Satellite 2", "Midpoint Satellite",
                    "Apparent Length AC ECEF (degrees)", "Apparent Length AC NTW (degrees)", 
                    "Sat Distance Diff AC POV (km)", "Distance from Sat1 to Midpoint (km)", "Distance from Sat2 to Midpoint (km)"
                ])

                # Write the data rows for length data
                for data in length_data:
                    writer.writerow([
                        data["aircraft"], data["furthest_satellites"][0], data["furthest_satellites"][1],
                        data["midpoint_satellite"], data["length_ecef_degrees"], data["length_enu_degrees"], 
                        data["distanceDiff_kms"], data["distance_sat1_to_mid"], data["distance_sat2_to_mid"]
                    ])

                writer.writerow([
                    "Aircraft", "Satellite", "Solar grazing angle (deg)", "AC heading (deg)", 
                    "Rel. Pos. X_ENU (km)", "Rel. Pos. Y_ENU (km)", "Rel. Pos. Z_ENU (km)", "Range ENU (km)", 
                    "Rel. Vel. X_ENU (km/s)", "Rel. Vel. Y_ENU (km/s)", "Rel. Vel. Z_ENU (km/s)", "Rel. Vel. ENU (km/s)", 
                    "Cockpit: range (km)", "Cockpit: look angle (deg)", "Cockpit: elevation angle (deg)"
                ])

                # Write the data rows for satellite and sun data
                writer.writerows(output_data)

    print(f"Done processing all csv files in all subdirectories...")

if __name__ == "__main__":
    main()
