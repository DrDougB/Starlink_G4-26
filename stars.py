# -----------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023/2024: Douglas J. Buettner, PhD. GPL-3.0 license
# specific terms of this GPL-3.0 license can be found here:
# https://github.com/DrDougB/Starlink_G4-26/blob/main/LICENSE 
#
# Python code to process comma delimited aircraft files output from csvout.py
# This file contains functions to transform Earth-Centered Earth-Fixed (ECEF) into Earth-North-Up (ENU)
# coordinates for aircraft relative views of stars at specific UTC times photographs were taken of
# potential Unidentified Aerospace Phenomena (UAP).
# 
# These calculations help to locate the UAP if stars were captured in the photographs.
# 
# Dr. Doug Buettner generated this code in part with GPT-4, OpenAI's large-scale language-generation
# model. Upon generating draft language, the author reviewed, edited, and revised the content 
# to his own liking (for example fixing errors) and takes ultimate responsibility for the content.
# 
# OpenAI's Terms of Use statement dated March 14, 2023 (last visited Aug 8th, 2023) is 
# available here: https://openai.com/policies/terms-of-use)
# 
# States the following:
#   (a) Your Content. You may provide input to the Services ("Input"), 
#   and receive output generated and returned by the Services based on the Input 
#   ("Output"). Input and Output are collectively "Content." As between the parties 
#   and to the extent permitted by applicable law, you own all Input. Subject to your 
#   compliance with these Terms, OpenAI hereby assigns to you all its right, title and 
#   interest in and to Output. This means you can use Content for any purpose, including 
#   commercial purposes such as sale or publication, if you comply with these Terms. OpenAI 
#   may use Content to provide and maintain the Services, comply with applicable law, and 
#   enforce our policies. You are responsible for Content, including for ensuring that it 
#   does not violate any applicable law or these Terms.
#
# Change History: Version 1.1, DJB (3/30/24):
#                 Added explicit copyright statements in this revision. Users should consider this 
#                 retroactive to the first version, prior version only had copyright applied at the GitHub
#                 level. This change brings that copyright notice into this code.
#
#                 Version 1.0: Initial version
# -----------------------------------------------------------------------------------------------------------------

import csv
import os
import math
import datetime
import numpy as np

from vallado import radec2azel
from cockpitview import ecef2enu, calculate_heading_from_velocity

# Table of stars with their names and coordinates
# Format:
# "Constellation Name", "Star's Name", RA decimal degrees, Decd ecimal degrees), # Reference Used
stars = [# Wolfram Research (2014), ConstellationData, Wolfram Language \function, https://reference.wolfram.com/language/ref/ConstellationData.html
    ("Gemini", "Pollux", 116.32896, 28.02620), # REF: ConstellationData[] Wolfram|Alpha Knowledgebase, 2024 - All Gemini
    ("Gemini", "Castor", 113.6495,  31.8883), 
    ("Gemini", "Alhena", 99.4280, 16.39928), 
    ("Gemini", "Castor B", 113.6504, 31.8885), 
    ("Gemini", "Tejat", 95.7399, 22.5139), 
    ("Gemini", "Mebsuta", 100.9830261,   25.1311254), 
    ("Gemini", "Propus", 93.719355, 22.506787), 
    ("Gemini", "Alzirr", 101.322351, 12.8955920), 
    ("Gemini", "Wasat", 110.030727,  21.982304), 
    ("Gemini", "kappa-Gem", 116.111890, 24.3979965), 
    ("Gemini", "lambda-Gem", 109.523245, 16.540386), 
    ("Gemini", "Theta-Gem", 103.197, 33.9614), 
    ("Gemini", "Iota-Gem", 111.4316471, 27.7980813), 
    ("Gemini", "Mekbuda", 106.0272135, 20.5702985), 
    ("Gemini", "Upsilon-Gem", 113.980625, 26.895744), 
    ("Gemini", "Nu-Gem", 97.240776, 20.2121349), 
    ("Gemini", "Rho-Gem", 112.27800, 31.784549), 
    ("Gemini", "Sigma-Gem", 115.828029, 28.8835117), 
    ("Gemini", "Tau-Gem", 107.7848763, 30.2451638), 
    ("Gemini", "Jishui", 114.7913944, 34.5843063), 
    ("Gemini", "Chi-Gem", 120.879582, 27.794351), 
    ("Gemini", "Phi-Gem", 118.3742007, 26.7657814), 
    ("Gemini", "Pi-Gem", 116.8763493, 33.4156969), 
    ("Gemini", "Omega-Gem", 105.6032500, 24.215446),
    ("Lynx", "alpha-Lyn", 140.2637509, 34.3925593), # REF: ConstellationData[] Wolfram|Alpha Knowledgebase, 2024, just alpha-Lyn
    ("Lynx", "Gaia", 140.091, 36.6993),             # REF: https://stellarium-web.org/
    ("Lynx", "HD 77912", 137.021, 35.3547),         # REF: https://stellarium-web.org/
    ("Lynx", "10 Ursae Maj", 135.556, 41.6868),     # REF: https://stellarium-web.org/
    ("Lynx", "Alsciaukat", 126.127, 43.1102),       # REF: https://stellarium-web.org/
    ("Lynx", "21 Lyncis", 112.139, 49.164),         # REF: https://stellarium-web.org/
    ("Lynx", "15 Lyncis", 104.848, 58.392),         # REF: https://stellarium-web.org/
    ("Lynx", "2 Lyncis", 95.4437, 59.0035),         # REF: https://stellarium-web.org/
    ("Auriga", "Capella", 79.17233, 45.99799),      # REF: ConstellationData[] Wolfram|Alpha Knowledgebase, 2024, everything else
    ("Auriga", "Menkalinan", 89.882179, 44.9474326), 
    ("Auriga", "Mahasim", 89.9301, 37.2128), 
    ("Auriga", "Hassaleh", 74.248417, 33.1660938), 
    ("Auriga", "Almaaz", 75.4922265, 43.8233103), 
    ("Auriga", "Haedus", 76.6287224, 41.2344758), 
    ("Auriga", "Saclateni", 75.619531, 41.075839), 
    ("Auriga", "Delta Aurigae", 89.881743, 54.284738), 
    ("Auriga", "Nu Aurigae", 87.872373, 39.148524), 
    ("Auriga", "Pi Aurigae", 89.983741, 45.936735), 
    ("Auriga", "Kappa Aurigae", 93.844538, 29.4980767), 
    ("Auriga", "Tau Aurigae", 87.2934927, 39.1810730), 
    ("Auriga", "Lambda Aurigae", 79.7853144, 40.0990514), 
    ("Auriga", "Chi Aurigae", 83.181971, 32.1920209), 
    ("Auriga", "Upsilon Aurigae", 87.7601554, 37.3055697), 
    ("Auriga", "Psi 2 Aurigae", 99.8326135, 42.4888766), 
    ("Auriga", "Mu Aurigae", 78.357175, 38.484499), 
    ("Auriga", "Psi 1 Aurigae", 96.2245898, 49.2878920), 
    ("Auriga", "Omega Aurigae", 74.8142057, 37.8902448), 
    ("Auriga", "Xi Aurigae", 88.7115289, 55.7069667), 
    ("Auriga", "Psi 7 Aurigae", 102.6914303, 41.7812305), 
    ("Auriga", "Sigma Aurigae", 81.1630895, 37.3853491), 
    ("Auriga", "Psi 4 Aurigae", 100.7707170, 44.5244495), 
    ("Auriga", "Phi Aurigae", 81.9120159, 34.4758797), 
    ("Auriga", "Psi 6 Aurigae", 101.9148995, 48.7894844), 
    ("Auriga", "Rho Aurigae", 80.4517356, 41.8045720), 
    ("Auriga", "Psi 5 Aurigae", 101.6847372, 43.5774246), 
    ("Auriga", "Psi 3 Aurigae", 99.7049183, 39.9025595), 
    ("Auriga", "Omicron Aurigae", 86.475175, 49.8262710), 
    ("Auriga", "Psi 9 Aurigae", 104.1335749, 46.2739988), 
    ("Auriga", "Psi 8 Aurigae", 103.4878043, 38.5050191),
    ("Ursa Major", "Alioth", 193.5072900, 55.9598230), 
    ("Ursa Major", "Dubhe", 165.93196, 61.75103), 
    ("Ursa Major", "Alkaid", 206.885157, 49.313267), 
    ("Ursa Major", "Mizar", 200.98142, 54.92535), 
    ("Ursa Major", "Merak", 165.4603189, 56.3824261), 
    ("Ursa Major", "Phecda", 178.457697, 53.694760), 
    ("Ursa Major", "Psi-Ursae Maj", 167.4158695, 44.4984867), 
    ("Ursa Major", "Tania Australis", 155.582199, 41.499538), 
    ("Ursa Major", "Talitha", 134.801890, 48.041826), 
    ("Ursa Major", "Theta-Ursae Maj", 143.2143079, 51.6773003), 
    ("Ursa Major", "Megrez", 183.8565026, 57.0326154), 
    ("Ursa Major", "Muscida", 127.5661257, 60.7181682), 
    ("Ursa Major", "Tania Borealis", 154.274080, 42.914414), 
    ("Ursa Major", "Alula Borealis", 169.6197360, 33.0943085), 
    ("Ursa Major", "Alkaphrah", 135.90637, 47.15652), 
    ("Ursa Major", "Taiyangshou", 176.5125586, 47.7794063), 
    ("Ursa Major", "Upsilon-Ursae Maj", 147.7473252, 59.0387336), 
    ("Ursa Major", "Mizar B", 200.984675, 54.921809), 
    ("Ursa Major", "Alula Australis", 169.5455, 31.5293), 
    ("Ursa Major", "Phi-Ursae Maj", 148.02648, 54.06433), 
    ("Ursa Major", "Pi-2 Ursae Maj", 130.0534049, 64.3279362), 
    ("Ursa Major", "Omega-Ursae Maj", 163.4947538, 43.1899575), 
    ("Ursa Major", "Tau-Ursae Maj", 137.729, 63.5138), 
    ("Ursa Major", "Rho-Ursae Maj", 135.6362145, 67.6296174), 
    ("Ursa Major", "Sigma-2 Ursae Maj", 137.59811, 67.13402), 
    ("Ursa Major", "Xi-Ursae Maj B", 169.545150, 31.529117), 
    ("Ursa Major", "Sigma-1 Ursae Maj", 137.0979153, 66.8732339), 
    ("Ursa Major", "Pi-1 Ursae Maj", 129.7987692, 65.0209064)
]

# Function to convert RA from hours, minutes, seconds to decimal degrees
def hms_to_decimal(hours, minutes, seconds):
    return 15 * (hours + minutes / 60 + seconds / 3600)

# Function to convert Dec from degrees, minutes, seconds to decimal degrees
def dms_to_decimal(degrees, minutes, seconds):
    sign = 1 if degrees >= 0 else -1
    return sign * (abs(degrees) + minutes / 60 + seconds / 3600)

# Function to extract a single value from a given line of the CSV
def extract_value_from_line(line):
    try:
        return float(line.split(",")[1].strip())
    except:
        return None

# Function to extract a formatted time value from a line with UTC in the CSV
# Returns the Vallado and astropy compatible datetime format
def extract_datetime_from_line(line):
    try:
        if "UTC" in line:
            parts = line.split(",")
            date_str = parts[0].strip()
            time_str = parts[1].strip()
            
            # Combine date and time into a datetime object
            datetime_str = f"{date_str} {time_str}"
            datetime_obj = datetime.datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S.%f")
            
            # No need to format datetime as string; return the datetime object directly
            return datetime_obj
    except Exception as e:
        print(f"Error extracting datetime: {e}")
        return None

# Function to extract data from a given CSV file
def extract_data_from_file(file_path):
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
            elif "UTC" in line:
                data["UTC"] = extract_datetime_from_line(line)
                 
    return data

# Function to convert degrees to radians
def deg_to_rad(degrees):
    return degrees * math.pi / 180.0

# Function to process stars and append to a single CSV file

def process_stars(
  subdir,
  aircraft,
  heading_deg,
  lat_deg,
  lon_deg,
  ac_datetime,
  stars,
  writer
):
    ac_path = os.path.join(subdir, aircraft)

    for constellation_name, star_name, ra_deg, dec_deg in stars:

        # Convert RA and Dec to Az and El coordinates using Vallado's radec2azel function 
        # converted to Python by Michael Hirsch, Ph.D. and incorporated into pymap3d

        azdeg, eldeg = radec2azel(ra_deg, dec_deg, lat_deg, lon_deg, ac_datetime)

        # Adjust azimuth by the heading to get the look angle
        look_deg = azdeg - heading_deg
    
        # Normalize the look angle to be within [-180, 180]
        # This ensures positive values for clockwise and negative for counterclockwise directions
        look_deg = (look_deg + 180) % 360 - 180
 
        writer.writerow([
          ac_path, star_name, 
          look_deg, eldeg]
        )

# Main function to execute the process
def main():
    with open("star_output.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if os.stat("star_output.csv").st_size == 0:
            writer.writerow(["AC Path", "Star","Look (deg)"," El (deg)"])

        for subdir, _, files in os.walk('.'):
            aircrafts = [file for file in files if file.startswith("ACA") and file.endswith(".csv")]

            for aircraft in aircrafts:
                print(f"Processing stellar data for: {aircraft}")
                data_aircraft = extract_data_from_file(os.path.join(subdir, aircraft))
                r_ac_ecef = [data_aircraft["X"], data_aircraft["Y"], data_aircraft["Z"]]
                v_ac_ecef = [data_aircraft["VX"], data_aircraft["VY"], data_aircraft["VZ"]]
                r_ac_eci = [data_aircraft["Xeci"], data_aircraft["Yeci"], data_aircraft["Zeci"]]
                v_ac_eci = [data_aircraft["VXeci"], data_aircraft["VYeci"], data_aircraft["VZeci"]]
                # Extract and Convert latitude and longitude from degrees to radians
                lat_deg = data_aircraft["LAT"]
                lon_deg = data_aircraft["LON"]
                # Extract altitude (km)
                alt = data_aircraft["ALT"]

                # Extract formatted datetime
                ac_datetime = data_aircraft["UTC"]
                # Print the datetime object
                print("Datetime object:", ac_datetime)
                # Print the type of the datetime object to confirm
                print("Type of object:", type(ac_datetime))

                lat_rad = deg_to_rad(lat_deg)
                lon_rad = deg_to_rad(lon_deg)

                heading_rad = calculate_heading_from_velocity(v_ac_ecef, lat_rad, lon_rad)
                heading_deg = np.degrees(heading_rad)

                process_stars(subdir, aircraft, heading_deg, lat_deg, lon_deg, ac_datetime, stars, writer)

if __name__ == "__main__":
    main()
