# 
# Batch SOAP exported Text-to-CSV Converter in Sub-Directories
# 
# Dr. Douglas Buettner generated this text in part with GPT-4, OpenAI(TM) large-scale 
# language-generation model. Upon generating draft language, the author reviewed, edited, 
# and revised the content to his own liking and takes ultimate responsibility for the 
# content of this code.
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
# Description:
# This script identifies sub-directories in the current directory that contain specific files 
# and then converts all `.txt` files in these sub-directories to `.csv` format. The specific criteria for 
# sub-directory selection are:
# 1. Contains at least one file that starts with "ACA".
# 2. Contains at least one file that starts with "sun".
# 3. Contains one or more files that start with numerical digits.
# 
# For the conversion:
# 1. Remove any existing commas from each line.
# 2. For files with "ACA" or "sun" in their names, remove colons between variable names and numerical values.
# 3. Remove leading and trailing whitespaces from each line.
# 4. Split the line by spaces to identify individual items.
# 5. Write the items to a new `.csv` file using commas as separators.
# 
# Requirements provided by Dr. Douglas Buettner.
# 

import csv
import os
import re

def should_process_directory(directory):
    # Check if the directory meets the criteria for processing.

    has_ACA, has_sun, has_numerical = False, False, False

    for filename in os.listdir(directory):
        if filename.startswith("ACA"):
            has_ACA = True
        elif filename.startswith("sun"):
            has_sun = True
        elif filename[0].isdigit():
            has_numerical = True

        if has_ACA and has_sun and has_numerical:
            return True

    return False

def parse_line(line, remove_colon=False):
    # Parse a line from the text file.

    line_without_commas = line.replace(',', '')

    if remove_colon:
       line_without_commas = re.sub(r'\s*([^:\s]+) :\s*', r'\1 ', line_without_commas)

    return line_without_commas.strip().split()

def convert_txt_to_csv(txt_path):
    # Convert a single .txt file to .csv format.

    csv_path = txt_path.rsplit('.', 1)[0] + '.csv'
    
    remove_colon = "ACA" in txt_path or "sun" in txt_path

    with open(txt_path, 'r') as txt_file, open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for line in txt_file:
            items = parse_line(line, remove_colon)
            writer.writerow(items)

# Start from the current directory
current_directory = os.getcwd()

# Find all sub-directories in the current directory
subdirs = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

# Process only those sub-directories that meet the criteria
for subdir in subdirs:
    full_subdir_path = os.path.join(current_directory, subdir)

    if should_process_directory(full_subdir_path):
        for filename in os.listdir(full_subdir_path):
            if filename.endswith(".txt"):
                txt_path = os.path.join(full_subdir_path, filename)
                convert_txt_to_csv(txt_path)

print("Batch conversion in sub-directories complete!")
