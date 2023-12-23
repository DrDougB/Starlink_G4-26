% John Stilley, Nick Snell
% Starlink Data Conversion 
% November 29th, 2023

% Reads data from STK spreadsheet
stkRaw = readtable('AER Data For Satellites.xlsx');

% Converts data from table to useable type
stkData = table2array(stkRaw);

% Determines size of data array
[row,col] = size(stkData);

% Converts degrees to radians from azimuth and elevation angles for use in
% sph2cart equation
stkData(:, 2) = deg2rad(stkData(:, 2));
stkData(:, 3) = deg2rad(stkData(:, 3));

% Creates empty array to store cartesian data
convertData = zeros(row, 3);

% Converts AER values to XYZ coordinates for use in Blender model
for i= 1:row
    [convertData(i,1), convertData(i,2), convertData(i,3)] = sph2cart(stkData(i, 2), stkData(i, 3), stkData(i, 4));
end
convertData = convertData./10;
% Writes converted data to csv file
writematrix(convertData, 'XYZSatData1.csv')