clc;
clear;

%% File details
filename  = 'real_time_electricity_prices_24h.xlsx';
sheetName = 1;          % Can also use sheet name as string
rtpRange  = 'C2:C25';   % RTP values (24 hours)

%% Read 24-hour RTP signal from Excel
rtp_price = readmatrix(filename, 'Sheet', sheetName, 'Range', rtpRange);

%% Markup factor (example: 30% markup)
markup_factor = 0.30;

%% Call the function to compute selling price
selling_price = computeSellingPrice(rtp_price, markup_factor);

%% Write selling price back to Excel in a new column
% Example: Column B
outputRange = 'D2:D25';
writematrix(selling_price, filename, 'Sheet', sheetName, 'Range', outputRange);

disp('Selling price successfully written to Excel file.');
