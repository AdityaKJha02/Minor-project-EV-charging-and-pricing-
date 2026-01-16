clc;
clear;

%% File details
filename  = 'RTP_grid_cost_10min.xlsx';
sheetName = 1;          % Can also use sheet name as string
rtpRange  = 'B2:B151';   % RTP values (24 hours)

%% Read 24-hour RTP signal from Excel
rtp_price = readmatrix(filename, 'Sheet', sheetName, 'Range', rtpRange);

%% Markup factor (example: 30% markup)
markup_factor = 0.30;

%% Call the function to compute selling price
selling_price = computeSellingPrice(rtp_price, markup_factor);

%% Write selling price back to Excel in a new column
% Example: Column B
outputRange = 'C2:C151';
writematrix(selling_price, filename, 'Sheet', sheetName, 'Range', outputRange);

disp('Selling price successfully written to Excel file.');
