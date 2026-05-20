clc;
clear;
close all;

% File name
filename = 'FPM_BESS_Results.xlsx';

% Read data (assuming data is in first sheet)
data = readmatrix(filename);

% Extract columns
BESS_kWh = data(:,1);   % X-axis
EV_price = data(:,2);   % Y-axis

% Plot
figure;
plot(BESS_kWh, EV_price, '-o', ...
    'LineWidth', 2);
%y = linspace(11.8,11.92,0.01)
% Labels and title
xlabel('BESS Capacity in FPM (kWh)');
ylabel('Price (Rs.)');
title('BESS Capacity in FPM vs Energy Price');

% Grid
%grid on;