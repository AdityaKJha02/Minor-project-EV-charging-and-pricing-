clc;
clear;

% =====================================================
% HOURLY INPUT DATA
% =====================================================
markup_factor=0.30

EVH_filename='EVFCS_10min_Arrival_Power'

filename  = 'RTP_grid_cost_10min.xlsx';
sheetName1 = 'EVFCS_1_6_8';              % Sheet index or name
sheetName2 = 'EVFCS_2_12_13';
sheetName3 = 'EVFCS_3_21_20';
sheetName4 = 'EVFCS_4_6_5';
sheetName5 = 'EVFCS_5_6_2';
sheetName6 = 'EVFCS_6_11_12';

rtpRange='B2:B151';
EhevRange  = 'H2:H151';       % RTP values (24 hours)

%% =====================================================
% READ 24-HOUR RTP SIGNAL FROM EXCEL
% =====================================================
sheetName=1;

rho_h=  readmatrix(filename, 'Sheet', sheetName, 'Range', rtpRange);
Ehev1 = readmatrix(EVH_filename, 'Sheet', sheetName1, 'Range', EhevRange);
Ehev2 = readmatrix(EVH_filename, 'Sheet', sheetName2, 'Range', EhevRange);
Ehev3 = readmatrix(EVH_filename, 'Sheet', sheetName3, 'Range', EhevRange);
Ehev4 = readmatrix(EVH_filename, 'Sheet', sheetName4, 'Range', EhevRange);
Ehev5 = readmatrix(EVH_filename, 'Sheet', sheetName5, 'Range', EhevRange);
Ehev6 = readmatrix(EVH_filename, 'Sheet', sheetName6, 'Range', EhevRange);
selling_price = computeSellingPrice(rho_h, markup_factor);
% Ensure all are column vectors
Ehev1 = Ehev1(:);
Ehev2 = Ehev2(:);
Ehev3 = Ehev3(:);
Ehev4 = Ehev4(:);
Ehev5 = Ehev5(:);
Ehev6 = Ehev6(:);

% Combine into a matrix (each column = one EVCS)
Ehev = [Ehev1, Ehev2, Ehev3, Ehev4, Ehev5, Ehev6];

% =====================================================
% FINANCIAL PARAMETERS
% =====================================================
r = 0.075;     % 7.5% interest rate
n = 15;        % Loan duration (years)

% =====================================================
% BCS DATA (6 STATIONS)
% =====================================================
N_co = [5, 2, 1, 1, 1, 2];                 % No. of chargers
beta_SS = [424.8, 283.2, 177, 106.2, 212.4, 150];  % Peak load (kW)

% Area calculation
A_fc = zeros(1, length(N_co));
for i = 1:length(N_co)
    A_fc(i) = Area_calc(N_co(i));
end

% =====================================================
% CONSTANT COST PARAMETERS
% =====================================================
U_L   = 40000/(12*n);    % Rs/m^2
U_BDG = 100000/(12*n);   % Rs/charger
U_SS  = 200000/(12*n);    % Rs/kW


% =====================================================
% DAILY COST CALCULATION (LOOP)
% =====================================================
Cd = zeros(1, length(N_co));

for i = 1:length(N_co)

    params.A_fc    = A_fc(i);
    params.N_CO    = N_co(i);
    params.beta_SS = beta_SS(i);

    params.U_L   = U_L;
    params.U_BDG = U_BDG;
    params.U_SS  = U_SS;

    Cd(i) = daily_cost_bcs(Ehev(:,i), rho_h, r, n, params);

end
% =====================================================
% DAILY REVENUE CALCULATION (LOOP)
% =====================================================
RD = zeros(1, length(N_co));

for i = 1:length(N_co)
    RD(i) = revenue_bcs(Ehev(:,i), rho_h, markup_factor);
end

% =====================================================
% DISPLAY REVENUE RESULTS
% =====================================================
disp('Daily revenue of each EVFCS (Rs/day):')
disp(RD)

% Total revenue
Total_Revenue = sum(RD);
disp(['Total daily revenue (all EVFCS) = ', num2str(Total_Revenue), ' Rs/day']);


% =====================================================
% DAILY PROFIT MARGIN CALCULATION (OPTIONAL)
% =====================================================
PM = zeros(1, length(N_co));

for i = 1:length(N_co)
    PM(i) = Profitm_bcs(RD(i), Cd(i));
end

% =====================================================
% DISPLAY PROFIT MARGIN
% =====================================================
disp('Daily profit margin of each EVFCS (%)');
disp(PM);

% =====================================================
% DISPLAY RESULTS
% =====================================================
disp('Daily cost of each BCS (Rs/day):')
disp(Cd)
disp(rho_h)


% =====================================================
% GENERATE EXCEL FILE WITH Cd, REVENUE & PROFIT MARGIN
% 6 SHEETS (ONE PER EVFCS)
% =====================================================

outputFile = 'EVFCS_Daily_Cost_Revenue_Profit.xlsx';

for i = 1:length(N_co)

    % ---------------------------------
    % Calculate profit margin
    % ---------------------------------
    PM = Profitm_bcs(RD(i), Cd(i));

    % ---------------------------------
    % Create table for EVFCS i
    % ---------------------------------
    EVFCS_ID          = i;
    No_of_Chargers    = N_co(i);
    Daily_Cost_Rs    = Cd(i);
    Daily_Revenue_Rs = RD(i);
    Profit_Margin_percent = PM;

    Results_Table = table( ...
        EVFCS_ID, ...
        No_of_Chargers, ...
        Daily_Cost_Rs, ...
        Daily_Revenue_Rs, ...
        Profit_Margin_percent, ...
        'VariableNames', ...
        {'EVFCS_ID', 'No_of_Chargers', ...
         'Daily_Cost_Rs', 'Daily_Revenue_Rs', ...
         'Profit_Margin_percent'} ...
    );

    % ---------------------------------
    % Sheet name
    % ---------------------------------
    sheetName = ['EVFCS_', num2str(i)];

    % ---------------------------------
    % Write to Excel
    % ---------------------------------
    writetable(Results_Table, outputFile, 'Sheet', sheetName);

end

disp('--------------------------------------------------');
disp('Excel file generated successfully!');
disp(['File name: ', outputFile]);
disp('Sheets: EVFCS_1 to EVFCS_6');
disp('--------------------------------------------------');

