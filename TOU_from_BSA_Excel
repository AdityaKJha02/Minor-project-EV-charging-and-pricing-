function TOU = TOU_from_BSA_Excel(params, BSA_excel)

fprintf('\n===== ENTERING TOU_from_BSA_Excel =====\n');

%% ---------------------------
% Unpack parameters
%% ---------------------------
PM_DES  = params.PM_DES;
base0   = params.basePrice0;
baseMax = params.basePriceMax;
dBase   = params.deltaPrice;

rho_PUR = params.rho_PUR(:);
dt      = params.dt;

C_FIX   = params.C_FIX;
C_OandM = params.C_OandM;

RTP_file      = params.RTP_file;
EV_file       = params.EV_file;
sheetNames_EV = params.sheetNames_EV;

tou_prices = params.tou_prices;

eta_EV = 0.92;

C_FIXED_TOTAL = C_FIX + C_OandM;

%% ---------------------------
% Detect FCS sheets
%% ---------------------------
[~, sheetNames] = xlsfinfo(BSA_excel);
nFCS = numel(sheetNames);

E_EV_slot   = cell(1,nFCS);
E_GRID_slot = cell(1,nFCS);

%% =========================================================
% READ BSA DATA
%% =========================================================

fprintf('\n===== READING BSA DATA =====\n');

for i = 1:nFCS

    fprintf('\nReading BSA sheet: %s\n', sheetNames{i});

    M = readmatrix(BSA_excel,'Sheet',sheetNames{i});

    P_EV = M(:,3);
    P_BC = M(:,5);
    P_BD = M(:,6);

    valid = ~isnan(P_EV);

    P_EV = P_EV(valid);
    P_BC = P_BC(valid);
    P_BD = P_BD(valid);

    E_EV_h = (P_EV/eta_EV) * dt;
    E_BC_h = P_BC * dt;
    E_BD_h = P_BD * dt;

    E_GRID_h = E_EV_h + E_BC_h - E_BD_h;

    E_GRID_h(E_GRID_h < 0) = 0;

    E_EV_slot{i}   = E_EV_h;
    E_GRID_slot{i} = E_GRID_h;

end

%% =========================================================
% PRICE ITERATION
%% =========================================================

fprintf('\n===== STARTING BASE PRICE SEARCH =====\n');

basePrice = base0;

while basePrice <= baseMax

    fprintf('\nTesting base price = %.3f\n', basePrice);

    Total_Revenue = 0;
    Total_Cost    = 0;

    for i = 1:nFCS

        %% --------------------------------
        % Generate TOU price profile
        %% --------------------------------
        [~,~,~,~,~,~,price_h] = compute_TOU_tauDriven( ...
            RTP_file, ...
            EV_file, sheetNames_EV, ...
            i, ...
            tou_prices, ...
            basePrice);

        %% Expand 24 → 144 slots
        rhoSELL_vec = repelem(price_h,6);

        %% --------------------------------
        % Revenue
        %% --------------------------------
        E_EV_h = E_EV_slot{i};

        Ng = min(length(rhoSELL_vec), length(E_EV_h));

        Revenue_i = sum(rhoSELL_vec(1:Ng) .* E_EV_h(1:Ng));

        %% --------------------------------
        % Grid energy cost
        %% --------------------------------
        E_GRID_h = E_GRID_slot{i};

        Ng = min(length(rho_PUR), length(E_GRID_h));

        EnergyCost_i = sum(rho_PUR(1:Ng) .* E_GRID_h(1:Ng));

        %% --------------------------------
        % Total cost
        %% --------------------------------
        C_i = C_FIXED_TOTAL(i) + EnergyCost_i;

        Total_Revenue = Total_Revenue + Revenue_i;
        Total_Cost    = Total_Cost + C_i;

    end

    %% --------------------------------
    % Profit Margin
    %% --------------------------------
    if Total_Revenue > 0
        PM = (Total_Revenue - Total_Cost) / Total_Revenue;
    else
        PM = NaN;
    end

    fprintf('\nTotal Revenue = %.2f\n', Total_Revenue);
    fprintf('Total Cost    = %.2f\n', Total_Cost);
    fprintf('Profit Margin = %.4f\n', PM);

    %% Stop condition
    if ~isnan(PM) && PM >= PM_DES
        fprintf('Desired PM reached\n');
        break
    end

    basePrice = basePrice + dBase;

end

%% =========================================================
% OUTPUT
%% =========================================================

TOU.basePrice     = basePrice;
TOU.Total_Revenue = Total_Revenue;
TOU.Total_Cost    = Total_Cost;
TOU.PM            = PM;

fprintf('\n========== TOU RESULTS ==========\n');
fprintf('Optimal base price       : %.3f\n', basePrice);
fprintf('Total revenue            : %.2f\n', Total_Revenue);
fprintf('Total cost               : %.2f\n', Total_Cost);
fprintf('Achieved profit margin   : %.3f\n', PM);

end
