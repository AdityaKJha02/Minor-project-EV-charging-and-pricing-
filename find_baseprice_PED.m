function OUT = dpm_revenue_energy_block( ...
        base_price, ...
        timeGrid_min, ...
        EV_file, RTP_file, ...
        sheetNames_EV, ...
        TEMP_BSA_file, PED_excel, ...
        battery_kWh, N_co, beta_SS, N_pv)

fprintf('\n==========================================\n');
fprintf('DPM PIPELINE @ Base price = %.2f Rs/kWh\n',base_price);
fprintf('==========================================\n');

nFCS = numel(sheetNames_EV);

%% ======================================================
% STEP 0 — FIXED COSTS
%% ======================================================
battery_vec = battery_kWh * ones(1,nFCS);

C_FIX_vec   = Cd_fixcost_FPM(battery_vec,N_co,beta_SS,N_pv);
C_OandM_vec = Cd_OandM_FPM(C_FIX_vec);

C_FIX   = sum(C_FIX_vec);
C_OandM = sum(C_OandM_vec);

fprintf('\nSTEP 0\n');
fprintf('TOTAL Fixed Cost = %.4f Rs\n',C_FIX);
fprintf('TOTAL O&M Cost   = %.4f Rs\n',C_OandM);

%% ======================================================
% STEP 1 — GENERATE PED EXCEL
%% ======================================================
fprintf('\nSTEP 1: Generating PED Excel\n');

if exist(PED_excel,'file')
    delete(PED_excel);
end

for fcsID = 1:nFCS

    [~,~,E_EV_DPM,~,price_h,~] = ...
        compute_PED_DPM( ...
        timeGrid_min, RTP_file, ...
        EV_file, sheetNames_EV, fcsID, ...
        base_price);


    Revenue_DPM = price_h .* E_EV_DPM;
    TotalRevenue = sum(Revenue_DPM);

    T = table(price_h, Revenue_DPM, ...
        'VariableNames',{'Price','Revenue_Rs'});

    sheet = sprintf('FCS_%d',fcsID);

    writetable(T,PED_excel,'Sheet',sheet);
    writematrix(TotalRevenue,PED_excel,'Sheet',sheet,'Range','K2');

    fprintf('FCS %d Revenue(K2)=%.4f Rs\n',fcsID,TotalRevenue);
end

%% ======================================================
% STEP 2 & 3 — REVENUE + PDT
%% ======================================================
TotalRevenue_all = 0;
EnergyCost_all  = 0;

fprintf('\nSTEP 2–3\n');

% ---- Read RTP purchase price (column 2) ----
RTP = readmatrix(RTP_file);
timeRTP = RTP(:,1);
rho_PUR = RTP(:,2);     % purchase price Rs/kWh

for i = 1:nFCS

    sheet = sprintf('FCS_%d',i);

    % ---------- Revenue from PED excel ----------
    Rev_i = readmatrix(PED_excel,'Sheet',sheet,'Range','K2');
    Rev_i = Rev_i(1);
    TotalRevenue_all = TotalRevenue_all + Rev_i;

    % ---------- Grid energy from BSA ----------
    Mbsa  = readmatrix(TEMP_BSA_file,'Sheet',sheet);
    Egrid = Mbsa(:,9);              % column I
    Egrid = Egrid(~isnan(Egrid));   % remove NaNs

    % ---------- Purchase price vector ----------
    rho_vec = zeros(length(Egrid),1);

    for k = 1:length(Egrid)
        idx = find(timeRTP == timeGrid_min(k),1);
        rho_vec(k) = rho_PUR(idx);
    end

    % ---------- Energy purchase cost ----------
    pdt = sum(rho_vec .* Egrid);
    EnergyCost_all = EnergyCost_all + pdt;

    fprintf('FCS %d | Revenue(K2)=%.4f | pdt=%.4f\n', ...
            i, Rev_i, pdt);
end


%% ======================================================
% STEP 4 — TOTAL COST Cd
%% ======================================================
Cd_total = C_FIX + C_OandM + EnergyCost_all;

fprintf('\nSTEP 4\n');
fprintf('Energy Cost = %.4f Rs\n',EnergyCost_all);
fprintf('TOTAL Cd    = %.4f Rs\n',Cd_total);

%% ======================================================
% STEP 5 — PROFIT MARGIN
%% ======================================================
PM = (TotalRevenue_all - Cd_total) / TotalRevenue_all;

fprintf('\nSTEP 5\n');
fprintf('TOTAL Revenue = %.4f Rs\n',TotalRevenue_all);
fprintf('PROFIT MARGIN = %.4f\n',PM);

%% ======================================================
% OUTPUT
%% ======================================================
OUT.Revenue = TotalRevenue_all;
OUT.Cd      = Cd_total;
OUT.PM      = PM;

end
