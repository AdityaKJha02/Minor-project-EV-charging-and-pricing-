function fixedCost = Cd_fixcost_FPM(batterySize, N_co, beta_SS, N_pv)
% =====================================================
% FIXED COST (Cdfix) COMPUTATION FOR EACH FCS (Fig. 7)
%
% INPUTS:
%   batterySize : [1×nFCS] vector (kWh)
%   N_co        : [1×nFCS] number of chargers
%   beta_SS     : [1×nFCS] substation capacity (kW)
%   N_pv        : [1×nFCS] number of PV panels
%
% OUTPUT:
%   fixedCost   : [1×nFCS] Rs/day
% =====================================================

%% ---------------------------
% FINANCIAL PARAMETERS
% ---------------------------
r = 0.075;
n = 15;

CRF = (r * (1 + r)^n) / ((1 + r)^n - 1);

%% ---------------------------
% UNIT COSTS
% ---------------------------
U_L    = 40000/(12*n);      % Rs/m^2/day
U_BDG  = 100000/(12*n);    % Rs/charger/day
U_SS   = 200000/(12*n);    % Rs/kW/day
U_SPV  = 17000;            % Rs / PV panel
U_BESS = 15000;            % Rs / kWh

nFCS = numel(N_co);

%% ---------------------------
% AREA
% ---------------------------
A_fc = zeros(1,nFCS);
for i = 1:nFCS
    A_fc(i) = Area_calc(N_co(i));
end

%% ---------------------------
% FIXED COST (Eq. 35)
% ---------------------------
fixedCost = zeros(1,nFCS);

for i = 1:nFCS
    fixedCost(i) = (1/30) * CRF * ( ...
        A_fc(i)        * U_L   + ...
        N_co(i)        * U_BDG + ...
        beta_SS(i)     * U_SS  + ...
        N_pv(i)        * U_SPV + ...
        batterySize(i) * U_BESS );
end

end
