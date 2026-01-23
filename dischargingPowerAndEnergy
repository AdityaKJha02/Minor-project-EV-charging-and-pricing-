function [timeVec_min, SOC_vec, P_kW_vec, E_kWh_total, P_kW_at_SOC] = ...
    dischargingPowerAndEnergy(SOC_initial, SOC_final, battery_kWh, SOC_query)
% =========================================================
% DISCHARGING POWER & ENERGY MODEL (POSITIVE OUTPUT)
%
% Optional:
%   SOC_query     - SOC (%) at which power is required
% =========================================================

%% SOC–time curve from paper (Fig. 3)
socPts = [20 50 75 85 95 100];     % %
tPts   = [0  10 15 22 34  60];     % minutes

%% Input validation
SOC_initial = min(max(SOC_initial,20),100);
SOC_final   = min(max(SOC_final,20),SOC_initial);

%% Absolute times on curve
t_start = interp1(socPts, tPts, SOC_initial, 'linear');
t_end   = interp1(socPts, tPts, SOC_final,   'linear');
totalTime = t_start - t_end;

%% Time discretization
dt = 10;   % minutes
timeVec_min = 0:dt:totalTime;
if timeVec_min(end) < totalTime
    timeVec_min(end+1) = totalTime;
end

%% SOC trajectory
t_abs = t_start - timeVec_min;
SOC_vec = interp1(tPts, socPts, t_abs, 'linear');

%% Segment-wise discharging power (magnitude)
nSeg = numel(socPts)-1;
Pseg_kW = zeros(nSeg,1);

for i = 1:nSeg
    dSOC   = socPts(i+1) - socPts(i);
    dtSeg = tPts(i+1) - tPts(i);
    slope = dSOC / dtSeg;                 % %/min
    Pseg_kW(i) = battery_kWh * (slope/100) * 60;
end

%% Power trajectory (positive)
P_kW_vec = zeros(size(timeVec_min));

for k = 1:numel(timeVec_min)
    soc_now = SOC_vec(k);
    idx = find(socPts >= soc_now, 1, 'first') - 1;
    idx = max(1, min(idx, nSeg));
    P_kW_vec(k) = Pseg_kW(idx);
end

%% Energy calculation (POSITIVE)
E_kWh_total = (SOC_initial - SOC_final)/100 * battery_kWh;

%% Power at requested SOC (optional)
P_kW_at_SOC = [];

if nargin == 4
    SOC_query = min(max(SOC_query,20),100);
    idx = find(socPts >= SOC_query, 1, 'first') - 1;
    idx = max(1, min(idx, nSeg));
    P_kW_at_SOC = Pseg_kW(idx);
end

end
