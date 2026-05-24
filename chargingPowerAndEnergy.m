function [timeVec_min, SOC_vec, P_kW_vec, E_kWh_total] = ...
    chargingPowerAndEnergy(SOC1, SOC2, battery_kWh)
% =========================================================
% chargingPowerAndEnergy
%
% Time starts at 0 minutes for SOC1
% Records charging power at 2-minute intervals
% Computes total energy between SOC1 and SOC2
% =========================================================

% -------- Charging curve (SOC vs absolute time) --------
socPts = [20 50 75 85 95 100];     % %
tPts   = [0  10 15 22 34  60];     % minutes

% Clamp SOCs
SOC1 = max(min(SOC1,100),20);
SOC2 = max(min(SOC2,100),SOC1);

% -------- Absolute times on curve --------
t_abs_start = interp1(socPts, tPts, SOC1, 'linear');
t_abs_end   = interp1(socPts, tPts, SOC2, 'linear');

% -------- Relative time (START AT ZERO) --------
dt = 10; % minutes
timeVec_min = 0 : dt : (t_abs_end - t_abs_start);
if timeVec_min(end) < (t_abs_end - t_abs_start)
    timeVec_min(end+1) = (t_abs_end - t_abs_start);
end

% -------- Map relative time → absolute time --------
t_abs = t_abs_start + timeVec_min;

% -------- SOC trajectory --------
SOC_vec = interp1(tPts, socPts, t_abs, 'linear');

% -------- Segment powers --------
nSeg = numel(socPts) - 1;
Pseg_kW = zeros(nSeg,1);

for i = 1:nSeg
    dSOC = socPts(i+1) - socPts(i);
    dtSeg = tPts(i+1) - tPts(i);
    socSlope = dSOC / dtSeg;                 % % per min
    Pseg_kW(i) = battery_kWh * (socSlope/100) * 60;
end

% -------- Power at each step --------
P_kW_vec = zeros(size(timeVec_min));

for k = 1:numel(timeVec_min)
    soc_now = SOC_vec(k);
    segIdx = find(socPts <= soc_now, 1, 'last');
    if segIdx == numel(socPts)
        segIdx = segIdx - 1;
    end
    P_kW_vec(k) = Pseg_kW(segIdx);
end

% -------- Energy calculation --------
E_kWh_total = sum(P_kW_vec(1:end-1) .* (dt/60));

end
