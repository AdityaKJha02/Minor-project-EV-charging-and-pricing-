function portStats = computeChargingPorts(stationStats, maxPortHours)
% =========================================================
% COMPUTE NUMBER OF CHARGING PORTS PER EVFCS
%
% INPUTS:
%   stationStats  -> output from simulateStationSalesAndTime
%   maxPortHours  -> max allowed usage per port (e.g. 20 or 22)
%
% OUTPUT:
%   portStats     -> struct + table-ready fields
% =========================================================

numStations = numel(stationStats);

portStats = struct([]);

for s = 1:numStations

    % ---------------------------------------------
    % Total charging time
    % ---------------------------------------------
    totalTime_min = stationStats(s).time_min;
    totalTime_hr  = totalTime_min / 60;

    % ---------------------------------------------
    % Number of ports (core logic)
    % ---------------------------------------------
    numPorts = max(1, ceil(totalTime_hr / maxPortHours));

    % ---------------------------------------------
    % Per-port usage
    % ---------------------------------------------
    avgPortUsage_hr = totalTime_hr / numPorts;

    % ---------------------------------------------
    % Utilization (% of 24h per port)
    % ---------------------------------------------
    utilization_pct = (avgPortUsage_hr / 24) * 100;

    % ---------------------------------------------
    % Store results
    % ---------------------------------------------
    portStats(s).stationID = s;
    portStats(s).edge      = stationStats(s).edge;
    portStats(s).numPorts  = numPorts;
    portStats(s).totalTime_hr = totalTime_hr;
    portStats(s).avgPortUsage_hr = avgPortUsage_hr;
    portStats(s).utilization_pct = utilization_pct;
end

end
