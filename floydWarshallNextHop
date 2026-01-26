function [dist, nextHop] = floydWarshallNextHop(dist)
%FLOYDWARSHALLNEXTHOP All-pairs shortest path (Floyd–Warshall)
% with next-hop matrix for path reconstruction
%
% INPUT:
%   dist    - NxN distance matrix
%             (Inf = no edge, 0 on diagonal)
%
% OUTPUT:
%   dist    - shortest-path distance matrix
%   nextHop - next-hop matrix
minSOC          = 60;
maxSOC          = 85;
consRate        = 1;     % %SOC per "distance unit"
chargeThreshold = 55;
stopSOC         = 20;



numStations_req = 6;        % number of stations desired
maxGreedyIter    = 1000;


% --- Time resolution: 1-hour slots, charging continuous (minutes)
slotDur_min = 60;          % 1 hour per slot
hoursInDay  = 24;
slotsPerDay = hoursInDay;  % exactly 24 slots
plotPoints  = slotsPerDay; % show all 24 hours


% --- Charger hardware
battery_kWh         = 59;    % Mahindra BE.6
range_km            = 557;
numAC               = 0;     % AC ports per station
numDC               = 5;     % DC ports per station
DC_power            = 75;    % kW (nameplate)
full_charge_minutes = 35;    % DC taper 0->100% in 35 min
kmPerUnit           = 2;     % km / "distance unit"
speed_kmh           = 60;    % driving speed for arrival time



%% -------------------- Read required files --------------------
fprintf('Reading input files...\n');

% distance matrix
M = readmatrix('distance_matrix.xlsx');
M = M(~all(isnan(M),2), ~all(isnan(M),1));
M(isnan(M)) = Inf;
n = size(M,1);
M(1:n+1:end) = 0;
offdiag = true(n); offdiag(1:n+1:end) = false;
M(offdiag & M==0) = Inf;
M = min(M, M.');
M(1:n+1:end) = 0;
dist = M;  % graph units
    n = size(dist,1);

    % Initialize nextHop
    nextHop = zeros(n);
    for i = 1:n
        for j = 1:n
            if i ~= j && isfinite(dist(i,j))
                nextHop(i,j) = j;
            end
        end
    end

    % Floyd–Warshall core
    for k = 1:n
        dik = dist(:,k);
        skj = dist(k,:);
        for i = 1:n
            if ~isfinite(dik(i)), continue; end
            for j = 1:n
                if ~isfinite(skj(j)), continue; end
                newDist = dik(i) + skj(j);
                if newDist < dist(i,j)
                    dist(i,j)    = newDist;
                    nextHop(i,j) = nextHop(i,k);
                end
            end
        end
    end
end
