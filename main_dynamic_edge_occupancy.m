clc; clear; close all;

%% ================= USER INPUTS =================
maxEVs    = 10;        % max EVs per TASK per HOUR
kmPerUnit = 2;        % km per distance unit
speed_kmh = 60;       % EV speed
numHours  = 24;

taskPaths = {
    [16,8,6,2,1]
    [16,10,11,12]
    [20,18,7,8,6,2,1]
    [20,21,24,13,12]
    [21,24,13,12,3,1]
    [21,24,13,12]
    [7,8,6,2]
    [16,10,11,12,13]
    [8,6,2]
    [18,20,21,24,13]
    [14,11,4,5,6,2]
    [20,21,24,13]
    [15,10,16,8,6,2]
    [6,5,4,11,14]
    [16,8,6,2]
    [4,5,9,10,15,22,21]
    [17,16,8,6,2]
    [6,8,7,18,20,21]
    [19,17,16,8,6,2]
    [1,3,12,13,24,21,22]
    [20,18,7,8,6,2]
    [2,6,8,16,10,15,22]
    [21,20,18,7,8,6]
    [3,12,13,24,21,22]
    [13,24,21,20,18,7]
    [4,5,9,10,15,22]
    [16,10,11]
    [6,8,16,10,15,22]
    [17,19,15,22,21,24]
};

%% ================= LOAD DATA =================
vehFlowRatio = readmatrix('24hr_vehicles.xlsx');        % 24×1
edgeMap      = readmatrix('edge_number_mapping.xlsx'); % [EdgeID From To]
distMatrix   = readmatrix('distance_matrix.xlsx');

numEdges = size(edgeMap,1);

%% ============ FIX DISTANCE MATRIX ============
defaultWeight = 1;
for e = 1:numEdges
    i = edgeMap(e,2);
    j = edgeMap(e,3);
    if distMatrix(i,j) <= 0
        distMatrix(i,j) = defaultWeight;
    end
end

%% ======== TIME-CONSISTENT EDGE OCCUPANCY (FRACTIONAL) ========
edgeOccupancy = computeEdgeOccupancy( ...
    taskPaths, vehFlowRatio, maxEVs, ...
    edgeMap, distMatrix, kmPerUnit, speed_kmh, numHours);

%% ======== TOTAL-PRESERVING ROUNDING (PER EDGE) ========
edgeOccupancyRounded = totalPreservingHourlyEVs(edgeOccupancy);

%% ======== SANITY CHECK (OPTIONAL BUT RECOMMENDED) ========
for e = 1:numEdges
    if round(sum(edgeOccupancy(e,:))) ~= sum(edgeOccupancyRounded(e,:))
        error('Total EV mismatch after rounding at edge %d', e);
    end
end
disp('✔ All edge totals preserved after rounding');

%% ================= SAVE =================
hourNames = "H" + string(1:numHours);
EdgeIDCol = table(edgeMap(:,1),'VariableNames',{'EdgeID'});
OccTable  = array2table(edgeOccupancyRounded,'VariableNames',hourNames);

writetable([EdgeIDCol OccTable], ...
    'EdgeID_Hourly_EV_Occupancy_ROUNDED.xlsx');

disp('================================================');
disp('TIME-CONSISTENT EDGE OCCUPANCY COMPLETED');
disp('Hourly values rounded to integers');
disp('Total EVs per edge preserved');
disp('================================================');



