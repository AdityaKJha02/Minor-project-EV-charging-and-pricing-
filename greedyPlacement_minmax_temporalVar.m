function [chosenEdges, history] = greedyPlacement_minmax_temporalVar( ...
    candidateEdges, taskPaths, OD, dist, ...
    consRate, numEVs, minSOC, maxSOC, chargeThreshold, stopSOC, ...
    numStations, wCharge, wStop, wVar, maxGreedyIter, battery_kWh, ...
    baseLoad24h, edgeToBus)

% GREEDY EVFCS PLACEMENT WITH MIN–MAX NORMALISATION (3 OBJECTIVES)

epsNorm = 1e-9;

remainingEdges = candidateEdges;
chosenEdges = zeros(0,2);
iter = 0;

history.C = {};
history.S = {};
history.V = {};
history.Cn = {};
history.Sn = {};
history.Vn = {};
history.score = [];
history.edge = {};

fprintf('\n[Greedy-MinMax-3OBJ] Starting greedy placement\n');

while size(chosenEdges,1) < numStations && ...
      ~isempty(remainingEdges) && ...
      iter < maxGreedyIter

    iter = iter + 1;
    nCand = size(remainingEdges,1);

    C = zeros(nCand,1);
    S = zeros(nCand,1);
    V = zeros(nCand,1);

    for e = 1:nCand
        trialEdges = [chosenEdges; remainingEdges(e,:)];

        [C(e), S(e), V(e)] = simulateWithStationsTemporalVar( ...
            trialEdges, taskPaths, OD, dist, ...
            consRate, numEVs, minSOC, maxSOC, ...
            chargeThreshold, stopSOC, battery_kWh, ...
            baseLoad24h, edgeToBus);
    end

    % --- MIN–MAX NORMALISATION ---
    Cn = minmaxNorm(C, epsNorm);
    Sn = minmaxNorm(S, epsNorm);
    Vn = minmaxNorm(V, epsNorm);

    score = wCharge * Cn;

    [bestScore, idx] = max(score);
    bestEdge = remainingEdges(idx,:);

    history.C{iter}  = C;
    history.S{iter}  = S;
    history.V{iter}  = V;
    history.Cn{iter} = Cn;
    history.Sn{iter} = Sn;
    history.Vn{iter} = Vn;
    history.score(iter) = bestScore;
    history.edge{iter}  = bestEdge;

    fprintf(['Selected station %d: edge %d->%d | ', ...
             'trialCharge=%.2f trialStops=%d trialVar=%.2f score=%.4f\n'], ...
        iter, bestEdge(1), bestEdge(2), ...
        C(idx), S(idx), V(idx), bestScore);

    chosenEdges = [chosenEdges; bestEdge];
    remainingEdges(idx,:) = [];
end

fprintf('[Greedy-MinMax-3OBJ] Finished. Selected %d stations.\n', size(chosenEdges,1));
end

% ========================================================
% HELPERS
% ========================================================
function xnorm = minmaxNorm(x, epsNorm)
xmin = min(x);
xmax = max(x);
if abs(xmax - xmin) < epsNorm
    xnorm = zeros(size(x));
else
    xnorm = (x - xmin) ./ (xmax - xmin);
end
end

% ========================================================
% TEMPORAL VAR SIMULATION
% ========================================================
function [totalCharge_kWh, totalStops, busVar] = ...
    simulateWithStationsTemporalVar(stationEdges, taskPaths, OD, dist, ...
        consRate, numEVs, minSOC, maxSOC, ...
        chargeThreshold, stopSOC, battery_kWh, ...
        baseLoad24h, edgeToBus)

K = size(stationEdges,1);
totalCharge_pct = 0;
totalStops = 0;
stationEnergyPerHour = zeros(K,24);

if K == 0
    totalCharge_kWh = 0;
    busVar = 0;
    return;
end

keys = arrayfun(@(i) sprintf('%d-%d',stationEdges(i,1),stationEdges(i,2)), ...
                1:K,'UniformOutput',false);

% --- Target SOC distribution (95% in [75,80])
muTargetSOC    = 77.5;
sigmaTargetSOC = (80 - 75) / (2*1.96);

for hr = 1:24
    numEVs_hr = numEVs(hr);
    if numEVs_hr == 0, continue; end

    for t = 1:numel(taskPaths)
        p = taskPaths{t};
        if isempty(p), continue; end

        % Initial SOC (unchanged)
        % -------- INITIAL SOC: 95% inside [minSOC,maxSOC], 5% outside --------

soc = zeros(numEVs_hr,1);

% Decide which EVs are in the main 95%
isMain95 = rand(numEVs_hr,1) <= 0.95;

% ---- 95% vehicles: SOC ~ N(mean, sigma) clipped to [minSOC,maxSOC]
muSOC    = (minSOC + maxSOC)/2;
sigmaSOC = (maxSOC - minSOC)/(2*1.96);

soc95 = muSOC + sigmaSOC*randn(sum(isMain95),1);
soc95 = min(max(soc95, minSOC), maxSOC);
soc(isMain95) = soc95;

% ---- 5% vehicles: SOC outside range
nOut = sum(~isMain95);

if nOut > 0
    % Half below minSOC, half above maxSOC
    below = rand(nOut,1) < 0.5;

    socOut = zeros(nOut,1);

    % Below minSOC (e.g., minSOC-5 to minSOC)
    socOut(below) = minSOC - 5*rand(sum(below),1);

    % Above maxSOC (e.g., maxSOC to maxSOC+5)
    socOut(~below) = maxSOC + 5*rand(sum(~below),1);

    soc(~isMain95) = socOut;
end
% -------------------------------------------------------------------


        stopped = false(numEVs_hr,1);

        for k = 1:(numel(p)-1)
            u = p(k); 
            v = p(k+1);
            d = dist(u,v);

            idx = find(strcmp(keys,sprintf('%d-%d',u,v)) | ...
                       strcmp(keys,sprintf('%d-%d',v,u)),1);

            % --------- CHANGED PART (FINAL SOC LOGIC) ---------
            % --------- FINAL SOC LOGIC: 95% inside [75,80], 5% outside ---------
if ~isempty(idx)

    mask = soc < chargeThreshold;
    nCharge = sum(mask);

    if nCharge > 0

        % Decide which EVs belong to the 95%
        isMain95 = rand(nCharge,1) <= 0.95;

        targetSOC = soc(mask);   % default = no extra charge

        % ---- 95% vehicles: SOC ~ N(77.5, σ) clipped to [75,80]
        muTargetSOC    = 77.5;
        sigmaTargetSOC = (80 - 75) / (2*1.96);

        if any(isMain95)
            soc95 = muTargetSOC + sigmaTargetSOC*randn(sum(isMain95),1);
            soc95 = min(max(soc95,75),80);
            targetSOC(isMain95) = soc95;
        end

        % ---- 5% vehicles: charge higher than 80 OR just enough
        if any(~isMain95)
            soc5 = 80 + 5*rand(sum(~isMain95),1);  % e.g., 80–85%
            soc5 = min(soc5, maxSOC);
            targetSOC(~isMain95) = soc5;
        end

        % Energy calculation
        deltaSOC = max(0, targetSOC - soc(mask));
        soc(mask) = targetSOC;

        stationEnergyPerHour(idx,hr) = ...
            stationEnergyPerHour(idx,hr) + ...
            sum(deltaSOC)/100 * battery_kWh;

        totalCharge_pct = totalCharge_pct + sum(deltaSOC);
    end
end
% -----------------------------------------------------------------

            % -------------------------------------------------

            % Driving consumption (unchanged)
            soc = soc - d*consRate;

            newStops = (soc < stopSOC) & ~stopped;
            totalStops = totalStops + sum(newStops);
            stopped(newStops) = true;
            soc(stopped) = stopSOC;
        end
    end
end

totalCharge_kWh = totalCharge_pct/100 * battery_kWh;

perStationVar = var(stationEnergyPerHour,0,2);
busVar = mean(perStationVar);

end

